// Copyright (c) 2024, Queen's University Belfast
// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <queue>
#include <vector>

#include "omp.h"
#ifdef PAPI_PROFILE
#include "papi.h"
#endif

#include "benchmark.h"
#include "bucketing/executor.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "leaves.h"
#include "parallel/atomics_array.h"
#include "parallel/padded_array.h"
#include "parallel/vector.h"
#include "platform_atomics.h"
#include "timer.h"

using namespace std;
using namespace bucketing;

static constexpr WeightT DIST_INF = numeric_limits<WeightT>::max() / 2;

template <typename GraphT, bool DIRECTED, bool CACHE_LEAVES>
auto BestDeltaStepping(const WGraph& g, NodeID source, double delta) {
#ifdef PAPI_PROFILE
  PAPI_hl_region_begin("sssp");
#endif

  parallel::atomics_array<WeightT> dist(g.num_nodes(), DIST_INF);
  dist[source] = 0;

  executor scheduler;

  const auto init_sssp = [&](executor::frontier& my_frontier) {
    my_frontier.push(source, 0);
  };

  const auto relax_push = [&](const NodeID u, const WNode wv) -> std::optional<WeightT> { // outgoing edge
    WeightT old_dist = dist[wv.v].load(std::memory_order_acquire);
    WeightT new_dist = dist[u].load(std::memory_order_acquire) + wv.w;
    while (new_dist < old_dist) {
      if (dist[wv.v].compare_exchange_weak(old_dist, new_dist, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return new_dist;
      }
    }
    return std::nullopt;
  };

  const auto relax_pull_safe = [&](WNode wu, NodeID v) -> std::optional<WeightT> { // incoming edge
    WeightT old_dist = dist[v].load(std::memory_order_acquire);
    WeightT new_dist = dist[wu.v].load(std::memory_order_acquire) + wu.w;
    while (new_dist < old_dist) {
      if (dist[v].compare_exchange_weak(old_dist, new_dist, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return new_dist;
      }
    }
    return std::nullopt;
  };

  is_leaf<WGraph, DIRECTED, CACHE_LEAVES> is_leaf(g);

  const auto process_node = [&](executor::frontier& my_frontier, const NodeID u, const priority_level bucket) {
    if (dist[u].load(std::memory_order_relaxed) >= delta * bucket) {
      if constexpr (!DIRECTED) {
        if (g.out_degree(u) < hardware_constructive_interference_size / sizeof(WNode))
          for (WNode wn : g.out_neigh(u)) {
            relax_pull_safe(wn, u);
          }
      }

      for (WNode wn : g.out_neigh(u)) {
        if (auto new_prio = relax_push(u, wn)) {
          if (!is_leaf(wn.v))
            my_frontier.push(wn.v, new_prio.value() / delta);
        }
      }
    }
  };
  scheduler.run(init_sssp, process_node);

#ifdef PAPI_PROFILE
  PAPI_hl_region_end("sssp");
#endif
  return dist;
}

parallel::atomics_array<WeightT> DeltaStep(const WGraph& g, NodeID source, double delta) {

  if (g.directed())
    return BestDeltaStepping<WGraph, true, true>(g, source, delta);
  else
    return BestDeltaStepping<WGraph, false, true>(g, source, delta);
}

void PrintSSSPStats(const WGraph& g, const parallel::atomics_array<WeightT>& dist) {
  auto NotInf = [](const std::atomic<WeightT>& d) { return d != DIST_INF; };
  int64_t num_reached = std::count_if(dist.begin(), dist.end(), NotInf);
  cout << "SSSP Tree reaches " << num_reached << " nodes" << endl;

  WeightT max_dist = 0;
  for (auto i = 0; i < g.num_nodes(); i++)
    if (dist[i] != DIST_INF && dist[i] > max_dist)
      max_dist = dist[i];

  cout << "Max dist " << max_dist << endl;
}

bool approximatelyEqual(WeightT a, WeightT a_ref, double absError = 1e-7, double relError = 1e-5) {
  if (std::fabs(a - a_ref) <= absError || std::fabs(a - a_ref) <= a_ref * relError)
    return true;
  return false;
}

bool SSSPVerifier(const WGraph& g, NodeID source, const parallel::atomics_array<WeightT>& dist_to_test) {
  parallel::atomics_array<WeightT> dist(g.num_nodes(), DIST_INF);
  vector<bool> settled(g.num_nodes());
  dist[source] = 0;

  typedef pair<WeightT, NodeID> WN;
  priority_queue<WN, vector<WN>, greater<WN>> mq;
  mq.push(make_pair(0, source));

  while (!mq.empty()) {
    WeightT tentative_dist = mq.top().first;
    NodeID u = mq.top().second;
    mq.pop();
    settled[u] = true;

    for (WNode wn : g.out_neigh(u)) {
      if (!settled[wn.v])
        if (tentative_dist + wn.w < dist[wn.v]) {
          dist[wn.v] = tentative_dist + wn.w;
          mq.push(make_pair(tentative_dist + wn.w, wn.v));
        }
    }
  }

  // Report any mismatches
  bool all_ok = true;
  for (NodeID n : g.vertices()) {
    if (!approximatelyEqual(dist_to_test[n], dist[n])) {
      // if (dist_to_test[n] != DIST_INF)
      //   cout << n << ": " << dist_to_test[n] << " != " << dist[n] << endl;
      all_ok = false;
      break;
    }
  }
  return all_ok;
}

int main(int argc, char* argv[]) {
#ifdef PAPI_PROFILE
  PAPI_library_init(PAPI_VER_CURRENT);
#endif

  int max_threads = omp_get_max_threads();
  cout << "OMP max threads: " << max_threads << endl;

  CLDelta<WeightT> cli(argc, argv, "single-source shortest-path");
  if (!cli.ParseArgs())
    return -1;

  WeightedBuilder b(cli);
  WGraph g = b.MakeGraph();
  g.PrintStats();

  double sum_weights = 0;
  WeightT min_nonzero_weight = std::numeric_limits<WeightT>::max(), max_weight = 0;
  WNode* edges = g.out_index()[0];

#pragma omp parallel for reduction(+ : sum_weights) reduction(max : max_weight) reduction(min : min_nonzero_weight)
  for (size_t i = 0; i < g.num_edges(); i++) {
    auto w = edges[i].w;
    sum_weights += w;
    max_weight = std::max(max_weight, w);
    min_nonzero_weight = (w > 0) ? std::min(min_nonzero_weight, w) : min_nonzero_weight;
  }

  NodeID num_leaves = 0;
  int64_t max_degree = 0;
#pragma omp parallel for reduction(+ : num_leaves)
  for (size_t i = 0; i < g.num_nodes(); i++) {
    if (g.out_degree(i) == 1) {
      num_leaves += 1;
    }
    max_degree = std::max(max_degree, g.out_degree(i));
  }
  double perc_leaves = ((double)num_leaves / g.num_nodes());

  double avg_weight = (double)sum_weights / g.num_edges();
  double avg_deg = (double)g.num_edges() / g.num_nodes();

  // Sampling weights to estimate median and mean
  std::mt19937 rng(27491095);
  std::uniform_int_distribution<int64_t> udist(0, g.num_edges() - 1);
  int num_samples = 100000;
  std::vector<WeightT> samples(num_samples);
  double sampled_sum_weight = 0;

  for (int i = 0; i < num_samples; i++) {
    samples[i] = edges[udist(rng)].w;
    sampled_sum_weight += samples[i];
  }

  std::sort(samples.begin(), samples.end());
  double sampled_median_weight = samples[num_samples / 2];
  double sampled_avg_weight = sampled_sum_weight / num_samples;

  double accum = 0;
  for (int i = 0; i < num_samples; i++) {
    accum += ((samples[i] - sampled_avg_weight) * (samples[i] - sampled_avg_weight));
  }
  double sampled_stddev_weight = std::sqrt(accum / (num_samples - 1));
  double sampled_skew_weight = (sampled_avg_weight - sampled_median_weight) / sampled_stddev_weight;

  std::cout << "threads: " << max_threads << std::endl
            << std::endl;
  std::cout << "N: " << g.num_nodes() << std::endl;
  std::cout << "M: " << g.num_edges() << std::endl;
  std::cout << "avg_deg: " << avg_deg << std::endl;
  std::cout << "max_deg: " << max_degree << std::endl;
  std::cout << "avg_weight: " << avg_weight << std::endl;
  std::cout << "max_weight: " << max_weight << std::endl;
  std::cout << "min_nonzero_weight: " << min_nonzero_weight << std::endl;
  std::cout << "num_leaves: " << num_leaves << std::endl;
  std::cout << "% num_leaves: " << perc_leaves * 100 << std::endl
            << std::endl;

  std::cout << "sampled_median_weight: " << sampled_median_weight << std::endl;
  std::cout << "sampled_avg_weight: " << sampled_avg_weight << std::endl;
  std::cout << "sampled_stddev_weight: " << sampled_stddev_weight << std::endl;
  std::cout << "sampled_skew_weight: " << sampled_skew_weight << std::endl;
  std::cout << "avg/median weights: " << sampled_avg_weight / sampled_median_weight << std::endl
            << std::endl;

  double delta = cli.delta();
  if (cli.use_heuristic()) {
    bool dense = avg_deg > 8.0;
    if (dense) { // dense
      delta = (sampled_median_weight / avg_deg);
    } else { // sparse
      delta = (sampled_median_weight / avg_deg) * max_threads;
    }
  }

  std::cout << "delta: " << delta << std::endl;

  SourcePicker<WGraph> sp(g, cli);

  for (auto i = 0; i < cli.num_sources(); i++) {
    auto source = 0;
    std::cout << "Source: " << source << std::endl;

    auto SSSPBound = [delta, source](const WGraph& g) {
      return DeltaStep(g, source, delta);
    };

    auto VerifierBound = [source](const WGraph& g, const parallel::atomics_array<WeightT>& dist) {
      return SSSPVerifier(g, source, dist);
    };

    BenchmarkKernel(cli, g, SSSPBound, PrintSSSPStats, VerifierBound);
  }

  return 0;
}
