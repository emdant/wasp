// Copyright (c) 2024, Queen's University Belfast
// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <queue>
#include <vector>

#include "bucketing/base.h"
#include "omp.h"
#ifdef PAPI_PROFILE
#include "profiling/papi_helper.h"
#endif

#include "benchmark.h"
#include "bucketing/executor.h"
#include "bucketing/frontier.h"
#include "command_line.h"
#include "leaves.h"
#include "parallel/atomics_array.h"

using namespace std;
using namespace bucketing;

static constexpr WeightT DIST_INF = numeric_limits<WeightT>::max() / 2;

template <typename GraphT, bool DIRECTED, bool CACHE_LEAVES>
auto BestDeltaStepping(const WGraph& g, NodeID source, int32_t delta) {
  int num_threads = omp_get_max_threads();
  parallel::atomics_array<WeightT> dist(g.num_nodes(), DIST_INF);
  dist[source] = 0;

  executor scheduler;

  const auto init_sssp = [&](bucketing::chunks_frontier& my_frontier) {
    my_frontier.push(source, 0);
  };

  const auto relax_push = [&](const NodeID u, const WNode wv) -> std::optional<WeightT> { // outgoing edge
    WeightT old_dist = dist[wv.v].load(std::memory_order_acquire);
    WeightT new_dist = dist[u].load(std::memory_order_acquire) + wv.w;
    while (new_dist < old_dist) {
      if (dist[wv.v].compare_exchange_weak(old_dist, new_dist, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return new_dist;
      }
      new_dist = dist[u].load(std::memory_order_acquire) + wv.w;
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
  constexpr std::int64_t MAX_DEG = 1 << 20;

  const auto is_stale = [&](const priority_level bucket, const NodeID u) {
    return dist[u].load(std::memory_order_relaxed) < delta * bucket;
  };

  const auto inspect_node = [&](bucketing::chunks_frontier& my_frontier, const priority_level bucket, const NodeID u) {
    auto deg = g.out_degree(u);
    if (deg < MAX_DEG)
      return std::pair{0, deg};

    auto begin = MAX_DEG;
    while (begin < deg) {
      auto chunk = new nodes_chunk(u, bucket, begin, std::min(begin + MAX_DEG, deg));
      my_frontier.push(chunk);
      begin += MAX_DEG;
    }
    return std::pair{0, MAX_DEG};
  };

  const auto process_node = [&](bucketing::chunks_frontier& my_frontier, const priority_level bucket, const NodeID u, const std::int64_t begin, const std::int64_t end) {
    auto edges = g.out_index()[u];

    if constexpr (!DIRECTED) {
      if ((end - begin) < hardware_constructive_interference_size / sizeof(WNode))
        for (auto i = begin; i < end; i++) {
          WNode wn = edges[i];
          relax_pull_safe(wn, u);
        }
    }

    for (auto i = begin; i < end; i++) {
      WNode wn = edges[i];
      if (auto new_prio = relax_push(u, wn)) {
        if (!is_leaf(wn.v))
          my_frontier.push(wn.v, new_prio.value() / delta);
      }
    }
  };
  scheduler.run(init_sssp, is_stale, inspect_node, process_node);

  return dist;
}

parallel::atomics_array<WeightT> DeltaStep(const WGraph& g, NodeID source, int32_t delta) {

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
    if (dist_to_test[n] != dist[n]) {
      // if (dist_to_test[n] != DIST_INF)
      //   cout << n << ": " << dist_to_test[n] << " != " << dist[n] << endl;
      all_ok = false;
    }
  }
  return all_ok;
}

int main(int argc, char* argv[]) {
#ifdef PAPI_PROFILE
  papi_helper::initialize();
#endif

  numa_distance_map::initialize();
  int max_threads = omp_get_max_threads();
  cout << "OMP max threads: " << max_threads << endl;

  CLDelta<WeightT> cli(argc, argv, "single-source shortest-path");
  if (!cli.ParseArgs())
    return -1;

  WeightedBuilder b(cli);
  WGraph g = b.MakeGraph();
  g.PrintStats();

  SourcePicker<WGraph> sp(g, cli);

  for (auto i = 0; i < cli.num_sources(); i++) {
    auto source = sp.PickNext();
    std::cout << "Source: " << source << std::endl;

    auto SSSPBound = [&cli, source](const WGraph& g) {
      return DeltaStep(g, source, cli.delta());
    };

    auto VerifierBound = [source](const WGraph& g, const parallel::atomics_array<WeightT>& dist) {
      return SSSPVerifier(g, source, dist);
    };

    BenchmarkKernel(cli, g, SSSPBound, PrintSSSPStats, VerifierBound);
  }

  return 0;
}
