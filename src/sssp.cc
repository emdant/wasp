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
#include "parallel/atomics_array.h"
#include "parallel/pparray.h"
#include "parallel/pvector.h"
#include "platform_atomics.h"
#include "timer.h"

using namespace std;
using namespace bucketing;

const WeightT kDistInf = numeric_limits<WeightT>::max() / 2;

using Distances = parallel::atomics_array<WeightT>;

template <bool logging = false>
Distances DeltaStep(const WGraph& g, NodeID source, WeightT delta) {
#ifdef PAPI_PROFILE
  PAPI_hl_region_begin("sssp-ws");
#endif

  Timer internal_execution_timer;

  internal_execution_timer.Start();

  Distances dist(g.num_nodes(), kDistInf);
  dist[source] = 0;

  auto relax_edge = [&](NodeID u, WNode wv) -> std::optional<WeightT> {
    WeightT old_dist = dist[wv.v].load(std::memory_order_acquire);
    WeightT new_dist = dist[u].load(std::memory_order_acquire) + wv.w;
    while (new_dist < old_dist) {
      if (dist[wv.v].compare_exchange_weak(old_dist, new_dist, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return new_dist;
      }
    }
    return std::nullopt;
  };

  bucketing::executor bucketing(g, dist, delta, relax_edge);

  internal_execution_timer.Stop();
  cout << "Allocation time: " << internal_execution_timer.Seconds() << endl;

  internal_execution_timer.Start();
  bucketing.run(source);
  internal_execution_timer.Stop();

#ifdef PAPI_PROFILE
  PAPI_hl_region_end("sssp-ws");
#endif

  cout << "Trial Time: " << internal_execution_timer.Seconds() << endl;
  // bucketing.print_content();

  return dist;
}

void PrintSSSPStats(const WGraph& g, const Distances& dist) {
  auto NotInf = [](const std::atomic<WeightT>& d) { return d != kDistInf; };
  int64_t num_reached = std::count_if(dist.begin(), dist.end(), NotInf);
  cout << "SSSP Tree reaches " << num_reached << " nodes" << endl;

  WeightT max_dist = 0;
  for (auto i = 0; i < g.num_nodes(); i++)
    if (dist[i] != kDistInf && dist[i] > max_dist)
      max_dist = dist[i];

  cout << "Max dist " << max_dist << endl;
}

// Compares against simple serial implementation
bool SSSPVerifier(const WGraph& g, NodeID source, const Distances& dist_to_test) {
  // Serial Dijkstra implementation to get oracle distances
  Distances oracle_dist(g.num_nodes(), kDistInf);
  oracle_dist[source] = 0;
  typedef pair<WeightT, NodeID> WN;
  priority_queue<WN, vector<WN>, greater<WN>> mq;
  mq.push(make_pair(0, source));
  while (!mq.empty()) {
    WeightT td = mq.top().first;
    NodeID u = mq.top().second;
    mq.pop();
    if (td == oracle_dist[u]) {
      for (WNode wn : g.out_neigh(u)) {
        if (td + wn.w < oracle_dist[wn.v]) {
          oracle_dist[wn.v] = td + wn.w;
          mq.push(make_pair(td + wn.w, wn.v));
        }
      }
    }
  }
  // Report any mismatches
  bool all_ok = true;
  for (NodeID n : g.vertices()) {
    if (dist_to_test[n] != oracle_dist[n]) {
      // if (dist_to_test[n] != kDistInf)
      //   cout << n << ": " << dist_to_test[n] << " != " << oracle_dist[n] << endl;
      all_ok = false;
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
  SourcePicker<WGraph> sp(g, cli);

  auto max_out = g.out_degree(0);
  auto max_in = g.in_degree(0);
  auto min_out = numeric_limits<int64_t>::max();
  auto min_in = numeric_limits<int64_t>::max();

  for (auto i = 1; i < g.num_nodes(); i++) {
    max_out = std::max(max_out, g.out_degree(i));
    max_in = std::max(max_in, g.in_degree(i));

    if (g.out_degree(i) != 0)
      min_out = std::min(min_out, g.out_degree(i));
    if (g.in_degree(i) != 0)
      min_in = std::min(min_in, g.in_degree(i));
  }

  WeightT min = kDistInf;
  WeightT max = 0;
  for (auto i = 0; i < g.num_nodes(); i++) {
    for (WNode wn : g.out_neigh(i)) {
      if (wn.w < min)
        min = wn.w;
      if (wn.w > max)
        max = wn.w;
    }
  }

  std::cout << "Min edge weight: " << min << std::endl;
  std::cout << "Max edge weight: " << max << std::endl;

  std::cout << "Max out_degree: " << max_out << std::endl;
  std::cout << "Max in_degree: " << max_in << std::endl;
  std::cout << "Min out_degree: " << min_out << std::endl;
  std::cout << "Min in_degree: " << min_in << std::endl;

  auto SSSPBound = [&sp, &cli](const WGraph& g) {
    auto source = sp.PickNext();
    std::cout << "Source: " << source << std::endl;

    if (cli.logging_en())
      return DeltaStep<true>(g, source, cli.delta());
    else
      return DeltaStep<false>(g, source, cli.delta());
  };

  SourcePicker<WGraph> vsp(g, cli);
  auto VerifierBound = [&vsp](const WGraph& g, const Distances& dist) {
    return SSSPVerifier(g, vsp.PickNext(), dist);
  };

  BenchmarkKernel(cli, g, SSSPBound, PrintSSSPStats, VerifierBound);

  return 0;
}
