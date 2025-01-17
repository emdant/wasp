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

struct bfs_pair {
  NodeID parent;
  WeightT level;
};

template <typename GraphT, bool DIRECTED, bool CACHE_LEAVES>
parallel::atomics_array<bfs_pair> BestBFS(const GraphT& g, NodeID source, int32_t delta = 1) {
#ifdef PAPI_PROFILE
  PAPI_hl_region_begin("bfs");
#endif
  parallel::atomics_array<bfs_pair> bfs_tree(g.num_nodes(), {-1, DIST_INF});
  bfs_tree[source] = {source, 0};

  executor scheduler;
  is_leaf<GraphT, DIRECTED, CACHE_LEAVES> is_leaf(g);

  const auto init_bfs = [&](executor::frontier& my_frontier) {
    my_frontier.push(source, 0);
  };

  auto visit_edge = [&](NodeID u, NodeID v) -> std::optional<WeightT> {
    bfs_pair old = bfs_tree[v].load(std::memory_order_relaxed);
    bfs_pair np = {u, bfs_tree[u].load(std::memory_order_relaxed).level + 1};
    while (np.level < old.level) {
      if (bfs_tree[v].compare_exchange_weak(old, np, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return np.level;
      }
    }
    return std::nullopt;
  };

  const auto process_node = [&](executor::frontier& my_frontier, const NodeID u, const priority_level bucket) {
    if (bfs_tree[u].load(std::memory_order_relaxed).level >= delta * bucket) {
      for (NodeID v : g.out_neigh(u)) {
        if (auto level = visit_edge(u, v)) {
          if (!is_leaf(v))
            my_frontier.push(v, level.value() / delta);
        }
      }
    }
  };
  scheduler.run(init_bfs, process_node);

#ifdef PAPI_PROFILE
  PAPI_hl_region_end("bfs");
#endif
  return bfs_tree;
}

parallel::atomics_array<bfs_pair> BFS(const Graph& g, NodeID source, int32_t delta) {
  if (g.directed())
    return BestBFS<Graph, true, true>(g, source, delta);
  else
    return BestBFS<Graph, false, true>(g, source, delta);
}

void PrintBFSStats(const Graph& g, parallel::atomics_array<bfs_pair>& bfs_tree) {
  int64_t tree_size = 0;
  int64_t n_edges = 0;
  for (NodeID n : g.vertices()) {
    if (bfs_tree[n].load().parent >= 0) {
      n_edges += g.out_degree(n);
      tree_size++;
    }
  }
  cout << "BFS Tree has " << tree_size << " nodes and ";
  cout << n_edges << " edges" << endl;
}

bool BFSVerifier(const Graph& g, NodeID source, parallel::atomics_array<bfs_pair>& parent) {
  parallel::vector<int> depth(g.num_nodes(), -1);
  depth[source] = 0;
  vector<NodeID> to_visit;
  to_visit.reserve(g.num_nodes());
  to_visit.push_back(source);
  for (auto it = to_visit.begin(); it != to_visit.end(); it++) {
    NodeID u = *it;
    for (NodeID v : g.out_neigh(u)) {
      if (depth[v] == -1) {
        depth[v] = depth[u] + 1;
        to_visit.push_back(v);
      }
    }
  }
  for (NodeID u : g.vertices()) {
    if ((depth[u] != -1) && (parent[u].load().parent != -1)) {
      if (u == source) {
        if (!((parent[u].load().parent == u) && (depth[u] == 0))) {
          cout << "Source wrong" << endl;
          return false;
        }
        continue;
      }
      bool parent_found = false;
      for (NodeID v : g.in_neigh(u)) {
        if (v == parent[u].load().parent) {
          if (depth[v] != depth[u] - 1) {
            cout << "Wrong depths for " << u << " & " << v << endl;
            return false;
          }
          parent_found = true;
          break;
        }
      }
      if (!parent_found) {
        cout << "Couldn't find edge from " << parent[u].load().parent << " to " << u << endl;
        return false;
      }
    } else if (depth[u] != parent[u].load().parent) {
      cout << "Reachability mismatch" << endl;
      return false;
    }
  }
  return true;
}

int main(int argc, char* argv[]) {
  CLDelta<int> cli(argc, argv, "breadth-first search");
  if (!cli.ParseArgs())
    return -1;
  Builder b(cli);
  Graph g = b.MakeGraph();
  g.PrintStats();

  SourcePicker<Graph> sp(g, cli);

  for (auto i = 0; i < cli.num_sources(); i++) {
    auto source = sp.PickNext();
    std::cout << "Source: " << source << std::endl;

    auto BFSBound = [&cli, source](const Graph& g) {
      return BFS(g, source, cli.delta());
    };

    auto VerifierBound = [source](const Graph& g, parallel::atomics_array<bfs_pair>& parent) {
      return BFSVerifier(g, source, parent);
    };

    BenchmarkKernel(cli, g, BFSBound, PrintBFSStats, VerifierBound);
  }

  return 0;
}