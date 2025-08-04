// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <algorithm>
#include <iostream>
#include <unordered_map>

#include "benchmark.h"
#include "command_line.h"
#include "graph.h"
#include "parallel/vector.h"
#include "writer.h"

using namespace std;

// Connected Components algorithm from GAP

// Place nodes u and v in same component of lower component ID
void Link(NodeID u, NodeID v, parallel::vector<NodeID>& comp) {
  NodeID p1 = comp[u];
  NodeID p2 = comp[v];
  while (p1 != p2) {
    NodeID high = p1 > p2 ? p1 : p2;
    NodeID low = p1 + (p2 - high);
    NodeID p_high = comp[high];
    // Was already 'low' or succeeded in writing 'low'
    if ((p_high == low) ||
        (p_high == high && compare_and_swap(comp[high], high, low)))
      break;
    p1 = comp[comp[high]];
    p2 = comp[low];
  }
}

// Reduce depth of tree for each component to 1 by crawling up parents
void Compress(const Graph& g, parallel::vector<NodeID>& comp) {
#pragma omp parallel for schedule(dynamic, 16384)
  for (NodeID n = 0; n < g.num_nodes(); n++) {
    while (comp[n] != comp[comp[n]]) {
      comp[n] = comp[comp[n]];
    }
  }
}

NodeID SampleFrequentElement(const parallel::vector<NodeID>& comp, bool logging_enabled = false, int64_t num_samples = 1024) {
  std::unordered_map<NodeID, int> sample_counts(32);
  using kvp_type = std::unordered_map<NodeID, int>::value_type;
  // Sample elements from 'comp'
  std::mt19937 gen;
  std::uniform_int_distribution<NodeID> distribution(0, comp.size() - 1);
  for (NodeID i = 0; i < num_samples; i++) {
    NodeID n = distribution(gen);
    sample_counts[comp[n]]++;
  }
  // Find most frequent element in samples (estimate of most frequent overall)
  auto most_frequent = std::max_element(
      sample_counts.begin(), sample_counts.end(),
      [](const kvp_type& a, const kvp_type& b) { return a.second < b.second; }
  );
  float frac_of_graph = static_cast<float>(most_frequent->second) / num_samples;
  if (logging_enabled)
    std::cout
        << "Skipping largest intermediate component (ID: " << most_frequent->first
        << ", approx. " << static_cast<int>(frac_of_graph * 100)
        << "% of the graph)" << std::endl;
  return most_frequent->first;
}

parallel::vector<NodeID> Afforest(const Graph& g, bool logging_enabled = false, int32_t neighbor_rounds = 2) {
  parallel::vector<NodeID> comp(g.num_nodes());

// Initialize each node to a single-node self-pointing tree
#pragma omp parallel for
  for (NodeID n = 0; n < g.num_nodes(); n++)
    comp[n] = n;

  // Process a sparse sampled subgraph first for approximating components.
  // Sample by processing a fixed number of neighbors for each node (see paper)
  for (int r = 0; r < neighbor_rounds; ++r) {
#pragma omp parallel for schedule(dynamic, 16384)
    for (NodeID u = 0; u < g.num_nodes(); u++) {
      for (NodeID v : g.out_neigh(u, r)) {
        // Link at most one time if neighbor available at offset r
        Link(u, v, comp);
        break;
      }
    }
    Compress(g, comp);
  }

  // Sample 'comp' to find the most frequent element -- due to prior
  // compression, this value represents the largest intermediate component
  NodeID c = SampleFrequentElement(comp, logging_enabled);

  // Final 'link' phase over remaining edges (excluding the largest component)
  if (!g.directed()) {
#pragma omp parallel for schedule(dynamic, 16384)
    for (NodeID u = 0; u < g.num_nodes(); u++) {
      // Skip processing nodes in the largest component
      if (comp[u] == c)
        continue;
      // Skip over part of neighborhood (determined by neighbor_rounds)
      for (NodeID v : g.out_neigh(u, neighbor_rounds)) {
        Link(u, v, comp);
      }
    }
  } else {
#pragma omp parallel for schedule(dynamic, 16384)
    for (NodeID u = 0; u < g.num_nodes(); u++) {
      if (comp[u] == c)
        continue;
      for (NodeID v : g.out_neigh(u, neighbor_rounds)) {
        Link(u, v, comp);
      }
      // To support directed graphs, process reverse graph completely
      for (NodeID v : g.in_neigh(u)) {
        Link(u, v, comp);
      }
    }
  }
  // Finally, 'compress' for final convergence
  Compress(g, comp);
  return comp;
}

int main(int argc, char* argv[]) {
  CLConvert cli(argc, argv, "converter");
  cli.ParseArgs();

  std::size_t ext_pos = cli.filename().rfind('.');
  if (cli.filename().substr(ext_pos) == ".mtx" && cli.out_format() == Format::MATRIX_MARKET && cli.out_largest()) {
    std::cout << "Output is largest (non-strongly) connected component" << std::endl;
    Builder b(cli);
    Graph g = b.MakeGraph();
    g.PrintStats();

    auto components = Afforest(g, false);
    LargestComponentWriter w(g, components);
    w.WriteGraph(cli.out_filename(), cli.out_format(), cli.out_weighted());
    return 0;
  }

  if (cli.out_weighted()) {
    std::cout << "Output is weighted" << std::endl;
    WeightedBuilder bw(cli);
    WGraph wg = bw.MakeGraph();
    wg.PrintStats();
    WeightedWriter ww(wg);
    ww.WriteGraph(cli.out_filename(), cli.out_format());
  } else {
    std::cout << "Output is not weighted" << std::endl;
    Builder b(cli);
    Graph g = b.MakeGraph();
    g.PrintStats();
    Writer w(g);
    w.WriteGraph(cli.out_filename(), cli.out_format());
  }
  return 0;
}
