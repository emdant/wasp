#pragma once
#include <cstddef>
#include <functional>
#include <stack>
#include <vector>

#include "benchmark.h"

struct SCCInfo {
  NodeID component_id;
  NodeID max_size;
  std::vector<NodeID> components;
};

struct CCInfo {
  NodeID component_id;
  NodeID max_size;
  std::vector<NodeID> components;
};

template <typename Graph>
SCCInfo find_scc(Graph& g) {
  std::vector<NodeID> low_links(g.num_nodes());
  std::vector<NodeID> defined(g.num_nodes());
  std::vector<NodeID> components(g.num_nodes());
  std::stack<NodeID> stack;
  NodeID timestamp = 0;
  std::size_t num_scc = 0;
  NodeID max_component = 0;
  NodeID max_component_size = 0;

  const std::function<void(NodeID)> dfs = [&](NodeID u) -> void {
    low_links[u] = defined[u] = ++timestamp;
    stack.push(u);
    for (const auto& v : g.out_neigh(u)) {
      if (!defined[v]) {
        dfs(v);
        low_links[u] = std::min(low_links[u], low_links[v]);
      } else if (!components[v]) {
        low_links[u] = std::min(low_links[u], defined[v]);
      }
    }
    if (low_links[u] == defined[u]) {
      num_scc++;
      NodeID size = 0;
      while (1) {
        NodeID v = stack.top();
        stack.pop();
        size++;
        components[v] = num_scc;
        if (v == u) {
          break;
        }
      }
      if (size > max_component_size) {
        max_component_size = size;
        max_component = num_scc;
      }
    }
  };

  // Tarjan algorithm

  // initialization
  for (std::size_t i = 0; i < g.num_nodes(); i++) {
    defined[i] = components[i] = 0;
  }

  stack = std::stack<NodeID>();
  for (std::size_t i = 0; i < g.num_nodes(); i++) {
    if (!defined[i]) {
      dfs(i);
    }
  }

  return {max_component, max_component_size, components};
}

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

template <typename Graph>
CCInfo find_cc(Graph& g) {
  auto components = Afforest(g, false);

  std::unordered_map<NodeID, NodeID> count;
  for (NodeID comp_i : components)
    count[comp_i] += 1;

  std::pair<NodeID, NodeID> max_pair = {0, 0};
  for (auto kv_pair : count) {
    if (kv_pair.second > max_pair.second)
      max_pair = kv_pair;
  }

  return {max_pair.first, max_pair.second, components};
}
