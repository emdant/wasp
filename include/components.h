#ifndef COMPONENTS_H_
#define COMPONENTS_H_

#include <cstdlib>
#include <functional>
#include <random>
#include <stack>
#include <vector>

#include "graph.h"
#include "parallel/vector.h"

template <typename NodeID_, typename DestID_>
class ComponentFinder {

public:
  struct ComponentInfo {
    NodeID_ max_id;
    NodeID_ max_size;
    std::vector<NodeID_> components;
  };

private:
  using Graph = CSRGraph<NodeID_, DestID_>;

  const Graph& g_;
  const bool strongly_;
  ComponentInfo info_;

public:
  explicit ComponentFinder(const Graph& g, bool strongly_connected) : g_(g), strongly_(strongly_connected) {
    if (strongly_) info_ = find_scc();
    else info_ = find_cc();
  }

  const ComponentInfo& info() {
    return info_;
  }

private:
  ComponentInfo find_scc() {
    const NodeID_ num_nodes = g_.num_nodes();

    std::vector<NodeID_> low_links(num_nodes);
    std::vector<NodeID_> defined(num_nodes);
    std::vector<NodeID_> components(num_nodes);

    std::stack<NodeID_> tarjan_stack;

    NodeID_ timestamp = 0, num_scc = 0;
    NodeID_ max_component = 0, max_component_size = 0;

    // initialization
    for (NodeID_ i = 0; i < num_nodes; i++) {
      defined[i] = components[i] = 0;
    }

    const std::function<void(NodeID_)> dfs = [&](NodeID_ u) -> void {
      low_links[u] = defined[u] = ++timestamp;
      tarjan_stack.push(u);
      for (auto& vv : g_.out_neigh(u)) {
        auto& v = static_cast<NodeID_&>(vv);
        if (!defined[v]) {
          dfs(v);
          low_links[u] = std::min(low_links[u], low_links[v]);
        } else if (!components[v]) {
          low_links[u] = std::min(low_links[u], defined[v]);
        }
      }

      if (low_links[u] == defined[u]) {
        num_scc++;
        NodeID_ size = 0;
        while (1) {
          NodeID_ v = tarjan_stack.top();
          tarjan_stack.pop();
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

    // Recursive calls might cause stack overflow. Use `ulimit -s unlimited`
    for (NodeID_ i = 0; i < num_nodes; i++) {
      if (!defined[i]) {
        dfs(i);
      }
    }

    return {max_component, max_component_size, components};
  }

  // Place nodes u and v in same component of lower component ID
  inline void Link(NodeID_ u, NodeID_ v, parallel::vector<NodeID_>& comp) {
    NodeID_ p1 = comp[u];
    NodeID_ p2 = comp[v];
    while (p1 != p2) {
      NodeID_ high = p1 > p2 ? p1 : p2;
      NodeID_ low = p1 + (p2 - high);
      NodeID_ p_high = comp[high];
      // Was already 'low' or succeeded in writing 'low'
      if ((p_high == low) ||
          (p_high == high && compare_and_swap(comp[high], high, low)))
        break;
      p1 = comp[comp[high]];
      p2 = comp[low];
    }
  }

  // Reduce depth of tree for each component to 1 by crawling up parents
  inline void Compress(const Graph& g, parallel::vector<NodeID_>& comp) {
#pragma omp parallel for schedule(dynamic, 16384)
    for (NodeID_ n = 0; n < g.num_nodes(); n++) {
      while (comp[n] != comp[comp[n]]) {
        comp[n] = comp[comp[n]];
      }
    }
  }

  inline NodeID_ SampleFrequentElement(const parallel::vector<NodeID_>& comp, int64_t num_samples = 1024) {
    std::unordered_map<NodeID_, int> sample_counts(32);
    using kvp_type = typename std::unordered_map<NodeID_, int>::value_type;
    // Sample elements from 'comp'
    std::mt19937 gen;
    std::uniform_int_distribution<NodeID_> distribution(0, comp.size() - 1);
    for (NodeID_ i = 0; i < num_samples; i++) {
      NodeID_ n = distribution(gen);
      sample_counts[comp[n]]++;
    }
    // Find most frequent element in samples (estimate of most frequent overall)
    auto most_frequent = std::max_element(
        sample_counts.begin(), sample_counts.end(),
        [](const kvp_type& a, const kvp_type& b) { return a.second < b.second; }
    );
    if (most_frequent == sample_counts.end()) {
      std::cout << "failed sampling" << std::endl;
      std::exit(-1);
    }
    return most_frequent->first;
  }

  inline parallel::vector<NodeID_> Afforest(const Graph& g, int32_t neighbor_rounds = 2) {
    parallel::vector<NodeID_> comp(g.num_nodes());

// Initialize each node to a single-node self-pointing tree
#pragma omp parallel for
    for (NodeID_ n = 0; n < g.num_nodes(); n++)
      comp[n] = n;

    // Process a sparse sampled subgraph first for approximating components.
    // Sample by processing a fixed number of neighbors for each node (see paper)
    for (int r = 0; r < neighbor_rounds; ++r) {
#pragma omp parallel for schedule(dynamic, 16384)
      for (NodeID_ u = 0; u < g.num_nodes(); u++) {
        for (NodeID_ v : g.out_neigh(u, r)) {
          // Link at most one time if neighbor available at offset r
          Link(u, v, comp);
          break;
        }
      }
      Compress(g, comp);
    }

    // Sample 'comp' to find the most frequent element -- due to prior
    // compression, this value represents the largest intermediate component
    NodeID_ c = SampleFrequentElement(comp);

    // Final 'link' phase over remaining edges (excluding the largest component)
    if (!g.directed()) {
#pragma omp parallel for schedule(dynamic, 16384)
      for (NodeID_ u = 0; u < g.num_nodes(); u++) {
        // Skip processing nodes in the largest component
        if (comp[u] == c)
          continue;
        // Skip over part of neighborhood (determined by neighbor_rounds)
        for (NodeID_ v : g.out_neigh(u, neighbor_rounds)) {
          Link(u, v, comp);
        }
      }
    } else {
#pragma omp parallel for schedule(dynamic, 16384)
      for (NodeID_ u = 0; u < g.num_nodes(); u++) {
        if (comp[u] == c)
          continue;
        for (NodeID_ v : g.out_neigh(u, neighbor_rounds)) {
          Link(u, v, comp);
        }
        // To support directed graphs, process reverse graph completely
        for (NodeID_ v : g.in_neigh(u)) {
          Link(u, v, comp);
        }
      }
    }
    // Finally, 'compress' for final convergence
    Compress(g, comp);
    return comp;
  }

  ComponentInfo find_cc() {
    auto components = Afforest(g_);

    std::unordered_map<NodeID_, NodeID_> count;
    for (NodeID_ comp_i : components)
      count[comp_i] += 1;

    std::pair<NodeID_, NodeID_> max_pair = {0, 0};
    for (auto kv_pair : count) {
      if (kv_pair.second > max_pair.second)
        max_pair = kv_pair;
    }

    return {max_pair.first, max_pair.second, std::vector(components.begin(), components.end())};
  }
};

template <typename NodeID_, typename DestID_>
ComponentFinder(CSRGraph<NodeID_, DestID_>) -> ComponentFinder<NodeID_, DestID_>;

#endif