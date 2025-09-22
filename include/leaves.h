#ifndef LEAVES_H_
#define LEAVES_H_

#include "benchmark.h"
#include "containers/bitmap.h"

template <typename GraphT, bool directed, bool cached>
class is_leaf {};

template <typename GraphT>
class is_leaf<GraphT, false, false> {
public:
  is_leaf(const GraphT& g) : g_(g) {}

  bool operator()(NodeID n) const {
    return g_.out_degree(n) == 1;
  }

private:
  const GraphT& g_;
};

template <typename GraphT>
class is_leaf<GraphT, false, true> {
public:
  is_leaf(const GraphT& g) : leaves(g.num_nodes()) {
#pragma omp parallel for schedule(static, 1024)
    for (NodeID node = 0; node < g.num_nodes(); node++) {
      if (g.out_degree(node) == 1)
        leaves.set(node);
    }
  }

  bool operator()(NodeID n) const {
    return leaves.test(n);
  }

private:
  bitmap leaves;
};

template <typename GraphT>
class is_leaf<GraphT, true, true> {
public:
  is_leaf(const GraphT& g) : leaves(g.num_nodes()) {
#pragma omp parallel for schedule(static, 1024)
    for (NodeID node = 0; node < g.num_nodes(); node++) {
      auto in_deg = g.in_degree(node);
      auto out_deg = g.out_degree(node);
      if (in_deg == 1 && out_deg == 0)
        leaves.set(node);
      else if (in_deg == 1 && out_deg == 1) {
        NodeID in_src = static_cast<NodeID>(g.in_index()[node][0]);
        NodeID out_dst = static_cast<NodeID>(g.out_index()[node][0]);
        if (in_src == out_dst)
          leaves.set(node);
      }
    }
  }

  bool operator()(NodeID n) const {
    return leaves.test(n);
  }

private:
  bitmap leaves;
};

template <typename GraphT>
class is_leaf<GraphT, true, false> {
public:
  is_leaf(const GraphT& g) : g_(g) {}

  bool operator()(NodeID node) const {
    auto in_deg = g_.in_degree(node);
    auto out_deg = g_.out_degree(node);
    if (in_deg == 1 && out_deg == 0)
      return true;
    else if (in_deg == 1 && out_deg == 1) {
      NodeID in_src = static_cast<NodeID>(g_.in_index()[node][0]);
      NodeID out_dst = static_cast<NodeID>(g_.out_index()[node][0]);
      if (in_src == out_dst)
        return true;
    }
    return false;
  }

private:
  const GraphT& g_;
};

#endif // LEAVES_H_