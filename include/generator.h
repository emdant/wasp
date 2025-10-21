// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef GENERATOR_H_
#define GENERATOR_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>

#include "graph.h"
#include "parallel/vector.h"
#include "timer.h"
#include "util.h"

namespace graph_utils {

namespace detail {

template <typename ValueT>
class RNG {
  using UValueT = std::make_unsigned_t<ValueT>;

public:
  using type = std::conditional_t<std::numeric_limits<UValueT>::digits == 32, std::mt19937, std::mt19937_64>;
};

template <typename T>
using RNG_t = typename RNG<T>::type;
constexpr std::size_t block_size = 1 << 19;

template <typename RngT, typename DistT>
auto generate(RngT& rng, DistT& dist, bool positive) {
  if (!positive) return dist(rng);
  else {
    auto ret = dist(rng);
    while (ret < 0)
      ret = dist(rng);
    return ret;
  }
};

template <typename IntegerT>
class GAPLegacyUniformDistribution {
  using UIntegerT = std::make_unsigned_t<IntegerT>;

public:
  // The GAP uniform distribution used a Mersenne-Twister RNG se we hardcode that for the constructor
  GAPLegacyUniformDistribution(IntegerT max_value) {
    no_mod_ = RNG_t<IntegerT>::max() == static_cast<UIntegerT>(max_value);
    mod_ = max_value + 1;
    UIntegerT remainder_sub_1 = RNG_t<IntegerT>::max() % mod_;
    if (remainder_sub_1 == mod_ - 1)
      cutoff_ = 0;
    else
      cutoff_ = RNG_t<IntegerT>::max() - remainder_sub_1;
  }

  template <typename RngT>
  IntegerT operator()(RngT& rng) {
    UIntegerT rand_num = rng();
    if (no_mod_)
      return rand_num;
    if (cutoff_ != 0) {
      while (rand_num >= cutoff_)
        rand_num = rng();
    }
    return rand_num % mod_;
  }

private:
  bool no_mod_;
  UIntegerT mod_;
  UIntegerT cutoff_;
};

} // namespace detail

template <typename WeightT>
struct GAPLegacyDistribution : public detail::GAPLegacyUniformDistribution<WeightT> {
  GAPLegacyDistribution() : detail::GAPLegacyUniformDistribution<WeightT>(254) {}
};

template <typename WeightT>
struct GAPDistribution : public std::uniform_int_distribution<WeightT> {
  GAPDistribution() : std::uniform_int_distribution<WeightT>(1, 255) {}
};

template <typename WeightT>
struct Graph500Distribution : public std::uniform_real_distribution<WeightT> {
  Graph500Distribution() : std::uniform_real_distribution<WeightT>(0, 1) {}
};

template <typename WeightT>
struct JulienneDistribution : public std::uniform_real_distribution<WeightT> {
  JulienneDistribution(std::size_t n) : std::uniform_real_distribution<WeightT>(1, std::log2(n)) {}
};

template <typename WeightT>
struct GraphBasedNormalDistribution : public std::normal_distribution<WeightT> {
  GraphBasedNormalDistribution(std::size_t n, std::size_t m) : std::normal_distribution<WeightT>(1.0, std::sqrt(static_cast<double>(n) / m)) {}
};

template <typename NodeID_, typename WeightT_, typename DistT>
void replace_weights(CSRGraph<NodeID_, NodeWeight<NodeID_, WeightT_>>& g, DistT& dist, bool positive) {
  using WNode = NodeWeight<NodeID_, WeightT_>;
  using RNG = detail::RNG_t<NodeID_>;

  // When comparing edges we only want to compare the vertex in this case
  auto compare_node = [](const WNode& a, const WNode& b) { return a.v < b.v; };
  std::size_t n = static_cast<std::size_t>(g.num_nodes());

#pragma omp parallel for
  for (std::size_t block = 0; block < n; block += detail::block_size) {

    // RNG is seeded with graph stats and current block
    RNG rng(g.num_nodes() + g.num_edges_directed() + (block / detail::block_size));

    for (std::size_t u = block; u < std::min(block + detail::block_size, n); u++) {
      for (WNode& wn : g.out_neigh(u)) {
        if (g.directed() || static_cast<NodeID_>(u) < wn.v) { // for undirected graphs: only consider edges where u < v
          WeightT_ w = detail::generate(rng, dist, positive);
          wn.w = w;

          // Look for the in-edge and update the weight
          auto in_neigh = g.in_neigh(wn.v); // for undirected graphs: out_neigh == in_neigh
          auto it = std::lower_bound(in_neigh.begin(), in_neigh.end(), static_cast<WNode>(u), compare_node);
          if (it == in_neigh.end()) {
            auto repr = g.directed() ? "Inverse (CSC) for directed" : "CSR for undirected";
            std::cout << repr << " graph is malformed. Aborting." << std::endl;
            std::exit(-1);
          }
          (*it).w = w;
        }
      }
    }
  }
}

template <typename NodeID_, typename WeightT_, typename DistT>
void replace_weights(parallel::vector<EdgePair<NodeID_, NodeWeight<NodeID_, WeightT_>>>& el, DistT& dist, bool positive) {
  using RNG = detail::RNG_t<NodeID_>;

#pragma omp parallel for
  for (std::size_t block = 0; block < el.size(); block += detail::block_size) {
    RNG rng(kRandSeed + block / detail::block_size);
    for (std::size_t e = block; e < std::min(block + detail::block_size, el.size()); e++)
      el[e].v.w = detail::generate(rng, dist, positive);
  }
}

} // namespace graph_utils

// TODO: remove and use GAPLegacyUniformDistribution
// maps to range [0,max_value], tailored to std::mt19937
template <typename NodeID_, typename rng_t_, typename uNodeID_ = typename std::make_unsigned<NodeID_>::type>
class UniDist {
public:
  UniDist(NodeID_ max_value, rng_t_& rng) : rng_(rng) {
    no_mod_ = rng_.max() == static_cast<uNodeID_>(max_value);
    mod_ = max_value + 1;
    uNodeID_ remainder_sub_1 = rng_.max() % mod_;
    if (remainder_sub_1 == mod_ - 1)
      cutoff_ = 0;
    else
      cutoff_ = rng_.max() - remainder_sub_1;
  }

  NodeID_ operator()() {
    uNodeID_ rand_num = rng_();
    if (no_mod_)
      return rand_num;
    if (cutoff_ != 0) {
      while (rand_num >= cutoff_)
        rand_num = rng_();
    }
    return rand_num % mod_;
  }

private:
  rng_t_& rng_;
  bool no_mod_;
  uNodeID_ mod_;
  uNodeID_ cutoff_;
};

/*
GAP Benchmark Suite
Class:  Generator
Author: Scott Beamer

Given scale and degree, generates edgelist for synthetic graph
 - Intended to be called from Builder
 - GenerateEL(uniform) generates and returns the edgelist
 - Can generate uniform random (uniform=true) or R-MAT graph according
   to Graph500 parameters (uniform=false)
 - Can also randomize weights within a weighted edgelist (InsertWeights)
 - Blocking/reseeding is for parallelism with deterministic output edgelist
*/
template <
    typename NodeID_,
    typename DestID_ = NodeID_,
    typename WeightT_ = NodeID_,
    typename uNodeID_ = typename std::make_unsigned<NodeID_>::type,
    int uNodeID_bits_ = std::numeric_limits<uNodeID_>::digits,
    typename rng_t_ = typename std::conditional<(uNodeID_bits_ == 32), std::mt19937, std::mt19937_64>::type>
class Generator {
  typedef EdgePair<NodeID_, DestID_> Edge;
  typedef EdgePair<NodeID_, NodeWeight<NodeID_, WeightT_>> WEdge;
  typedef parallel::vector<Edge> EdgeList;

public:
  Generator(int scale, int degree) {
    scale_ = scale;
    num_nodes_ = 1l << scale;
    num_edges_ = num_nodes_ * degree;
    if (num_nodes_ > std::numeric_limits<NodeID_>::max()) {
      std::cout << "NodeID type (max: " << std::numeric_limits<NodeID_>::max();
      std::cout << ") too small to hold " << num_nodes_ << std::endl;
      std::cout << "Recommend changing NodeID (typedef'd in src/benchmark.h)";
      std::cout << " to a wider type and recompiling" << std::endl;
      std::exit(-31);
    }
  }

  void PermuteIDs(EdgeList& el) {
    parallel::vector<NodeID_> permutation(num_nodes_);
    std::mt19937 rng(kRandSeed);
#pragma omp parallel for
    for (NodeID_ n = 0; n < num_nodes_; n++)
      permutation[n] = n;
    shuffle(permutation.begin(), permutation.end(), rng);
#pragma omp parallel for
    for (int64_t e = 0; e < num_edges_; e++)
      el[e] = Edge(permutation[el[e].u], permutation[el[e].v]);
  }

  EdgeList MakeUniformEL() {
    EdgeList el(num_edges_);
#pragma omp parallel
    {
      rng_t_ rng;
      UniDist<NodeID_, rng_t_> udist(num_nodes_ - 1, rng);
#pragma omp for
      for (int64_t block = 0; block < num_edges_; block += block_size) {
        rng.seed(kRandSeed + block / block_size);
        for (int64_t e = block; e < std::min(block + block_size, num_edges_); e++) {
          el[e] = Edge(udist(), udist());
        }
      }
    }
    return el;
  }

  EdgeList MakeRMatEL() {
    const uNodeID_ max = std::numeric_limits<uNodeID_>::max();
    const uNodeID_ A = 0.57 * max, B = 0.19 * max, C = 0.19 * max;
    EdgeList el(num_edges_);
#pragma omp parallel
    {
      rng_t_ rng;
#pragma omp for
      for (int64_t block = 0; block < num_edges_; block += block_size) {
        rng.seed(kRandSeed + block / block_size);
        for (int64_t e = block; e < std::min(block + block_size, num_edges_); e++) {
          NodeID_ src = 0, dst = 0;
          for (int depth = 0; depth < scale_; depth++) {
            uNodeID_ rand_point = rng();
            src = src << 1;
            dst = dst << 1;
            if (rand_point < A + B) {
              if (rand_point > A)
                dst++;
            } else {
              src++;
              if (rand_point > A + B + C)
                dst++;
            }
          }
          el[e] = Edge(src, dst);
        }
      }
    }
    PermuteIDs(el);
    // TIME_PRINT("Shuffle", std::shuffle(el.begin(), el.end(),
    //                                    std::mt19937()));
    return el;
  }

  EdgeList GenerateEL(bool uniform) {
    EdgeList el;
    Timer t;
    t.Start();
    if (uniform)
      el = MakeUniformEL();
    else
      el = MakeRMatEL();
    t.Stop();
    PrintTime("Generate Time", t.Seconds());
    return el;
  }

private:
  int scale_;
  int64_t num_nodes_;
  int64_t num_edges_;
  static const int64_t block_size = 1 << 18;
};

#endif // GENERATOR_H_
