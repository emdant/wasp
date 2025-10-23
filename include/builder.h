// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef BUILDER_H_
#define BUILDER_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <iterator>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "command_line.h"
#include "generator.h"
#include "graph.h"
#include "omp.h"
#include "parallel/vector.h"
#include "platform_atomics.h"
#include "reader.h"
#include "timer.h"
#include "util.h"

/*
GAP Benchmark Suite
Class:  BuilderBase
Author: Scott Beamer

Given arguments from the command line (cli), returns a built graph
 - MakeGraph() will parse cli and obtain edgelist to call
   MakeGraphFromEL(edgelist) to perform the actual graph construction
 - edgelist can be from file (Reader) or synthetically generated (Generator)
 - Common case: BuilderBase typedef'd (w/ params) to be Builder (benchmark.h)
*/

template <typename NodeID_, typename DestID_, typename WeightT_, bool invert = true>
class BuilderBase {
  static constexpr bool WEIGHTED_BUILDER = !std::is_same_v<NodeID_, DestID_>;
  typedef EdgePair<NodeID_, DestID_> Edge;
  typedef parallel::vector<Edge> EdgeList;
  typedef Reader<NodeID_, DestID_, WeightT_, invert> ReaderT;

  bool symmetrize_;
  bool needs_weights_ = false;
  bool relabel_vertices_ = false;
  int64_t num_nodes_ = -1;
  int64_t num_edges_ = -1;

  std::string graph_filename_ = "";
  std::string weights_filename_ = "";

  bool generate_graph = false;
  GraphGenerator graph_gen_ = GraphGenerator::NO_GEN;
  int synthetic_scale_;
  int synthetic_degree_;

  bool generate_weights_ = false;
  WeightGenerator weight_dist_ = WeightGenerator::NO_GEN;
  std::pair<WeightT_, WeightT_> dist_params_;

public:
  explicit BuilderBase(const CLBase& cli) {
    symmetrize_ = cli.symmetrize();
    graph_filename_ = cli.graph_filename();
    if (graph_filename_ == "") generate_graph = true;

    if (generate_graph) {
      graph_gen_ = cli.graph_generator();
      synthetic_scale_ = cli.synthetic_scale();
      synthetic_degree_ = cli.synthetic_degree();
    }

    if constexpr (WEIGHTED_BUILDER) {
      weight_dist_ = cli.weight_distribution();
      dist_params_ = cli.distribution_params();
      weights_filename_ = cli.weights_filename();
      generate_weights_ = weight_dist_ != WeightGenerator::NO_GEN;
      needs_weights_ = WEIGHTED_BUILDER;
    }
  }

  void toggleRelabeling(bool state) {
    relabel_vertices_ = state;
  }

  DestID_ GetSource(EdgePair<NodeID_, NodeID_> e) {
    return e.u;
  }

  DestID_ GetSource(EdgePair<NodeID_, NodeWeight<NodeID_, WeightT_>> e) {
    return NodeWeight<NodeID_, WeightT_>(e.u, e.v.w);
  }

  CSRGraph<NodeID_, DestID_, invert> MakeGraph() {
    if (graph_filename_ != "") {
      auto suffix = GetSuffix(graph_filename_);
      if (suffix == ".sg" || suffix == ".wsg") {
        return readSerialized();
      } else {
        return readTextual();
      }
    } else {
      return generateGraph();
    }
  }

  // Relabels (and rebuilds) graph by order of decreasing degree
  static CSRGraph<NodeID_, DestID_, invert> RelabelByDegree(const CSRGraph<NodeID_, DestID_, invert>& g) {
    if (g.directed()) {
      std::cout << "Cannot relabel directed graph" << std::endl;
      std::exit(-11);
    }
    Timer t;
    t.Start();
    typedef std::pair<int64_t, NodeID_> degree_node_p;
    parallel::vector<degree_node_p> degree_id_pairs(g.num_nodes());
#pragma omp parallel for
    for (NodeID_ n = 0; n < g.num_nodes(); n++)
      degree_id_pairs[n] = std::make_pair(g.out_degree(n), n);
    std::sort(degree_id_pairs.begin(), degree_id_pairs.end(), std::greater<degree_node_p>());
    parallel::vector<NodeID_> degrees(g.num_nodes());
    parallel::vector<NodeID_> new_ids(g.num_nodes());
#pragma omp parallel for
    for (NodeID_ n = 0; n < g.num_nodes(); n++) {
      degrees[n] = degree_id_pairs[n].first;
      new_ids[degree_id_pairs[n].second] = n;
    }
    parallel::vector<SGOffset> offsets = ParallelPrefixSum(degrees);
    DestID_* neighs = new DestID_[offsets[g.num_nodes()]];
    DestID_** index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, neighs);
#pragma omp parallel for
    for (NodeID_ u = 0; u < g.num_nodes(); u++) {
      for (NodeID_ v : g.out_neigh(u))
        neighs[offsets[new_ids[u]]++] = new_ids[v];
      std::sort(index[new_ids[u]], index[new_ids[u] + 1]);
    }
    t.Stop();
    PrintTime("Relabel", t.Seconds());
    return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), index, neighs);
  }

private:
  template <typename RelNodeID_, typename RelDestID_>
  void RelabelEL(parallel::vector<EdgePair<RelNodeID_, RelDestID_>>& el) {
    using RelEdge = EdgePair<RelNodeID_, RelDestID_>;
    Timer t;
    t.Start();
    if (el.size() < (1 << 20)) {
      // Sequential
      std::unordered_map<RelNodeID_, RelNodeID_> mapping;
      RelNodeID_ i = 0;
      for (RelEdge& e : el) {
        if (auto it = mapping.find(e.u); it != mapping.end()) {
          e.u = it->second;
        } else {
          e.u = mapping[e.u] = i++;
        }

        RelNodeID_& v = static_cast<RelNodeID_&>(e.v); // if typeof(e.v) == NodeWeight, then static_cast<NodeID_&>(e) == Node&
        if (auto it = mapping.find(v); it != mapping.end()) {
          v = it->second;
        } else {
          v = mapping[v] = i++;
        }
      }
    } else {
      std::vector<std::unordered_set<RelNodeID_>> thread_local_sets(omp_get_max_threads());

#pragma omp parallel
      {
        std::unordered_set<RelNodeID_> local_nodes;
// Each thread processes a chunk of the edge list to find unique node ids
#pragma omp for nowait
        for (auto& e : el) {
          local_nodes.insert(e.u);
          const RelNodeID_& v = static_cast<RelNodeID_&>(e.v);
          local_nodes.insert(v);
        }

        thread_local_sets[omp_get_thread_num()] = std::move(local_nodes);
      }

      // Unique node ids are consolidated into a single set and new ids are calculated
      std::unordered_map<RelNodeID_, RelNodeID_> mapping;

      std::unordered_set<RelNodeID_> all_unique_nodes;
      for (const auto& s : thread_local_sets) {
        all_unique_nodes.insert(s.begin(), s.end());
      }
      RelNodeID_ i = 0;
      for (const RelNodeID_& old_id : all_unique_nodes) {
        mapping[old_id] = i++;
      }

      // Each thread updates the ids of a chunk of vertices
#pragma omp parallel for
      for (RelEdge& e : el) {
        e.u = mapping.at(e.u);
        RelNodeID_& v = static_cast<RelNodeID_&>(e.v);
        v = mapping.at(v);
      }
    }
    t.Stop();
    PrintTime("Relabeling Time", t.Seconds());
  }

  NodeID_ FindMaxNodeID(const EdgeList& el) {
    NodeID_ max_seen = 0;
#pragma omp parallel for reduction(max : max_seen)
    for (auto it = el.begin(); it < el.end(); it++) {
      Edge e = *it;
      max_seen = std::max(max_seen, e.u);
      max_seen = std::max(max_seen, static_cast<NodeID_>(e.v));
    }
    return max_seen;
  }

  parallel::vector<NodeID_> CountDegrees(const EdgeList& el, bool transpose) {
    parallel::vector<NodeID_> degrees(num_nodes_, 0);
#pragma omp parallel for
    for (auto it = el.begin(); it < el.end(); it++) {
      Edge e = *it;
      if (symmetrize_ || (!symmetrize_ && !transpose))
        fetch_and_add(degrees[e.u], 1);
      if (!symmetrize_ && transpose)
        fetch_and_add(degrees[static_cast<NodeID_>(e.v)], 1);
    }
    return degrees;
  }

  static parallel::vector<SGOffset> PrefixSum(const parallel::vector<NodeID_>& degrees) {
    parallel::vector<SGOffset> sums(degrees.size() + 1);
    SGOffset total = 0;
    for (size_t n = 0; n < degrees.size(); n++) {
      sums[n] = total;
      total += degrees[n];
    }
    sums[degrees.size()] = total;
    return sums;
  }

  static parallel::vector<SGOffset> ParallelPrefixSum(const parallel::vector<NodeID_>& degrees) {
    const size_t block_size = 1 << 20;
    const size_t num_blocks = (degrees.size() + block_size - 1) / block_size;
    parallel::vector<SGOffset> local_sums(num_blocks);
#pragma omp parallel for
    for (size_t block = 0; block < num_blocks; block++) {
      SGOffset lsum = 0;
      size_t block_end = std::min((block + 1) * block_size, degrees.size());
      for (size_t i = block * block_size; i < block_end; i++)
        lsum += degrees[i];
      local_sums[block] = lsum;
    }
    parallel::vector<SGOffset> bulk_prefix(num_blocks + 1);
    SGOffset total = 0;
    for (size_t block = 0; block < num_blocks; block++) {
      bulk_prefix[block] = total;
      total += local_sums[block];
    }
    bulk_prefix[num_blocks] = total;
    parallel::vector<SGOffset> prefix(degrees.size() + 1);
#pragma omp parallel for
    for (size_t block = 0; block < num_blocks; block++) {
      SGOffset local_total = bulk_prefix[block];
      size_t block_end = std::min((block + 1) * block_size, degrees.size());
      for (size_t i = block * block_size; i < block_end; i++) {
        prefix[i] = local_total;
        local_total += degrees[i];
      }
    }
    prefix[degrees.size()] = bulk_prefix[num_blocks];
    return prefix;
  }

  /*
  In-Place Graph Building Steps
    - sort edges and squish (remove self loops and redundant edges)
    - overwrite EdgeList's memory with outgoing neighbors
    - if graph not being symmetrized
      - finalize structures and make incoming structures if requested
    - if being symmetrized
      - search for needed inverses, make room for them, add them in place
  */
  void MakeCSRInPlace(EdgeList& el, DestID_*** index, DestID_** neighs, DestID_*** inv_index, DestID_** inv_neighs) {
    // Preprocessing of the EdgeList - sort & squish in place
    std::cout << "Edgelist size: " << el.size() << std::endl;

    // Removing parallel edges
    std::sort(el.begin(), el.end());
    auto old_end = el.end();
    auto new_end = std::unique(el.begin(), el.end());
    std::cout << "Redundant edges: " << std::distance(new_end, old_end) << std::endl;
    el.resize(new_end - el.begin());

    // Removing self-loops
    auto self_loop = [](Edge e) { return e.u == e.v; };
    old_end = el.end();
    new_end = std::remove_if(el.begin(), el.end(), self_loop);
    std::cout << "Self-loops: " << std::distance(new_end, old_end) << std::endl;
    el.resize(new_end - el.begin());

    // analyze EdgeList and repurpose it for outgoing edges
    // std::cout << "Counting degrees the first time" << std::endl;
    parallel::vector<NodeID_> degrees = CountDegrees(el, false);

    // std::cout << "Performing parallel prefix sum" << std::endl;
    parallel::vector<SGOffset> offsets = ParallelPrefixSum(degrees);

    // std::cout << "Counting degrees the second time" << std::endl;
    parallel::vector<NodeID_> indegrees = CountDegrees(el, true);

    // Reusing EdgeList data
    *neighs = reinterpret_cast<DestID_*>(el.data());
    // Iterate over the edge list, while at the same time modifying the underlying buffer
    // Only the destination vertex data is copied, and offsets are incremented in order to
    // push the next destination vertex into the next available slot
    // std::cout << "Copying out_edges and incrementing offsets" << std::endl;
    for (Edge e : el)
      (*neighs)[offsets[e.u]++] = e.v;

    size_t num_edges = el.size();
    el.leak();

    // Now the offsets array needs to be shifted down as they have been incremented,
    // this is simple, as offsets[i] is now offsets[i+1]
    // std::cout << "Shifting down offsets" << std::endl;
    for (NodeID_ n = num_nodes_; n >= 0; n--)
      offsets[n] = n != 0 ? offsets[n - 1] : 0;

    if (!symmetrize_) { // not going to symmetrize so no need to add edges
      size_t new_size = num_edges * sizeof(DestID_);
      // std::cout << "Reallocating" << std::endl;
      *neighs = static_cast<DestID_*>(std::realloc(*neighs, new_size));
      *index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, *neighs);

      if (invert) { // create inv_neighs & inv_index for incoming edges
        // std::cout << "Parallel prefix sum for in-offsets" << std::endl;
        parallel::vector<SGOffset> inoffsets = ParallelPrefixSum(indegrees);
        // std::cout << "Allocating space for offsets" << std::endl;
        *inv_neighs = new DestID_[inoffsets[num_nodes_]];
        *inv_index = CSRGraph<NodeID_, DestID_>::GenIndex(inoffsets, *inv_neighs);
        for (NodeID_ u = 0; u < num_nodes_; u++) {
          for (DestID_* it = (*index)[u]; it < (*index)[u + 1]; it++) {
            NodeID_ v = static_cast<NodeID_>(*it);
            if constexpr (WEIGHTED_BUILDER) {
              WeightT_ w = it->w;
              (*inv_neighs)[inoffsets[v]] = {u, w};
            } else {
              (*inv_neighs)[inoffsets[v]] = u;
            }
            inoffsets[v]++;
          }
        }
      }
    } else { // symmetrize graph by adding missing inverse edges
      // Step 1 - count number of needed inverses
      // std::cout << "Allocating and counting invs_needed" << std::endl;
      parallel::vector<NodeID_> invs_needed(num_nodes_, 0);
      for (NodeID_ u = 0; u < num_nodes_; u++) {
        for (SGOffset i = offsets[u]; i < offsets[u + 1]; i++) {
          DestID_ dest_v = (*neighs)[i];
          NodeID_ v = static_cast<NodeID_>(dest_v); // only get vertex

          bool inv_found = std::binary_search(
              *neighs + offsets[v],
              *neighs + offsets[v + 1],
              static_cast<DestID_>(u),
              [](const DestID_& a, const DestID_& b) {
                if constexpr (WEIGHTED_BUILDER) return a.v < b.v;
                else return a < b;
              }
          );
          if (!inv_found)
            invs_needed[v]++;
        }
      }

      // increase offsets to account for missing inverses, realloc neighs
      SGOffset total_missing_inv = 0;
      // std::cout << "Increasing offsets for missing inverses" << std::endl;
      for (NodeID_ n = 0; n < num_nodes_; n++) {
        offsets[n] += total_missing_inv;
        total_missing_inv += invs_needed[n];
      }
      offsets[num_nodes_] += total_missing_inv;

      // std::cout << "Reallocating neighs array for missing inverses" << std::endl;
      size_t newsize = (offsets[num_nodes_] * sizeof(DestID_));
      *neighs = static_cast<DestID_*>(std::realloc(*neighs, newsize));
      if (*neighs == nullptr) {
        std::cout << "Call to realloc() failed" << std::endl;
        exit(-33);
      }

      // Step 2 - spread out existing neighs to make room for inverses
      //   copies backwards (overwrites) and inserts free space at starts
      // std::cout << "Spreading existign neighs" << std::endl;
      SGOffset tail_index = offsets[num_nodes_] - 1;
      for (NodeID_ n = num_nodes_ - 1; n >= 0; n--) {
        SGOffset new_start = offsets[n] + invs_needed[n];
        for (SGOffset i = offsets[n + 1] - 1; i >= new_start; i--) {
          (*neighs)[tail_index] = (*neighs)[i - total_missing_inv];
          tail_index--;
        }
        total_missing_inv -= invs_needed[n];
        tail_index -= invs_needed[n];
      }

      // Step 3 - add missing inverse edges into free spaces from Step 2
      // std::cout << "Adding missing edges" << std::endl;
      for (NodeID_ u = 0; u < num_nodes_; u++) {
        for (SGOffset i = offsets[u] + invs_needed[u]; i < offsets[u + 1]; i++) {
          DestID_ dest_v = (*neighs)[i];
          NodeID_ v = static_cast<NodeID_>(dest_v);
          bool inv_found = std::binary_search(
              *neighs + offsets[v] + invs_needed[v],
              *neighs + offsets[v + 1],
              static_cast<DestID_>(u),
              [](const DestID_& a, const DestID_& b) {
                if constexpr (WEIGHTED_BUILDER) return a.v < b.v;
                else return a < b;
              }
          );
          if (!inv_found) {
            if constexpr (WEIGHTED_BUILDER) (*neighs)[offsets[v] + invs_needed[v] - 1] = {u, dest_v.w};
            else (*neighs)[offsets[v] + invs_needed[v] - 1] = u;
            invs_needed[v]--;
          }
        }
      }

      // std::cout << "Sorting edge lists" << std::endl;
      for (NodeID_ n = 0; n < num_nodes_; n++)
        std::sort(*neighs + offsets[n], *neighs + offsets[n + 1]);
      *index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, *neighs);
    }
  }

  CSRGraph<NodeID_, DestID_, invert> MakeGraphFromEL(EdgeList& el) {
    DestID_ **index = nullptr, **inv_index = nullptr;
    DestID_ *neighs = nullptr, *inv_neighs = nullptr;

    std::cout << "Making CSR from edge list" << std::endl;

    Timer t;
    t.Start();

    if (num_nodes_ == -1) {
      std::cout << "Finding Max Node ID" << std::endl;
      num_nodes_ = FindMaxNodeID(el) + 1;
    }
    num_edges_ = el.size();

    if constexpr (WEIGHTED_BUILDER) {
      if (needs_weights_ || generate_weights_) {
        std::cout << "Generating Weights" << std::endl;
        generateWeights(el);
      }
    }

    std::cout << "Making CSR in-place" << std::endl;
    MakeCSRInPlace(el, &index, &neighs, &inv_index, &inv_neighs);

    t.Stop();
    PrintTime("Build Time", t.Seconds());

    if (symmetrize_) return CSRGraph<NodeID_, DestID_, invert>(num_nodes_, index, neighs);
    else return CSRGraph<NodeID_, DestID_, invert>(num_nodes_, index, neighs, inv_index, inv_neighs);
  }

  template <typename ReprT>
  void generateWeights(ReprT& edge_repr) {
    if constexpr (std::is_integral_v<WeightT_>) {
      switch (weight_dist_) {

      case WeightGenerator::U_GAP_LEGACY: {
        std::cout << "Generating uniformly distributed weight in set {1, 255}" << std::endl;
        graph_utils::GAPLegacyDistribution<WeightT_> dist;
        graph_utils::replace_weights(edge_repr, dist, true);
        break;
      }

      case WeightGenerator::U_GAP: {
        std::cout << "Generating uniformly distributed weight in set {1, 255}" << std::endl;
        graph_utils::GAPDistribution<WeightT_> dist;
        graph_utils::replace_weights(edge_repr, dist, true);
        break;
      }

      default:
        std::cout << "Weight generator not supported for type " << typeid(WeightT_).name() << std::endl;
        std::exit(-1);
      }
    } else {
      switch (weight_dist_) {

      case WeightGenerator::U_JULIENNE: {
        std::cout << "Generating uniformly distributed weight in interval [0,logn)" << std::endl;
        graph_utils::JulienneDistribution<WeightT_> dist(static_cast<std::size_t>(num_nodes_));
        graph_utils::replace_weights(edge_repr, dist, true);
        break;
      }

      case WeightGenerator::U_GRAPH_500: {
        std::cout << "Generating uniformly distributed weight in interval [0,1)" << std::endl;
        graph_utils::Graph500Distribution<WeightT_> dist;
        graph_utils::replace_weights(edge_repr, dist, true);
        break;
      }

      case WeightGenerator::N_GRAPH_BASED: {
        std::cout << "Generating normally distibuted weights with mean=1.0 and stddev=sqrt(n/m)" << std::endl;
        graph_utils::GraphBasedNormalDistribution<WeightT_> dist(static_cast<std::size_t>(num_nodes_), static_cast<std::size_t>(num_edges_));
        graph_utils::replace_weights(edge_repr, dist, true);
        break;
      }

      default:
        std::cout << "Weight generator not supported for type " << typeid(WeightT_).name() << std::endl;
        std::exit(-1);
      }
    }
  }

  CSRGraph<NodeID_, DestID_, invert> readSerialized() {
    ReaderT r(graph_filename_);
    auto g = r.ReadSerializedGraph();
    num_nodes_ = g.num_nodes();
    num_edges_ = g.num_edges();

    if constexpr (WEIGHTED_BUILDER) {
      if (weights_filename_ != "") {
        std::cout << "Replacing weights" << std::endl;
        VectorReader<typename DestID_::WeightT> wreader(weights_filename_);
        g.ReplaceWeights(wreader.ReadSerialized("Weights"));
      }

      if (generate_weights_) {
        generateWeights(g);
      }
    }

    return g;
  }

  CSRGraph<NodeID_, DestID_, invert> readTextual() {
    CSRGraph<NodeID_, DestID_, invert> g;

    {
      EdgeList el;
      if (relabel_vertices_) {
        using LargeNodeID = int64_t;
        using LargeDestID = std::conditional_t<WEIGHTED_BUILDER, NodeWeight<LargeNodeID, WeightT_>, LargeNodeID>;
        using LargeReader = Reader<LargeNodeID, LargeDestID, WeightT_, invert>;

        LargeReader r(graph_filename_);
        auto result = r.ReadFile();

        // ratio: if matrix market format is used, the format can specify whether the graph is symmetric
        if (result.needs_symmetrize)
          symmetrize_ = true;

        // ratio: we can read from an unweighted textual graph, so if unweighted the weight will be generated later
        needs_weights_ = result.needs_weights;
        RelabelEL<LargeNodeID, LargeDestID>(result.el);

        el.reserve(el.size());
        for (const auto& e : result.el) {
          if constexpr (WEIGHTED_BUILDER) {
            el.push_back(Edge(static_cast<NodeID_>(e.u), NodeWeight(static_cast<NodeID_>(e.v.v), e.v.w)));
          } else {
            el.push_back(Edge(static_cast<NodeID_>(e.u), static_cast<NodeID_>(e.v)));
          }
        }

      } else {
        ReaderT r(graph_filename_);
        auto result = r.ReadFile();

        el = std::move(result.el);

        // ratio: if matrix market format is used, the format can specify whether the graph is symmetric
        if (result.needs_symmetrize)
          symmetrize_ = true;

        // ratio: we can read from an unweighted textual graph, so if unweighted the weight will be generated later
        needs_weights_ = result.needs_weights;
      }
      g = MakeGraphFromEL(el);
    }

    return g;
  }

  CSRGraph<NodeID_, DestID_, invert> generateGraph() {
    CSRGraph<NodeID_, DestID_, invert> g;

    {
      EdgeList el;
      Generator<NodeID_, DestID_> gen(synthetic_scale_, synthetic_degree_);
      el = gen.GenerateEL(graph_gen_ == GraphGenerator::UNIFORM);
      g = MakeGraphFromEL(el);
    }

    return g;
  }
};

#endif // BUILDER_H_
