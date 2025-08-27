// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef WRITER_H_
#define WRITER_H_

#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "command_line.h"
#include "graph.h"
#include "parallel/vector.h"

/*
GAP Benchmark Suite
Class:  Writer
Author: Scott Beamer

Given filename and graph, writes out the graph to storage
 - Should use WriteGraph(filename, serialized)
 - If serialized, will write out as serialized graph, otherwise, as edgelist
*/

template <typename NodeID_, typename DestID_ = NodeID_>
class WriterBase {
public:
  explicit WriterBase(CSRGraph<NodeID_, DestID_>& g) : g_(g) {}

  void WriteEL(std::fstream& out) {
    for (NodeID_ u = 0; u < g_.num_nodes(); u++) {
      for (DestID_ v : g_.out_neigh(u))
        out << u << " " << v << std::endl;
    }
  }

  void WriteMM(std::fstream& out) {
    constexpr size_t THRESHOLD = size_t(1) << 32; // 4 GiB
    std::stringstream buffer;

    // MM header
    buffer << "\%\%MatrixMarket matrix coordinate ";
    if constexpr (std::is_same_v<NodeID_, DestID_>)
      buffer << "pattern ";
    else if constexpr (std::is_integral_v<typename DestID_::WeightT>)
      buffer << "integer ";
    else
      buffer << "real ";

    if (g_.directed())
      buffer << "general";
    else
      buffer << "symmetric";
    buffer << std::endl;

    // M x N x NNZ
    buffer << g_.num_nodes() << " " << g_.num_nodes() << " " << g_.num_edges() << std::endl;

    for (NodeID_ u = 0; u < g_.num_nodes(); u++) {
      for (DestID_ v : g_.out_neigh(u))
        if (g_.directed() || u < v)
          buffer << u + 1 << " " << v + 1 << std::endl;

      if ((size_t)buffer.tellp() >= THRESHOLD) {
        out << buffer.rdbuf();
        std::stringstream().swap(buffer);
      }
    }
    out << buffer.rdbuf();
  }

  void WriteSerializedGraph(std::fstream& out) {
    if (!std::is_same<NodeID_, SGID>::value) {
      std::cout << "serialized graphs only allowed for 32b IDs" << std::endl;
      std::exit(-4);
    }
    // if (!std::is_same<DestID_, NodeID_>::value &&
    //     !std::is_same<DestID_, NodeWeight<NodeID_, SGID>>::value) {
    //   std::cout << ".wsg only allowed for int32_t weights" << std::endl;
    //   std::exit(-8);
    // }
    bool directed = g_.directed();
    SGOffset num_nodes = g_.num_nodes();
    SGOffset edges_to_write = g_.num_edges_directed();
    std::streamsize index_bytes = (num_nodes + 1) * sizeof(SGOffset);
    std::streamsize neigh_bytes;
    if (std::is_same<DestID_, NodeID_>::value)
      neigh_bytes = edges_to_write * sizeof(SGID);
    else
      neigh_bytes = edges_to_write * sizeof(NodeWeight<NodeID_, SGID>);
    out.write(reinterpret_cast<char*>(&directed), sizeof(bool));
    out.write(reinterpret_cast<char*>(&edges_to_write), sizeof(SGOffset));
    out.write(reinterpret_cast<char*>(&num_nodes), sizeof(SGOffset));
    parallel::vector<SGOffset> offsets = g_.VertexOffsets(false);
    out.write(reinterpret_cast<char*>(offsets.data()), index_bytes);
    out.write(reinterpret_cast<char*>(g_.out_neigh(0).begin()), neigh_bytes);
    if (directed) {
      offsets = g_.VertexOffsets(true);
      out.write(reinterpret_cast<char*>(offsets.data()), index_bytes);
      out.write(reinterpret_cast<char*>(g_.in_neigh(0).begin()), neigh_bytes);
    }
  }

  void WriteGraph(std::string filename, OutputFormat format) {
    if (filename == "") {
      std::cout << "No output filename given (Use -h for help)" << std::endl;
      std::exit(-8);
    }
    std::fstream file(filename, std::ios::out | std::ios::binary);
    if (!file) {
      std::cout << "Couldn't write to file " << filename << std::endl;
      std::exit(-5);
    }
    file << std::setprecision(std::numeric_limits<float>::max_digits10);
    switch (format) {
    case GAP_BINARY:
      WriteSerializedGraph(file);
      break;
    case EDGE_LIST:
      WriteEL(file);
      break;
    case MATRIX_MARKET:
      WriteMM(file);
      break;
    default:
      std::cout << "Output format not supported." << std::endl;
      break;
    }
    file.close();
  }

protected:
  CSRGraph<NodeID_, DestID_>& g_;
};

// Specialized writer for writing only the largest connected components
// The writer always gets an unweighted graph, and uses needs_weight to generate weights depending on the CLI options
template <typename NodeID_>
class LargestComponentWriter : public WriterBase<NodeID_> {
  using rng_t_ = std::mt19937;

  struct LCInfo {
    NodeID_ id;
    NodeID_ num_nodes;
    int64_t num_edges;
    NodeID_ max_node;
  };

public:
  explicit LargestComponentWriter(CSRGraph<NodeID_>& g, parallel::vector<NodeID_>& components)
      : WriterBase<NodeID_>(g), components_(components), largest_component_(largest_component()) {
  }

  void WriteMM(std::fstream& out, bool needs_weights) {

    constexpr size_t THRESHOLD = size_t(1) << 32; // 4 GiB
    std::stringstream buffer;
    rng_t_ rng;
    std::normal_distribution<float> ndist(
        static_cast<float>(1.0),
        static_cast<float>(std::sqrt(static_cast<double>(largest_component_.num_nodes) / largest_component_.num_edges))
    );
    rng.seed(kRandSeed);

    // MM header
    buffer << "\%\%MatrixMarket matrix coordinate ";
    if (needs_weights) {
      // TODO: implement integer weights
      if (true) {
        buffer << "real ";
      } else {
        buffer << "integer ";
      }
    } else {
      buffer << "pattern ";
    }

    if (this->g_.directed())
      buffer << "general";
    else
      buffer << "symmetric";
    buffer << std::endl;

    int64_t lc_edges = 0;

    // M x N x NNZ
    buffer << largest_component_.max_node + 1 << " " << largest_component_.max_node + 1 << " " << largest_component_.num_edges << std::endl;

    for (NodeID_ u = 0; u < largest_component_.max_node + 1; u++) {
      for (NodeID_ v : this->g_.out_neigh(u))
        if (this->g_.directed() || u < v) {
          if (components_[u] == largest_component_.id && components_[v] == largest_component_.id) {
            buffer << u + 1 << " " << v + 1;
            if (needs_weights) {
              float weight;
              while ((weight = ndist(rng)) <= 0)
                ;
              buffer << " " << weight;
            }
            buffer << std::endl;
          }
        }

      if ((size_t)buffer.tellp() >= THRESHOLD) {
        out << buffer.rdbuf();
        std::stringstream().swap(buffer);
      }
    }
    out << buffer.rdbuf();
  }

  void WriteGraph(std::string filename, OutputFormat format, bool needs_weights) {
    if (filename == "") {
      std::cout << "No output filename given (Use -h for help)" << std::endl;
      std::exit(-8);
    }
    std::fstream file(filename, std::ios::out | std::ios::binary);
    if (!file) {
      std::cout << "Couldn't write to file " << filename << std::endl;
      std::exit(-5);
    }

    switch (format) {
    case MATRIX_MARKET:
      WriteMM(file, needs_weights);
      break;
    default:
      std::cout << "Output format not supported." << std::endl;
      break;
    }

    file.close();
  }

private:
  parallel::vector<NodeID_>& components_;
  LCInfo largest_component_;

  LCInfo largest_component() {
    std::unordered_map<NodeID_, NodeID_> count;
    for (NodeID_ comp_i : components_)
      count[comp_i] += 1;

    std::pair<NodeID_, NodeID_> max_pair = {0, 0};
    for (auto kv_pair : count) {
      if (kv_pair.second > max_pair.second)
        max_pair = kv_pair;
    }

    int64_t lc_edges = 0;
    NodeID_ max_node = 0;
#pragma omp parallel for reduction(+ : lc_edges) reduction(max : max_node)
    for (NodeID_ u = 0; u < this->g_.num_nodes(); u++) {
      for (NodeID_ v : this->g_.out_neigh(u)) {
        if (this->g_.directed() || u < v) {
          if (components_[u] == max_pair.first && components_[v] == max_pair.first) {
            lc_edges++;
            if (u > max_node)
              max_node = u;
            if (v > max_node)
              max_node = v;
          }
        }
      }
    }

    std::cout << "Largest Component: " << std::endl;
    std::cout << "component id: " << max_pair.first << std::endl
              << " - num_nodes: " << max_pair.second << std::endl
              << " - num_edges: " << lc_edges << std::endl
              << " - max_node: " << max_node << std::endl;

    return LCInfo{max_pair.first, max_pair.second, lc_edges, max_node};
  }
};
#endif // WRITER_H_
