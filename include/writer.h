// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef WRITER_H_
#define WRITER_H_

#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

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
    buffer << "%%MatrixMarket matrix coordinate ";
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

      if (static_cast<size_t>(buffer.tellp()) >= THRESHOLD) {
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

template <typename NodeID_>
class SourcesWriter {
  const std::vector<NodeID_>& sources_;

public:
  explicit SourcesWriter(const std::vector<NodeID_>& sources) : sources_(sources) {}

  void WriteSources(std::string filename) {
    if (filename == "") {
      std::cout << "No output filename given (Use -h for help)" << std::endl;
      std::exit(-8);
    }
    std::fstream file(filename, std::ios::out | std::ios::binary);
    if (!file) {
      std::cout << "Couldn't write to file " << filename << std::endl;
      std::exit(-5);
    }

    for (std::size_t i = 0; i < sources_.size(); i++) {
      file << sources_[i] << std::endl;
    }

    file.close();
    std::cout << "Sources written to file " << filename << std::endl;
  }
};

template <typename NodeID_, typename WeightT_>
class WeightsWriter {
  using Graph = CSRGraph<NodeID_, NodeWeight<NodeID_, WeightT_>>;

  const Graph& g_;

public:
  explicit WeightsWriter(const Graph& g) : g_(g) {}

  void WriteWeights(std::string filename) {
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
    for (NodeID_ i = 0; i < g_.num_nodes(); i++) {
      for (NodeWeight<NodeID_, WeightT_>& nw : g_.out_neigh(i)) {
        file << nw.w << std::endl;
      }
    }

    file.close();
    std::cout << "Weights written to file " << filename << std::endl;
  }

  void WriteWeightsSerialized(std::string filename) {
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

    auto num_edges = g_.num_edges();
    file.write(reinterpret_cast<char*>(&num_edges), sizeof(num_edges));

    for (NodeID_ i = 0; i < g_.num_nodes(); i++) {
      for (NodeWeight<NodeID_, WeightT_>& nw : g_.out_neigh(i)) {
        file.write(reinterpret_cast<char*>(&(nw.w)), sizeof(WeightT_));
      }
    }

    file.close();
    std::cout << "Weights written to file " << filename << std::endl;
  }
};

template <typename NodeID_, typename DestID_>
WeightsWriter(CSRGraph<NodeID_, DestID_>) -> WeightsWriter<NodeID_, typename DestID_::WeightT>;

#endif // WRITER_H_
