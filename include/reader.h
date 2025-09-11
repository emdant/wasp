// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef READER_H_
#define READER_H_

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iosfwd>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "graph.h"
#include "parallel/vector.h"
#include "timer.h"
#include "util.h"

/*
GAP Benchmark Suite
Class:  Reader
Author: Scott Beamer

Given filename, returns an edgelist or the entire graph (if serialized)
 - Intended to be called from Builder
 - Determines file format from the filename's suffix
 - If the input graph is serialized (.sg or .wsg), reads the graph
   directly into the returned graph instance
 - Otherwise, reads the file and returns an edgelist
*/

template <typename NodeID_, typename DestID_ = NodeID_, typename WeightT_ = NodeID_, bool invert = true>
class Reader {
  static constexpr bool WEIGHTED_READER = !std::is_same_v<NodeID_, DestID_>;
  typedef EdgePair<NodeID_, DestID_> Edge;
  typedef parallel::vector<Edge> EdgeList;
  std::string filename_;

public:
  struct ReadFileResult {
    EdgeList el;
    bool needs_symmetrize;
    bool needs_weights;

    operator std::tuple<EdgeList&, bool&, bool&>() {
      return std::tie(el, needs_symmetrize, needs_weights);
    }
  };

  explicit Reader(std::string filename) : filename_(filename) {
  }

  /* Will read both weighted and unweighted edge lists, as DestID_ can be a NodeWeight */
  ReadFileResult ReadInEL(std::ifstream& in, bool one_indexed) {
    ReadFileResult result;

    result.needs_symmetrize = false; // we assume edge lists are directed

    while (true) {
      char c = in.peek();
      if (c == '%') {
        in.ignore(200, '\n');
      } else {
        break;
      }
    }

    Edge e;
    bool has_weights = false;
    std::string line;
    std::istringstream edge_stream;

    while (std::getline(in, line)) {
      if (line.empty())
        continue;

      edge_stream.clear();   // recover from fail states
      edge_stream.str(line); // change underlying string
      edge_stream >> e;
      if (one_indexed) {
        e.u--;
        static_cast<NodeID_&>(e.v)--;
      }
      result.el.push_back(e);
    }

    // if edge_stream failed, it means it did not read a weight when needed
    // therefore edge weights must be generated
    if (WEIGHTED_READER && edge_stream.fail())
      result.needs_weights = true;
    else
      result.needs_weights = false;

    return result;
  }

  // Note: converts vertex numbering from 1..N to 0..N-1
  ReadFileResult ReadInGR(std::ifstream& in) {
    ReadFileResult result;
    char c;
    Edge e;
    std::string line;

    result.needs_symmetrize = false; // we assume edge lists are directed

    std::istringstream edge_stream;
    while (std::getline(in, line)) {
      if (line.empty())
        continue;

      edge_stream.clear();
      edge_stream.str(line);

      c = edge_stream.peek();
      if (c == 'a') {
        edge_stream >> c >> e;
        e.u--;
        static_cast<NodeID_&>(e.v)--; // Cast to NodeID_ in case e.v is a NodeWeight
        result.el.push_back(e);
      }
    }

    if (WEIGHTED_READER && edge_stream.fail())
      result.needs_weights = true;
    else
      result.needs_weights = false;

    return result;
  }

  // Note: converts vertex numbering from 1..N to 0..N-1
  ReadFileResult ReadInMetis(std::ifstream& in) {
    ReadFileResult result;
    NodeID_ num_nodes, num_edges;
    char c;
    std::string line;
    bool read_weights = false;

    while (true) {
      c = in.peek();
      if (c == '%') {
        in.ignore(200, '\n');
      } else {
        std::getline(in, line, '\n');
        std::istringstream header_stream(line);
        header_stream >> num_nodes >> num_edges;
        header_stream >> std::ws;
        if (!header_stream.eof()) {
          int32_t fmt;
          header_stream >> fmt;
          if (fmt == 1) {
            read_weights = true;
          } else if ((fmt != 0) && (fmt != 100)) {
            std::cout << "Do not support METIS fmt type: " << fmt << std::endl;
            std::exit(-20);
          }
        }
        break;
      }
    }

    if (read_weights && !WEIGHTED_READER) {
      std::cout << "Trying to read weights using a non-weighted Builder, use WeightedBuilder." << std::endl;
      std::exit(-1);
    }
    if (!read_weights && WEIGHTED_READER) {
      std::cout << "Trying to read an unweighted graph using a WeightedBuilder, use a Builder" << std::endl;
      std::exit(-1);
    }

    NodeID_ u = 0;
    while (u < num_nodes) {
      c = in.peek();
      if (c == '%') {
        in.ignore(200, '\n');
      } else {
        std::getline(in, line);
        if (line != "") {
          std::istringstream edge_stream(line);
          if (read_weights) {
            NodeWeight<NodeID_, WeightT_> v;
            while (edge_stream >> v >> std::ws) {
              v.v -= 1;
              result.el.push_back(Edge(u, v));
            }
          } else {
            NodeID_ v;
            while (edge_stream >> v >> std::ws) {
              result.el.push_back(Edge(u, v - 1));
            }
          }
        }
        u++;
      }
    }

    result.needs_weights = false; // If we are using a weighted reader, we expect a weighted file
    result.needs_symmetrize = false;
    return result;
  }

  /* Reads a Matrix Market file.
      - if the Reader is unweighted: the MTX file can be weighted, and weights will be ignored, the `needs_weight` flag will be false.
      - if the Reader in weighted: the MTX file can be unweighted, and the `needs_weight` flag will be true.
    Converts vertex numbering from 1..N to 0..N-1
  */
  ReadFileResult ReadInMTX(std::ifstream& in) {
    ReadFileResult result;

    std::string start, object, format, field, symmetry, line;
    in >> start >> object >> format >> field >> symmetry >> std::ws;
    if (start != "%%MatrixMarket") {
      std::cout << ".mtx file did not start with %%MatrixMarket" << std::endl;
      std::exit(-21);
    }
    if ((object != "matrix") || (format != "coordinate")) {
      std::cout << "only allow matrix coordinate format for .mtx" << std::endl;
      std::exit(-22);
    }
    if (field == "complex") {
      std::cout << "do not support complex weights for .mtx" << std::endl;
      std::exit(-23);
    }

    bool ignore_weights = false;
    bool has_weights;
    if (field == "pattern") {
      has_weights = false;
    } else if ((field == "real") || (field == "double") || (field == "integer")) {
      has_weights = true;
      if (!WEIGHTED_READER)
        ignore_weights = true;
    } else {
      std::cout << "unrecognized field type for .mtx" << std::endl;
      std::exit(-24);
    }

    if (symmetry == "symmetric") {
      result.needs_symmetrize = true; // MakeCSR will automatically symmetrize the edges
    } else if ((symmetry == "general") || (symmetry == "skew-symmetric")) {
      result.needs_symmetrize = false;
    } else {
      std::cout << "unsupported symmetry type for .mtx" << std::endl;
      std::exit(-25);
    }

    while (true) {
      char c = in.peek();
      if (c == '%') {
        in.ignore(200, '\n');
      } else {
        break;
      }
    }

    int64_t m, n, nonzeros;
    in >> m >> n >> nonzeros >> std::ws;
    if (m != n) {
      std::cout << m << " " << n << " " << nonzeros << std::endl;
      std::cout << "matrix must be square for .mtx" << std::endl;
      std::exit(-26);
    }

    while (std::getline(in, line)) {
      if (line.empty())
        continue;

      std::istringstream edge_stream(line);
      NodeID_ u;
      edge_stream >> u;
      if (has_weights && !ignore_weights) {
        NodeWeight<NodeID_, WeightT_> v;
        edge_stream >> v;
        v.v -= 1;
        result.el.push_back(Edge(u - 1, v));
      } else {
        NodeID_ v;
        edge_stream >> v;
        result.el.push_back(Edge(u - 1, v - 1));
        if (ignore_weights)
          edge_stream.ignore(200, '\n');
      }
    }

    if (WEIGHTED_READER && !has_weights)
      result.needs_weights = true;
    else
      result.needs_weights = false;
    return result;
  }

  ReadFileResult ReadFile() {
    Timer t;
    t.Start();
    ReadFileResult result;

    std::string suffix = GetSuffix(filename_);
    std::ifstream file(filename_);
    if (!file.is_open()) {
      std::cout << "Couldn't open file " << filename_ << std::endl;
      std::exit(-2);
    }

    std::cout << "Reading file: " << filename_ << std::endl;
    if (suffix == ".el" || suffix == ".wel") {
      result = ReadInEL(file, false);
    } else if (suffix == ".tsv") {
      result = ReadInEL(file, true);
    } else if (suffix == ".gr") {
      result = ReadInGR(file);
    } else if (suffix == ".graph") {
      result = ReadInMetis(file);
    } else if (suffix == ".mtx") {
      result = ReadInMTX(file);
    } else {
      std::cout << "Unrecognized suffix: " << suffix << std::endl;
      std::exit(-3);
    }

    file.close();
    t.Stop();

    PrintTime("Read Time", t.Seconds());

    return result;
  }

  CSRGraph<NodeID_, DestID_, invert> ReadSerializedGraph() {
    bool weighted = GetSuffix(filename_) == ".wsg";
    if (!std::is_same<NodeID_, SGID>::value) {
      std::cout << "serialized graphs only allowed for 32bit" << std::endl;
      std::exit(-5);
    }
    if (!weighted && !std::is_same<NodeID_, DestID_>::value) {
      std::cout << ".sg not allowed for weighted graphs" << std::endl;
      std::exit(-5);
    }

    std::ifstream file(filename_);
    if (!file.is_open()) {
      std::cout << "Couldn't open file " << filename_ << std::endl;
      std::exit(-6);
    }

    Timer t;
    t.Start();

    bool directed;
    SGOffset num_nodes, num_edges;
    DestID_ **index = nullptr, **inv_index = nullptr;
    DestID_ *neighs = nullptr, *inv_neighs = nullptr;
    file.read(reinterpret_cast<char*>(&directed), sizeof(bool));
    file.read(reinterpret_cast<char*>(&num_edges), sizeof(SGOffset));
    file.read(reinterpret_cast<char*>(&num_nodes), sizeof(SGOffset));

    parallel::vector<SGOffset> offsets(num_nodes + 1);
    neighs = new DestID_[num_edges];
    std::streamsize num_index_bytes = (num_nodes + 1) * sizeof(SGOffset);
    std::streamsize num_neigh_bytes = num_edges * sizeof(DestID_);
    file.read(reinterpret_cast<char*>(offsets.data()), num_index_bytes);
    if (!(WEIGHTED_READER != weighted)) { // weighted reader and weighted graph or unweighted reader and unweighted graph
      file.read(reinterpret_cast<char*>(neighs), num_neigh_bytes);
    } else { // unweighted reader and weighted graph, the opposite is not possible (exited early)
      for (SGOffset i = 0; i < num_edges; i++) {
        file.read(reinterpret_cast<char*>(neighs + i), sizeof(NodeID_));
        file.ignore(sizeof(WeightT_)); // will be 32-bit anyway
      }
    }

    index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, neighs);
    if (directed && invert) {
      inv_neighs = new DestID_[num_edges];
      file.read(reinterpret_cast<char*>(offsets.data()), num_index_bytes);
      if (!(WEIGHTED_READER != weighted)) { // weighted reader and weighted graph or unweighted reader and unweighted graph
        file.read(reinterpret_cast<char*>(inv_neighs), num_neigh_bytes);
      } else { // unweighted reader and weighted graph, the opposite is not possible (exited early)
        for (SGOffset i = 0; i < num_edges; i++) {
          file.read(reinterpret_cast<char*>(inv_neighs + i), sizeof(NodeID_));
          file.ignore(sizeof(WeightT_)); // will be 32-bit anyway
        }
      }

      inv_index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, inv_neighs);
    }
    file.close();
    t.Stop();

    PrintTime("Read Time", t.Seconds());

    if (directed)
      return CSRGraph<NodeID_, DestID_, invert>(num_nodes, index, neighs, inv_index, inv_neighs);
    else
      return CSRGraph<NodeID_, DestID_, invert>(num_nodes, index, neighs);
  }
};

template <typename ValueT_>
class VectorReader {
  std::string filename_;

public:
  explicit VectorReader(std::string filename) : filename_(filename) {
    if (filename == "") {
      std::cout << "No sources filename given (Use -h for help)" << std::endl;
      std::exit(-8);
    }
  }

  std::vector<ValueT_> Read() {
    std::ifstream file(filename_, std::ios::binary);
    if (!file.is_open()) {
      std::cout << "Couldn't open file " << filename_ << std::endl;
      std::exit(-2);
    }

    Timer t;
    t.Start();
    std::vector<ValueT_> sources;
    while (!file.eof()) {
      ValueT_ source;
      file >> source;
      sources.push_back(source);
    }
    file.close();

    t.Stop();
    PrintTime("Values Read Time", t.Seconds());

    return sources;
  }

  std::vector<ValueT_> ReadSerialized() {
    std::ifstream file(filename_, std::ios::binary);
    if (!file.is_open()) {
      std::cout << "Couldn't open file " << filename_ << std::endl;
      std::exit(-2);
    }

    Timer t;
    t.Start();
    std::vector<ValueT_> values;
    int64_t num_values; // must be 64-bit value
    file.read(reinterpret_cast<char*>(&num_values), sizeof(num_values));

    values.resize(num_values);
    file.read(reinterpret_cast<char*>(values.data()), num_values * sizeof(ValueT_));
    file.close();

    t.Stop();
    PrintTime("Serialized Values Read Time", t.Seconds());

    return values;
  }
};

#endif // READER_H_
