// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef READER_H_
#define READER_H_

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iosfwd>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <type_traits>
#include <unistd.h>
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
    std::string line;
    std::istringstream edge_stream;

    while (std::getline(in, line)) {
      if (line.empty())
        continue;

      edge_stream.clear();   // recover from fail states
      edge_stream.str(line); // change underlying string
      edge_stream >> e;

      if (edge_stream.fail()) {
        // Decide on an error strategy: throw, print a warning, or skip.
        // For now, we'll just skip the malformed line.
        std::cerr << "Warning: Skipping malformed line: '" << line << "'\n";
        continue;
      }

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

      if (u == 0) {
        std::cout << "Vertex with id 0 found. Aborting." << std::endl;
        std::exit(-1);
      }

      if (has_weights && !ignore_weights) {
        NodeWeight<NodeID_, WeightT_> v;
        edge_stream >> v;
        if (v.v == 0) {
          std::cout << "Vertex with id 0 found. Aborting." << std::endl;
          std::exit(-1);
        }

        v.v -= 1;
        result.el.push_back(Edge(u - 1, v));
      } else {
        NodeID_ v;
        edge_stream >> v;
        if (v == 0) {
          std::cout << "Vertex with id 0 found. Aborting." << std::endl;
          std::exit(-1);
        }

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

    int fd = open(filename_.c_str(), O_RDONLY);
    if (fd == -1) {
      std::perror("Error opening file");
      std::exit(-1);
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
      std::perror("Error getting file size");
      close(fd);
      std::exit(-1);
    }
    size_t file_size = sb.st_size;

    char* file_ptr = static_cast<char*>(mmap(NULL, file_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0));
    if (file_ptr == MAP_FAILED) {
      std::perror("Error mapping file");
      close(fd);
      std::exit(-1);
    }
    close(fd);
    char* current_ptr = file_ptr;

    Timer t;
    t.Start();

    bool directed;
    SGOffset num_nodes, num_edges;
    DestID_ **index = nullptr, **inv_index = nullptr;
    DestID_ *neighs = nullptr, *inv_neighs = nullptr;

    directed = *(reinterpret_cast<bool*>(current_ptr));
    current_ptr += sizeof(bool);
    num_edges = *(reinterpret_cast<SGOffset*>(current_ptr));
    current_ptr += sizeof(SGOffset);
    num_nodes = *(reinterpret_cast<SGOffset*>(current_ptr));
    current_ptr += sizeof(SGOffset);

    size_t num_offsets = num_nodes + 1;
    parallel::vector<SGOffset> offsets(num_offsets);
    size_t num_offset_bytes = num_offsets * sizeof(SGOffset);
    memcpy(offsets.data(), current_ptr, num_offset_bytes);
    current_ptr += num_offset_bytes;

    neighs = new DestID_[num_edges];

    if (!(WEIGHTED_READER != weighted)) { // weighted reader and weighted graph or unweighted reader and unweighted graph
      std::streamsize num_neigh_bytes = num_edges * sizeof(DestID_);
      memcpy(neighs, current_ptr, num_neigh_bytes);
      current_ptr += num_neigh_bytes;
    } else { // unweighted reader and weighted graph, the opposite is not possible (exited early)
      struct WEdge {
        NodeID_ destination;
        WeightT_ weight;
      };
      const WEdge* edge_data = reinterpret_cast<const WEdge*>(current_ptr);
      for (SGOffset i = 0; i < num_edges; ++i) {
        neighs[i] = static_cast<DestID_>(edge_data[i].destination);
      }
      current_ptr += num_edges * sizeof(WEdge);
    }
    index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, neighs);

    if (directed && invert) {
      memcpy(offsets.data(), current_ptr, num_offset_bytes);
      current_ptr += num_offset_bytes;

      inv_neighs = new DestID_[num_edges];

      if (!(WEIGHTED_READER != weighted)) { // weighted reader and weighted graph or unweighted reader and unweighted graph
        std::streamsize num_neigh_bytes = num_edges * sizeof(DestID_);
        memcpy(inv_neighs, current_ptr, num_neigh_bytes);
        current_ptr += num_neigh_bytes;
      } else { // unweighted reader and weighted graph, the opposite is not possible (exited early)
        struct WeightedEdge {
          NodeID_ destination;
          WeightT_ weight;
        };
        const WeightedEdge* edge_data = reinterpret_cast<const WeightedEdge*>(current_ptr);
        for (SGOffset i = 0; i < num_edges; ++i) {
          inv_neighs[i] = static_cast<DestID_>(edge_data[i].destination);
        }
        current_ptr += num_edges * sizeof(WeightedEdge);
      }

      inv_index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, inv_neighs);
    }
    munmap(file_ptr, file_size);
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

  std::vector<ValueT_> Read(const std::string& values_name) {
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

    PrintTime(values_name + " Read Time", t.Seconds());

    return sources;
  }

  std::vector<ValueT_> ReadSerialized(const std::string& values_name) {
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
    PrintTime(values_name + " Read Time", t.Seconds());

    return values;
  }
};

#endif // READER_H_
