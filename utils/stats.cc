#include <filesystem>
#include <iostream>
#include <map>
#include <string>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"

using namespace std::string_literals;

int main(int argc, char* argv[]) {
  CLStats cli(argc, argv, "graph-stats");
  if (!cli.ParseArgs())
    return -1;

  WeightedBuilder b(cli);
  WGraph g = b.MakeGraph();

  std::map<NodeID, NodeID> out_hist;

  std::size_t n1 = 0;
  std::size_t max_deg = 0;
  NodeID max_v;
  for (auto i = 0; i < g.num_nodes(); i++) {
    if (g.out_degree(i) > max_deg) {
      max_deg = g.out_degree(i);
      max_v = i;
    }
    if (g.out_degree(i) == 1)
      n1++;
    out_hist[g.out_degree(i)] += 1;
  }

  std::size_t n_deg1 = 0;
  for (WNode wn : g.out_neigh(max_v)) {
    if (g.out_degree(wn.v) == 1)
      n_deg1++;
  }
  std::cout << "number of vertices: " << g.num_nodes() << std::endl;

  std::cout << "vertices with deg=1: " << n1 << std::endl;
  std::cout << "\% vertices with deg=1: " << (double)n1 / g.num_nodes() * 100 << std::endl;

  std::cout << "vertices with deg=1 connected to max deg vertex: " << n_deg1 << std::endl;
  std::cout << "\% vertices with deg=1 connected to max deg vertex: " << (double)n_deg1 / max_deg * 100 << std::endl;

  std::cout << "\% vertices connected to max deg vertex: " << (double)max_deg / g.num_nodes() * 100 << std::endl;
  std::cout << "\% vertices with deg=1 connected to max deg vertex over the whole number of nodes: " << (double)n_deg1 / g.num_nodes() * 100 << std::endl;

  std::string graph_name = std::filesystem::path(cli.filename()).filename().stem().string();
  std::filesystem::path out_dir = std::filesystem::path(cli.out_directory());

  std::ofstream out_hist_file;
  out_hist_file.open(out_dir / (graph_name + "_outdeg.csv"s));
  out_hist_file << "degree," << "frequency\n";
  for (std::pair<const NodeID, NodeID>& pair : out_hist) {
    out_hist_file << pair.first << "," << pair.second << "\n";
  }
  out_hist_file.close();

  if (g.directed()) {
    std::map<NodeID, NodeID> in_hist;

    for (auto i = 0; i < g.num_nodes(); i++) {
      out_hist[g.out_degree(i)] += 1;
      in_hist[g.in_degree(i)] += 1;
    }

    std::ofstream in_hist_file;

    in_hist_file.open(out_dir / (graph_name + "_indeg.csv"s));
    in_hist_file << "degree," << "frequency\n";
    for (std::pair<const NodeID, NodeID>& pair : in_hist) {
      in_hist_file << pair.first << "," << pair.second << "\n";
    }
    in_hist_file.close();
  }

  return 0;
}
