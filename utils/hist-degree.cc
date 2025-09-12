#include <atomic>
#include <filesystem>
#include <iostream>
#include <string>

#include "benchmark.h"
#include "command_line.h"

using namespace std::string_literals;

int main(int argc, char* argv[]) {
  CLStats cli(argc, argv, "degree distribution");
  cli.parse();

  WeightedBuilder b(cli);
  WGraph g = b.MakeGraph();

  std::vector<std::atomic<int32_t>> degree_hist(g.num_nodes());

  std::cout << "starting counting" << std::endl;

#pragma omp parallel for
  for (auto i = 0; i < g.num_nodes(); i++) {
    degree_hist[g.out_degree(i)].fetch_add(1);
  }

  std::string graph_name = std::filesystem::path(cli.graph_filename()).filename().stem().string();
  std::filesystem::path out_dir = std::filesystem::path(cli.out_directory());

  std::ofstream weights_file;
  weights_file.open(out_dir / (graph_name + "_degree.csv"s));
  weights_file << "degree," << "frequency\n";
  for (auto i = 0; i < degree_hist.size(); i++) {
    if (degree_hist[i] > 0)
      weights_file << i << "," << degree_hist[i] << "\n";
  }

  return 0;
}
