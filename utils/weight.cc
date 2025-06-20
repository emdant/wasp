#include <atomic>
#include <filesystem>
#include <iostream>
#include <string>

#include "benchmark.h"
#include "command_line.h"

using namespace std::string_literals;

int main(int argc, char* argv[]) {
  CLStats cli(argc, argv, "weights");
  if (!cli.ParseArgs())
    return -1;

  WeightedBuilder b(cli);
  WGraph g = b.MakeGraph();

  std::vector<std::atomic<int64_t>> weight_hist((int)4e8);
  std::atomic<int64_t> num_zeros{0};

  std::cout << "starting counting" << std::endl;

#pragma omp parallel for
  for (auto i = 0; i < g.num_nodes(); i++) {
    for (WNode wn : g.out_neigh(i)) {
      if (wn.w == 0)
        num_zeros.fetch_add(1);
      weight_hist[wn.w].fetch_add(1);
    }
  }

  std::cout << "Number of zero-weight edges " << num_zeros << std::endl;

  std::string graph_name = std::filesystem::path(cli.filename()).filename().stem().string();
  std::filesystem::path out_dir = std::filesystem::path(cli.out_directory());

  std::ofstream weights_file;
  weights_file.open(out_dir / (graph_name + "_weight.csv"s));
  weights_file << "weight," << "frequency\n";
  for (auto i = 0; i < weight_hist.size(); i++) {
    if (weight_hist[i] > 0)
      weights_file << i << "," << weight_hist[i] << "\n";
  }

  return 0;
}
