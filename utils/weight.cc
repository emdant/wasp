#include <atomic>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>

#include "benchmark.h"
#include "command_line.h"

using namespace std::string_literals;

int main(int argc, char* argv[]) {

  CLStats cli(argc, argv, "weights distribution");
  cli.parse();

  WeightedBuilder b(cli);
  WGraph g = b.MakeGraph();

#ifndef USE_FLOAT
  // ====================== INTEGER WEIGHTS ======================
  std::vector<std::atomic<int64_t>> weight_hist((int)4e8);
  std::atomic<int64_t> num_zeros{0};

  int64_t min_w = std::numeric_limits<int64_t>::max();
  int64_t max_w = std::numeric_limits<int64_t>::min();

  std::cout << "starting counting" << std::endl;

#pragma omp parallel
  {
    int64_t local_min = min_w;
    int64_t local_max = max_w;

#pragma omp for nowait
    for (auto i = 0; i < g.num_nodes(); i++) {
      for (WNode wn : g.out_neigh(i)) {
        if (wn.w < local_min)
          local_min = wn.w;
        if (wn.w > local_max)
          local_max = wn.w;
        if (wn.w == 0)
          num_zeros.fetch_add(1);
        weight_hist[wn.w].fetch_add(1);
      }
    }

#pragma omp critical
    {
      if (local_min < min_w)
        min_w = local_min;
      if (local_max > max_w)
        max_w = local_max;
    }
  }

  std::cout << "Number of zero-weight edges: " << num_zeros << std::endl;
  std::cout << "Max weight: " << max_w << std::endl;
  std::cout << "Min weight: " << min_w << std::endl;

#else
  // ====================== FLOAT WEIGHTS ======================
  using weight_t = decltype(std::declval<WNode>().w);
  const size_t num_buckets = 100000; // High resolution

  weight_t min_w = std::numeric_limits<weight_t>::infinity();
  weight_t max_w = -std::numeric_limits<weight_t>::infinity();

  // First pass: find min and max
#pragma omp parallel
  {
    weight_t local_min = min_w;
    weight_t local_max = max_w;
#pragma omp for nowait
    for (auto i = 0; i < g.num_nodes(); i++) {
      for (WNode wn : g.out_neigh(i)) {
        if (wn.w < local_min)
          local_min = wn.w;
        if (wn.w > local_max)
          local_max = wn.w;
      }
    }
#pragma omp critical
    {
      if (local_min < min_w)
        min_w = local_min;
      if (local_max > max_w)
        max_w = local_max;
    }
  }

  std::cout << "Max weight: " << max_w << std::endl;
  std::cout << "Min weight: " << min_w << std::endl;

  weight_t range = max_w - min_w;
  if (range == 0)
    range = 1; // Avoid div by zero
  weight_t bucket_width = range / num_buckets;

  std::vector<std::atomic<int64_t>> weight_hist(num_buckets);
  for (auto& x : weight_hist)
    x.store(0);

  std::cout << "starting counting (float weights)" << std::endl;

#pragma omp parallel for
  for (auto i = 0; i < g.num_nodes(); i++) {
    for (WNode wn : g.out_neigh(i)) {
      size_t idx = std::min((size_t)((wn.w - min_w) / bucket_width), num_buckets - 1);
      weight_hist[idx].fetch_add(1);
    }
  }
#endif

  // ====================== OUTPUT ======================
  std::string graph_name = std::filesystem::path(cli.filename()).filename().stem().string();
  std::filesystem::path out_dir = std::filesystem::path(cli.out_directory());

  std::ofstream weights_file;
  weights_file.open(out_dir / (graph_name + "_weight.csv"s));
  weights_file << "weight," << "frequency\n";

#ifndef USE_FLOAT
  for (auto i = 0; i < weight_hist.size(); i++) {
    if (weight_hist[i] > 0)
      weights_file << i << "," << weight_hist[i] << "\n";
  }
#else
  for (size_t i = 0; i < weight_hist.size(); i++) {
    if (weight_hist[i] > 0) {
      weight_t bucket_center = min_w + (i + 0.5) * bucket_width;
      weights_file << bucket_center << "," << weight_hist[i] << "\n";
    }
  }
#endif

  return 0;
}
