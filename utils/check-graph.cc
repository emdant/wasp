// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <cstdint>
#include <filesystem>
#include <iostream>

#include "benchmark.h"
#include "command_line.h"

using namespace std;

int main(int argc, char* argv[]) {
  int max_threads = omp_get_max_threads();
  cout << "OMP max threads: " << max_threads << endl;

  CLBase cli_gap(argc, argv, "read-gap");
  CLBase cli_mtx(argc, argv, "read-mtx");

  cli_gap.ParseArgs();
  auto graph_name = cli_gap.filename();

  filesystem::path mtx_folder("/path/to/folders");
  filesystem::path bin_folder("/path/to/folders");

  cout << graph_name << endl;

  auto bin_g = graph_name;
  auto mtx_g = graph_name;
  bin_g += ".wsg";
  mtx_g += ".mtx";

  auto gap_binary_path = bin_folder / graph_name / "gap" / bin_g;
  auto mtx_path = mtx_folder / graph_name / mtx_g;

  cli_gap.set_filename(gap_binary_path);
  cli_mtx.set_filename(mtx_path);
  cli_mtx.ParseArgs();

  cout << cli_gap.filename() << " " << cli_mtx.filename() << endl;

  {
    WeightedBuilder b2(cli_mtx);
    WGraph mtx = b2.MakeGraph();

    WeightedBuilder b1(cli_gap);
    WGraph gap = b1.MakeGraph();

    gap.PrintStats();
    mtx.PrintStats();

    if (gap.directed() != mtx.directed()) {
      cout << "not same edge type" << endl;
      cout << gap.directed() << " " << mtx.directed() << endl;
      return -1;
    }

    if (gap.num_nodes() != mtx.num_nodes()) {
      cout << "not same num nodes" << endl;
      cout << gap.num_nodes() << " " << mtx.num_nodes() << endl;
      return -1;
    }

    if (gap.num_edges() != mtx.num_edges()) {
      cout << "not same num edges" << endl;
      cout << gap.num_edges() << " " << mtx.num_edges() << endl;
      return -1;
    }

    for (auto i = 0; i < gap.num_nodes(); i++) {
      auto odeg_gap = gap.out_degree(i);
      auto odeg_mtx = mtx.out_degree(i);
      if (odeg_gap != odeg_mtx) {
        cout << "not same out-deg for vertex " << i << endl;
        return -1;
      }
      auto index_gap = gap.out_index()[i];
      auto index_mtx = mtx.out_index()[i];
      for (int64_t j = 0; j < odeg_gap; j++) {
        auto nw_gap = index_gap[j];
        auto nw_mtx = index_mtx[j];
        if (nw_gap.v != nw_mtx.v) {
          cout << "not same edge destination for index " << j << endl;
          return -1;
        }
        if (nw_gap.w != nw_mtx.w) {
          cout << "not same edge weight for index " << j << endl;
          return -1;
        }
      }
    }

    if (gap.directed()) {
      for (auto i = 0; i < gap.num_nodes(); i++) {
        auto odeg_gap = gap.in_degree(i);
        auto odeg_mtx = mtx.in_degree(i);
        if (odeg_gap != odeg_mtx) {
          cout << "not same in-deg for vertex " << i << endl;
          return -1;
        }
        auto index_gap = gap.in_index()[i];
        auto index_mtx = mtx.in_index()[i];
        for (int64_t j = 0; j < odeg_gap; j++) {
          auto nw_gap = index_gap[j];
          auto nw_mtx = index_mtx[j];
          if (nw_gap.v != nw_mtx.v) {
            cout << "not same edge destination for index " << j << endl;
            return -1;
          }
          if (nw_gap.w != nw_mtx.w) {
            cout << "not same edge weight for index " << j << endl;
            return -1;
          }
        }
      }
    }
  }

  cout << "graphs are equivalent :-)" << endl;
  return 0;
}
