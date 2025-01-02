// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <iostream>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "writer.h"

using namespace std;

int main(int argc, char* argv[]) {
  CLApp cli(argc, argv, "read");
  cli.ParseArgs();
  WeightedBuilder bw(cli);
  WGraph g = bw.MakeGraph();
  g.PrintStats();

  auto n = g.num_nodes();
  for (size_t u = 0; u < n; u++) {
    if (g.out_degree(u) > 0) {
      auto min = g.out_degree(u) < 5 ? g.out_degree(u) : 5;

      auto edges = g.out_index()[u];
      for (size_t i = 0; i < min; i++) {
        WNode wn = edges[i];
        std::cout << i << "-th neighbor: " << wn.v
                  << ", weight: " << wn.w << std::endl;
      }
      break;
    }
  }

  return 0;
}
