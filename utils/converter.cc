// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <iostream>

#include "benchmark.h"
#include "command_line.h"

using namespace std;

int main(int argc, char* argv[]) {
  CLConverter cli(argc, argv, "converter");
  cli.parse();

  if (cli.out_weighted()) {
    std::cout << "Output is weighted" << std::endl;
    WeightedBuilder bw(cli);
    bw.toggleRelabeling(cli.relabel_vertices());
    WGraph wg = bw.MakeGraph();
    wg.PrintStats();

    WeightedWriter ww(wg);
    ww.WriteGraph(cli.out_filename(), cli.out_format());
  } else {
    std::cout << "Output is not weighted" << std::endl;
    Builder b(cli);
    b.toggleRelabeling(cli.relabel_vertices());
    Graph g = b.MakeGraph();
    g.PrintStats();

    Writer w(g);
    w.WriteGraph(cli.out_filename(), cli.out_format());
  }
  return 0;
}
