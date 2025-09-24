#include "benchmark.h"
#include "command_line.h"
#include "util.h"

using namespace std;

int main(int argc, char* argv[]) {
  CLEdge cli(argc, argv, "weights");
  cli.parse();

  auto suffix = GetSuffix(cli.graph_filename());
  if (suffix != ".wsg") {
    std::cout << "Graph format must be: .wsg" << std::endl;
    std::exit(-1);
  }

  WeightedBuilder b(cli);
  WGraph g = b.MakeGraph();
  g.PrintStats();

  for (auto wn : g.out_neigh(cli.source())) {
    if (wn.v == cli.destination()) {
      std::cout << "\nWeight of edge (" << cli.source() << ", " << cli.destination() << ") is: " << wn.w << std::endl;
    }
  }
}