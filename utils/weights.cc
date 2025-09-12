#include "benchmark.h"
#include "command_line.h"
#include "util.h"
#include "writer.h"

using namespace std;

int main(int argc, char* argv[]) {
  CLExport cli(argc, argv, "weights");
  cli.parse();

  auto suffix = GetSuffix(cli.graph_filename());
  if (suffix != ".wsg") {
    std::cout << "Graph format must be: .wsg" << std::endl;
    std::exit(-1);
  }

  WeightedBuilder b(cli);
  WGraph g = b.MakeGraph();
  g.PrintStats();

  WeightsWriter ww(g);
  ww.WriteWeightsSerialized(cli.out_filename());
}