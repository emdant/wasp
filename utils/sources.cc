
#include <algorithm>
#include <vector>

#include "benchmark.h"
#include "command_line.h"
#include "components.h"
#include "generator.h"
#include "util.h"
#include "writer.h"

using namespace std;

int main(int argc, char* argv[]) {
  CLExport cli(argc, argv, "sources");
  cli.parse();

  auto suffix = GetSuffix(cli.graph_filename());
  if (suffix != ".mtx" && suffix != ".wsg" && suffix != ".sg") {
    std::cout << "Graph format must be one of: .mtx .wsg .sg" << std::endl;
    std::exit(-1);
  }

  Builder b(cli);
  Graph g = b.MakeGraph();
  g.PrintStats();

  ComponentFinder cf(g, g.directed());
  auto comp = cf.info();
  cout << "Largest " << (g.directed() ? "strongly " : "") << "connected component has size: " << comp.max_size << endl;

  std::mt19937_64 rng(kRandSeed);
  UniDist dist(g.num_nodes() - 1, rng);

  const auto NUM_SOURCES = std::min(1024l, g.num_nodes());
  std::vector<NodeID> sources;

  for (size_t i = 0; i < NUM_SOURCES; i++) {
    NodeID source;
    do {
      source = dist();
    } while (comp.components[source] != comp.max_id ||
             std::find(sources.begin(), sources.end(), source) != sources.end()
    );
    sources.push_back(source);
  }

  SourcesWriter w(sources);
  w.WriteSources(cli.out_filename());
}