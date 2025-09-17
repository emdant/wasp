// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <iostream>
#include <limits>
#include <queue>
#include <vector>

#include "benchmark.h"
#include "command_line.h"

using namespace std;

const WeightT kDistInf = numeric_limits<WeightT>::max() / 2;
const size_t kMaxBin = numeric_limits<size_t>::max() / 2;
const size_t kBinSizeThreshold = 1000;

void PrintSSSPStats(const WGraph& g, const vector<WeightT>& dist) {
  WeightT max_dist = 0;
  int64_t num_reached = 0;

#pragma omp parallel for reduction(+ : num_reached) reduction(max : max_dist)
  for (size_t i = 0; i < dist.size(); i++) {
    if (dist[i] != kDistInf && dist[i] > max_dist)
      max_dist = dist[i];
    if (dist[i] != kDistInf)
      num_reached++;
  }

  cout << "SSSP Tree reaches " << num_reached << " nodes" << endl;
  cout << "Max dist " << max_dist << endl;
}

vector<WeightT> Dijkstra(const WGraph& g, NodeID source) {
  vector<WeightT> dist(g.num_nodes(), kDistInf);
  vector<bool> settled(g.num_nodes());
  dist[source] = 0;
  std::size_t num_visits = 0;

  typedef pair<WeightT, NodeID> WN;
  priority_queue<WN, vector<WN>, greater<WN>> mq;
  mq.push(make_pair(0, source));

  while (!mq.empty()) {
    WeightT tentative_dist = mq.top().first;
    NodeID u = mq.top().second;
    mq.pop();
    settled[u] = true;

    for (WNode wn : g.out_neigh(u)) {
      if (!settled[wn.v]) {
        num_visits++;
        if (tentative_dist + wn.w < dist[wn.v]) {
          dist[wn.v] = tentative_dist + wn.w;
          mq.push(make_pair(tentative_dist + wn.w, wn.v));
        }
      }
    }
  }
  cout << "Number of relaxations: " << num_visits << endl;
  return dist;
}

int main(int argc, char* argv[]) {
  CLTraversal cli(argc, argv, "single-source shortest-path");
  cli.parse();

  WeightedBuilder b(cli);
  WGraph g = b.MakeGraph();
  g.PrintStats();

  SourcePicker sp(g, cli.sources_filename(), cli.start_vertex());
  std::vector<NodeID> sources;

  for (auto i = 0; i < cli.num_sources(); i++) {
    auto source = sp.PickNext();
    std::cout << "Source: " << source << std::endl;

    auto SSSPBound = [&sp, &cli, source](const WGraph& g) {
      return Dijkstra(g, source);
    };

    BenchmarkKernel(cli, g, SSSPBound, PrintSSSPStats, VerifyUnimplemented);
  }

  return 0;
}