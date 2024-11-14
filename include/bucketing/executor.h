// Copyright (c) 2024, Queen's University Belfast
// See LICENSE.txt for license details

#ifndef BUCKETING_EXECUTOR_H_
#define BUCKETING_EXECUTOR_H_

#include <iostream>
#include <vector>

#include "omp.h"

#include "../parallel/atomics_array.h"
#include "../parallel/pparray.h"
#include "../timer.h"
#include "benchmark.h"
#include "frontier.h"

#define process_node(node, bucket, leaf, frontier)   \
  for (WNode wn : g_.out_neigh((node)))              \
    if (auto new_prio = edge_operation_((node), wn)) \
      if (!(leaf)[wn.v])                             \
        (frontier).push(wn.v, *new_prio / coarsening_);

namespace bucketing {

template <typename Graph, typename CondOpT, typename EdgeOpT>
class executor {
public:
  executor(const Graph& g, std::int32_t coarsening, CondOpT& cond_operation, EdgeOpT& edge_operation)
      : num_threads_(omp_get_max_threads()),
        g_(g),
        coarsening_(coarsening),
        cond_operation_(cond_operation),
        edge_operation_(edge_operation),
        frontiers_(num_threads_),
        is_leaf_(g_.num_nodes()) {}

  void run(NodeID source) {
    frontiers_[starting_thread_].push(source, 0);

#pragma omp parallel for schedule(static)
    for (NodeID node = 0; node < g_.num_nodes(); node++) {
      if (g_.in_degree(node) == 1 && g_.out_degree(node) == 0)
        is_leaf_[node] = true;
      else if ((g_.in_degree(node) == 1 && g_.out_degree(node) == 1)) {
        NodeID in_src = (NodeID)g_.in_index()[node][0];
        NodeID out_dst = (NodeID)g_.out_index()[node][0];
        if (in_src == out_dst)
          is_leaf_[node] = true;
      }
    }

#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      auto& my_frontier = frontiers_[tid];

      while (true) {

        // Using do-while so non-starting threads will try to steal first
        do {
          // Process current bucket
          for (auto node_pair = my_frontier.pop(); node_pair; node_pair = my_frontier.pop()) {
            auto [u, bucket] = node_pair.value();
            if (cond_operation_(u, bucket))
              process_node(u, bucket, is_leaf_, my_frontier);
          }

          std::vector<nodes_chunk*> stolen_chunks;
          stolen_chunks.reserve(num_threads_);

          auto next_bucket = my_frontier.next_index();
          bucket_index min = next_bucket;
          for (auto i = (tid + 1) % num_threads_; i != tid; i = (i + 1) % num_threads_) {
            if (frontiers_[i].current_index() <= next_bucket) {
              auto chunk = frontiers_[i].steal();
              if (chunk != nullptr) {
                min = std::min(min, chunk->priority);
                stolen_chunks.push_back(chunk);
              }
            }
          }

          // Processing stolen chunks
          if (!stolen_chunks.empty()) {
            my_frontier.set_current(min);

            for (auto i = 0; i < stolen_chunks.size(); i++) {
              auto& chunk = stolen_chunks[i];
              while (!chunk->empty()) {
                auto u = chunk->pop_front();
                if (cond_operation_(u, chunk->priority))
                  process_node(u, chunk->priority, is_leaf_, my_frontier);
              }
              delete chunk;
            }
          }

        } while (!my_frontier.current_empty());

        auto next_bucket = my_frontier.next_index();
        my_frontier.set_current(next_bucket);

        if (next_bucket != EMPTY_BUCKETS) {
          my_frontier.push_from(next_bucket);
        } else {
          bool all_finished = true;
          for (auto i = 0; i < num_threads_; i++) {
            if (frontiers_[i].current_index() != EMPTY_BUCKETS) {
              all_finished = false;
              break;
            }
          }

          if (all_finished)
            break;
        }
      }
    }
  }

  void print_content() {
    for (auto i = 0; i < num_threads_; i++) {
      auto& buckets = frontiers_[i];
      if (!buckets.current_empty() || buckets.next_index() != bucketing::EMPTY_BUCKETS)
        std::cout << "thread i:" << std::endl;

      if (!buckets.current_empty())
        std::cout << "\tcurrent bucket not empty" << std::endl;

      if (buckets.next_index() != bucketing::EMPTY_BUCKETS)
        std::cout << "\tnext bucket not empty" << std::endl;
    }
  }

private:
  static constexpr inline int starting_thread_ = 0;

  int num_threads_;
  const Graph& g_;
  int coarsening_;
  CondOpT& cond_operation_;
  EdgeOpT& edge_operation_;
  parallel::padded_array<frontier<nodes_chunk>> frontiers_;
  std::vector<bool> is_leaf_;
};

} // namespace bucketing

#endif