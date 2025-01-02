// Copyright (c) 2024, Queen's University Belfast
// See LICENSE.txt for license details

#ifndef BUCKETING_EXECUTOR_H_
#define BUCKETING_EXECUTOR_H_

#include <iostream>
#include <vector>

#include "omp.h"

#include "../parallel/atomics_array.h"
#include "../parallel/padded_array.h"
#include "../timer.h"
#include "benchmark.h"
#include "frontier.h"

#define PROCESS_NEIGHBOR(u, bucket)                           \
  if (cond_operation((u), (bucket))) {                        \
    for (WNode wn : g_.out_neigh((u))) {                      \
      if (auto new_prio = push_edge((u), wn)) {               \
        my_frontier.push(wn.v, coarsen_operation(*new_prio)); \
      }                                                       \
    }                                                         \
  }

namespace bucketing {

template <typename GraphT>
class executor {
public:
  executor(const GraphT& g)
      : num_threads_(omp_get_max_threads()),
        g_(g),
        frontiers_(num_threads_) {}

  template <typename... Args>
  inline void operator()(Args&&... args) {
    if (g_.directed())
      run_directed(std::forward<Args>(args)...);
    else
      run_undirected(std::forward<Args>(args)...);
  }

private:
  template <typename... Args>
  inline void run_directed(Args&&... args) {
    run<false, Args...>(std::forward<Args>(args)...);
  }

  template <typename... Args>
  inline void run_undirected(Args&&... args) {
    run<true, Args...>(std::forward<Args>(args)...);
  }

  template <bool UNDIRECTED, typename SrcT, typename PushOpT, typename CoarsenOpT, typename CondOpT = std::function<bool()>, typename PullSafeOpT = std::function<void()>, typename PullUnsafeOpT = std::function<void()>>
  inline void run(
      NodeID source,
      PushOpT push_edge,
      CoarsenOpT coarsen_operation,
      CondOpT cond_operation = [] { return true; }, PullSafeOpT pull_edge_safe = [] {}, PullUnsafeOpT pull_edge_unsafe = [] {}
  ) {
    frontiers_[starting_thread_].push(source, 0);

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
            PROCESS_NEIGHBOR(u, bucket);
          }

          std::vector<nodes_chunk*> stolen_chunks;
          stolen_chunks.reserve(num_threads_);

          auto next_bucket = my_frontier.next_priority_level();
          priority_level min = next_bucket;
          for (auto i = (tid + 1) % num_threads_; i != tid; i = (i + 1) % num_threads_) {
            if (frontiers_[i].current_priority_level() <= next_bucket) {
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
                PROCESS_NEIGHBOR(u, chunk->priority);
              }
              delete chunk;
            }
          }
        } while (!my_frontier.current_empty());

        auto next_bucket = my_frontier.next_priority_level();
        my_frontier.set_current(next_bucket);

        if (next_bucket != EMPTY_BUCKETS) {
          my_frontier.push_from(next_bucket);
        } else {
          bool all_finished = true;
          for (auto i = 0; i < num_threads_; i++) {
            if (frontiers_[i].current_priority_level() != EMPTY_BUCKETS) {
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

  static constexpr inline int starting_thread_ = 0;

  int num_threads_;
  const GraphT& g_;
  parallel::padded_array<frontier<nodes_chunk>> frontiers_;

  template <typename PullSafeOpT, typename... Args>
  constexpr inline void pull_safe(PullSafeOpT pull_edge_safe, Args&&... args) {
    if constexpr (!std::is_same_v<PullSafeOpT, std::function<void()>>) {
      pull_edge_safe(std::forward<Args>(args)...);
    }
  }

  template <typename PullUnsafeOpT, typename... Args>
  constexpr inline void pull_unsafe(PullUnsafeOpT pull_edge_unsafe, Args&&... args) {
    if constexpr (!std::is_same_v<PullUnsafeOpT, std::function<void()>>) {
      pull_edge_unsafe(std::forward<Args>(args)...);
    }
  }
};

} // namespace bucketing

#endif