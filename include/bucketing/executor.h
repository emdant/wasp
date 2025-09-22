// Copyright (c) 2024, Queen's University Belfast
// See LICENSE.txt for license details

#ifndef BUCKETING_EXECUTOR_H_
#define BUCKETING_EXECUTOR_H_

#include <vector>

#include "omp.h"
#ifdef PAPI_PROFILE
#include "../profiling/papi_helper.h"
#include "papi.h"
#endif

#include "../parallel/padded_array.h"
#include "bucketing/base.h"
#include "frontier.h"
#include "numa/numa_distance_map.h"

namespace bucketing {

class executor {
public:
  executor()
      : num_threads_(omp_get_max_threads()),
        frontiers_(num_threads_) {}

  template <typename InitOpT, typename StaleOpT, typename InspectOpT, typename ProcessOpT>
  inline void run(InitOpT init_operation, StaleOpT stale, InspectOpT inspect_operation, ProcessOpT process_operation) {
#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      auto& my_frontier = frontiers_[tid];
      int num_dist = numa_distance_map::get_num_distances();
      std::vector<nodes_chunk*> stolen_chunks;
      stolen_chunks.reserve(num_threads_);

#ifdef PAPI_PROFILE
      int event_set = PAPI_NULL;
      int retval;

      if (retval = PAPI_register_thread() != PAPI_OK) {
        std::cerr << "PAPI_register_thread error: " << retval << std::endl;
        exit(1);
      }

      if ((retval = PAPI_create_eventset(&event_set)) != PAPI_OK) {
        std::cerr << "PAPI_create_eventset error: " << retval << std::endl;
        exit(1);
      }

      if ((retval = PAPI_add_events(event_set, papi_helper::get_events(), papi_helper::get_num_events())) != PAPI_OK) {
        std::cerr << "PAPI_add_events error: " << retval << std::endl;
        exit(1);
      }

      if ((retval = PAPI_start(event_set)) != PAPI_OK) {
        std::cerr << "PAPI_start error: " << retval << std::endl;
        exit(1);
      }
#endif

      if (tid == starting_thread_)
        init_operation(my_frontier);

      while (true) {

        // Using do-while so non-starting threads will try to steal first
        do {
          // Process current bucket
          for (auto node_pair = my_frontier.pop(); node_pair; node_pair = my_frontier.pop()) {
            auto [u, bucket, c_begin, c_end] = node_pair.value();
            if (!stale(bucket, u)) {
              if (c_begin == 0 && c_end == 0) {
                auto [begin, end] = inspect_operation(my_frontier, bucket, u);
                process_operation(my_frontier, bucket, u, begin, end);
              } else {
                process_operation(my_frontier, bucket, u, c_begin, c_end);
              }
            }
          }

          auto next_bucket = my_frontier.next_priority_level();
          priority_level min = next_bucket;

          for (auto d = 0; d < num_dist && stolen_chunks.empty(); d++) {
            auto victims = numa_distance_map::get_threads_at_distance(tid, d);
            for (const auto victim : victims) {
              if (frontiers_[victim].current_priority_level() <= next_bucket) {
                auto chunk = frontiers_[victim].steal();
                if (chunk != nullptr) {
                  min = std::min(min, chunk->priority);
                  stolen_chunks.push_back(chunk);
                }
              }
            }
          }

          if (!stolen_chunks.empty()) {
            my_frontier.set_current(min);

            for (std::size_t i = 0; i < stolen_chunks.size(); i++) {
              auto& chunk = stolen_chunks[i];
              if (chunk->begin == 0 && chunk->end == 0) {
                while (!chunk->empty()) {
                  auto u = chunk->pop_front();
                  if (!stale(chunk->priority, u)) {
                    auto [begin, end] = inspect_operation(my_frontier, chunk->priority, u);
                    process_operation(my_frontier, chunk->priority, u, begin, end);
                  }
                }
              } else {
                auto u = chunk->pop_front();
                if (!stale(chunk->priority, u))
                  process_operation(my_frontier, chunk->priority, u, chunk->begin, chunk->end);
              }
              delete chunk;
            }
            stolen_chunks.clear();
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

#ifdef PAPI_PROFILE

      if ((retval = PAPI_stop(event_set, papi_helper::get_thread_values(tid))) != PAPI_OK) {
        std::cerr << "PAPI_stop error: " << retval << std::endl;
        exit(1);
      }

      if ((retval = PAPI_cleanup_eventset(event_set)) != PAPI_OK) {
        std::cerr << "PAPI_cleanup_eventset error: " << retval << std::endl;
        exit(1);
      }

      if ((retval = PAPI_destroy_eventset(&event_set)) != PAPI_OK) {
        std::cerr << "PAPI_destroy_eventset error: " << retval << std::endl;
        exit(1);
      }
#endif

    } // end parallel region
  }

private:
  static constexpr inline int starting_thread_ = 0;

  int num_threads_;
  int num_numa_nodes_;
  parallel::padded_array<chunks_frontier> frontiers_;
};

} // namespace bucketing

#endif