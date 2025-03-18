// Copyright (c) 2024, Queen's University Belfast
// See LICENSE.txt for license details

#ifndef BUCKETING_EXECUTOR_H_
#define BUCKETING_EXECUTOR_H_

#include <vector>

#include "omp.h"
#include "papi.h"

#include "../parallel/padded_array.h"
#include "../timer.h"
#include "benchmark.h"
#include "frontier.h"

namespace bucketing {

class executor {
public:
  using frontier = bucketing::frontier<nodes_chunk>;

  executor() : num_threads_(omp_get_max_threads()), frontiers_(num_threads_) {}

  template <typename InitOpT, typename ProcessOpT>
  inline void run(InitOpT init_operation, ProcessOpT process_operation) {
#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      auto& my_frontier = frontiers_[tid];

#ifdef PAPI_PROFILE
      int event_codes[] = {PAPI_TOT_CYC, PAPI_TOT_INS, PAPI_L1_DCM, PAPI_L2_DCM};
      int event_set = PAPI_NULL;
      int retval;

      long long values[4];

      if (retval = PAPI_register_thread() != PAPI_OK) {
        std::cerr << "PAPI_register_thread error: " << retval << std::endl;
        exit(1);
      }

      if ((retval = PAPI_create_eventset(&event_set)) != PAPI_OK) {
        std::cerr << "PAPI_create_eventset error: " << retval << std::endl;
        exit(1);
      }

      if ((retval = PAPI_add_events(event_set, event_codes, 4)) != PAPI_OK) {
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
            auto [u, bucket] = node_pair.value();
            process_operation(my_frontier, u, bucket);
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
                process_operation(my_frontier, u, chunk->priority);
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

#ifdef PAPI_PROFILE

      if ((retval = PAPI_stop(event_set, values)) != PAPI_OK) {
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

#pragma omp critical
      {
        std::cout << "Thread " << tid << " PAPI results:" << std::endl;
        std::cout << "Total cycles: " << values[0] << std::endl;
        std::cout << "Total instructions: " << values[1] << std::endl;
        std::cout << "L1 data cache misses: " << values[2] << std::endl;
        std::cout << "L2 data cache misses: " << values[3] << std::endl;
      }

#endif

    } // end parallel region
  }

private:
  static constexpr inline int starting_thread_ = 0;

  int num_threads_;
  parallel::padded_array<frontier> frontiers_;
};

} // namespace bucketing

#endif