// Copyright (c) 2024, Queen's University Belfast
// See LICENSE.txt for license details

#ifndef BUCKETING_FRONTIER_H_
#define BUCKETING_FRONTIER_H_

#include <atomic>
#include <limits>
#include <memory>
#include <vector>

#include "base.h"
#include "current_bucket.h"
#include "deque_bucket.h"

namespace bucketing {

template <typename ChunkT = nodes_chunk>
class frontier {
private:
  using value_type = typename ChunkT::value_type;
  using priority_type = typename ChunkT::priority_type;

public:
  frontier() {
    buckets_.reserve(8192);
    buckets_.resize(8192);

    stealing_deque_.store(buckets_[0].deque());
  }

  ~frontier() {
    if (omp_get_thread_num() == 0) {
      //       std::cout << "push time " << pusht.Seconds() << std::endl;
      //       std::cout << "pop time " << popt.Seconds() << std::endl;
    }
  }

  inline void push(value_type value, priority_type priority) {
    //     pusht.Start();
    if (priority >= buckets_.size())
      resize_buckets(priority);

    buckets_[priority].push(value, priority);
    //     pusht.Stop();
  }

  inline void push(ChunkT* chunk) {
    //     pusht.Start();
    if (chunk->priority >= buckets_.size())
      resize_buckets(chunk->priority);

    buckets_[chunk->priority].push(chunk);
    //     pusht.Stop();
  }

  inline std::optional<node_prio> pop() {
    //     popt.Start();
    auto curr = current_.load(std::memory_order_relaxed);

    std::optional<node_prio> tmp = std::nullopt;

    if (curr == EMPTY_BUCKETS) {
      //       popt.Stop();
      return std::nullopt;
    }

    tmp = buckets_[curr].pop();
    //     popt.Stop();
    return tmp;
  }

  inline ChunkT* steal() {
    auto sq = stealing_deque_.load();
    if (sq == nullptr)
      return nullptr;

    return sq->steal();
  }

  inline bucket_index current_index() const {
    return current_.load();
  }

  bool current_empty() {
    return buckets_[current_.load(std::memory_order_relaxed)].empty();
  }

  inline void advance() {
    auto next = next_index();

    if (next == EMPTY_BUCKETS)
      stealing_deque_.store(nullptr);
    else
      stealing_deque_.store(buckets_[next].deque());

    current_.store(next, std::memory_order_release);
  }

  bucket_index next_index(bucket_index start_from = 0) {
    for (auto i = start_from; i < buckets_.size(); i++) {
      if (!buckets_[i].empty()) {
        return i;
      }
    }
    return EMPTY_BUCKETS;
  }

private:
  std::atomic<bucket_index> current_{0};
  std::vector<bucket<ChunkT>> buckets_;
  std::atomic<parallel::deque<ChunkT*>*> stealing_deque_;
  //   CumulativeTimer pusht, popt;

  inline void resize_buckets(priority_type at_least) {
    auto size = buckets_.size();
    while ((size <<= 1) <= at_least)
      ;
    buckets_.resize(size);
  }
};

}; // namespace bucketing

#endif