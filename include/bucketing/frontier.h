// Copyright (c) 2024, Queen's University Belfast
// See LICENSE.txt for license details

#ifndef BUCKETING_FRONTIER_H_
#define BUCKETING_FRONTIER_H_

#include <atomic>

#include "base.h"
#include "current_bucket.h"
#include "next_buckets.h"

namespace bucketing {

template <typename ChunkT = nodes_chunk>
class frontier {
public:
  inline void push(NodeID node, priority_level priority) {
    if (current_.load(std::memory_order_relaxed) == priority) {
      current_bucket_.push(node, priority);
      return;
    }

    next_buckets_.push(node, priority);
  }

  inline void push(nodes_chunk* chunk) {
    if (current_.load(std::memory_order_relaxed) == chunk->priority) {
      current_bucket_.push(chunk);
      return;
    }

    next_buckets_.push(chunk, chunk->priority);
  }

  inline std::optional<node_prio> pop() {
    return current_bucket_.pop();
  }

  inline ChunkT* steal() {
    return current_bucket_.steal();
  }

  inline priority_level current_priority_level() const {
    return current_.load(std::memory_order_acquire);
  }

  inline priority_level next_priority_level() const {
    return next_buckets_.first_nonempty();
  }

  bool current_empty() {
    return current_bucket_.empty();
  }

  void push_from(priority_level p) {
    auto& bucket = next_buckets_.get(p);

    while (!bucket.empty())
      current_bucket_.push(bucket.pop_chunk());
  }

  inline void set_current(priority_level p) {
    current_.store(p, std::memory_order_release);
  }

private:
  std::atomic<priority_level> current_{0};
  current_bucket<ChunkT> current_bucket_;
  next_buckets<ChunkT> next_buckets_;
};

using chunks_frontier = bucketing::frontier<nodes_chunk>;

}; // namespace bucketing

#endif