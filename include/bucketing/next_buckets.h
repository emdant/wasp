// Copyright (c) 2024, Queen's University Belfast
// See LICENSE.txt for license details

#ifndef BUCKETING_NEXT_BUCKETS_H_
#define BUCKETING_NEXT_BUCKETS_H_

#include <vector>

#include "base.h"
#include "bucket.h"

namespace bucketing {

template <typename ChunkT>
class next_buckets {
public:
  next_buckets() {
    buckets_.reserve(1024);
    buckets_.resize(128);
  }

  bucket<ChunkT>& get(priority_level p) {
    return buckets_[p];
  }

  void push(NodeID node, priority_level p) {
    if (p >= buckets_.size()) {
      auto new_size = buckets_.size();
      new_size = std::max(static_cast<priority_level>(new_size) * 2, p + 1);
      buckets_.resize(new_size);
    }

    buckets_[p].push_value(node, p);
  }

  void push(nodes_chunk* chunk, priority_level p) {
    if (p >= buckets_.size()) {
      auto new_size = buckets_.size();
      new_size = std::max(static_cast<priority_level>(new_size) * 2, p + 1);
      buckets_.resize(new_size);
    }

    buckets_[p].push_chunk(chunk);
  }

  priority_level first_nonempty() const {
    for (std::size_t i = 0; i < buckets_.size(); i++) {
      if (!buckets_[i].empty()) {
        return i;
      }
    }
    return EMPTY_BUCKETS;
  }

private:
  std::vector<bucket<ChunkT>> buckets_;
};

} // namespace bucketing

#endif