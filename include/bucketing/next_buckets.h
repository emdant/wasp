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

  bucket<ChunkT>& get(bucket_index index) {
    return buckets_[index];
  }

  void push(NodeID node, bucket_index index) {
    auto current_size = buckets_.size();
    if (index >= current_size) {
      while ((current_size <<= 1) <= index)
        ;
      buckets_.resize(current_size);
    }

    buckets_[index].push_value(node, index);
  }

  bucket_index first_nonempty() {
    for (auto i = 0; i < buckets_.size(); i++) {
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