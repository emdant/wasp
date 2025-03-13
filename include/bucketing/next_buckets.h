// Copyright (c) 2024, Queen's University Belfast
// See LICENSE.txt for license details

#ifndef BUCKETING_NEXT_BUCKETS_H_
#define BUCKETING_NEXT_BUCKETS_H_

#include <vector>

#include "../containers/buckets_bitmap.h"
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

    bool was_empty = buckets_[p].empty();
    buckets_[p].push_value(node, p);

    if (was_empty) {
      bitmap_.set_bucket(p);
    }
  }

  priority_level first_nonempty() {
    return bitmap_.first_nonempty();
  }

  void mark_empty(priority_level p) {
    bitmap_.reset_bucket(p);
  }

private:
  std::vector<bucket<ChunkT>> buckets_;
  buckets_bitmap bitmap_;
};

} // namespace bucketing

#endif