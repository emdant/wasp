// Copyright (c) 2024, Queen's University Belfast
// See LICENSE.txt for license details

#ifndef BUCKETING_BASE_H
#define BUCKETING_BASE_H

#include <cstdint>
#include <limits>

#include "../benchmark.h"
#include "../containers/chunk.h"

namespace bucketing {

using bucket_index = std::int64_t;
constexpr inline bucket_index EMPTY_BUCKETS = std::numeric_limits<std::int64_t>::max() / 2;

struct node_prio {
  NodeID node;
  bucket_index bucket;
};

using nodes_chunk = containers::chunk<NodeID, bucket_index, 64>;

} // namespace bucketing

#endif