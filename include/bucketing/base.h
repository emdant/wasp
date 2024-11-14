// Copyright (c) 2024, Queen's University Belfast
// See LICENSE.txt for license details

#ifndef BUCKETING_BASE_H
#define BUCKETING_BASE_H

#include <cstdint>
#include <limits>

#include "../benchmark.h"
#include "../containers/chunk.h"

namespace bucketing {

using priority_level = std::int64_t;
constexpr inline priority_level EMPTY_BUCKETS = std::numeric_limits<std::int64_t>::max() / 2;

struct node_prio {
  NodeID node;
  priority_level bucket;
};

using nodes_chunk = containers::chunk<NodeID, priority_level, 64>;

} // namespace bucketing

#endif