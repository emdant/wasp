// Copyright (c) 2024, Queen's University Belfast
// See LICENSE.txt for license details

#ifndef BUCKETING_DEQUE_BUCKET_H
#define BUCKETING_DEQUE_BUCKET_H

#include <optional>

#include "../parallel/deque.h"
#include "base.h"

namespace bucketing {

template <typename ChunkT>
class bucket {
public:
  bucket()
      : deque_(new parallel::deque<ChunkT*>) {}

  bucket(bucket&& rhs)
      : deque_(std::exchange(rhs.deque_, nullptr)),
        local_chunk_(std::exchange(rhs.local_chunk_, nullptr)) {}

  bucket& operator=(bucket&& rhs) {
    deque_ = std::exchange(rhs.deque_, nullptr);
    local_chunk_ = std::exchange(rhs.local_chunk_, nullptr);
  }

  bool empty() const noexcept {
    if (local_chunk_ != nullptr)
      return false;

    return deque_->empty();
  }

  void push(NodeID node, bucket_index bucket) {
    if (local_chunk_ == nullptr) {
      local_chunk_ = new ChunkT(node, bucket);
      return;
    }
    local_chunk_->push_back(node);
    if (!local_chunk_->full())
      return;

    deque_->push(std::exchange(local_chunk_, nullptr));
    return;
  }

  void push(ChunkT* chunk) {
    if (local_chunk_ == nullptr)
      local_chunk_ = chunk;
    else
      deque_->push(chunk);
  }

  std::optional<node_prio> pop() {
    if (local_chunk_ != nullptr) {
      node_prio ret{local_chunk_->pop_front(), local_chunk_->priority};

      if (local_chunk_->empty()) {
        delete std::exchange(local_chunk_, nullptr);
      }

      return std::move(ret);
    }

    local_chunk_ = deque_->pop();
    if (local_chunk_ != nullptr) {
      node_prio ret{local_chunk_->pop_front(), local_chunk_->priority};

      if (local_chunk_->empty()) {
        delete std::exchange(local_chunk_, nullptr);
      }
      return std::move(ret);
    }

    // deque was empty
    return std::nullopt;
  }

  parallel::deque<ChunkT*>* deque() {
    return deque_;
  }

private:
  parallel::deque<ChunkT*>* deque_;
  ChunkT* local_chunk_ = nullptr;
};

} // namespace bucketing

#endif