// Copyright (c) 2024, Queen's University Belfast
// See LICENSE.txt for license details

#ifndef BUCKETING_BUCKET_H
#define BUCKETING_BUCKET_H

#include "base.h"

namespace bucketing {

template <typename ChunkT>
class bucket {
public:
  bucket() {}
  ~bucket() {} // we do not delete the contents because we assume that the pointers have been freed somewhere else

  bucket(bucket&) = delete;
  bucket(bucket&& other)
      : head_(std::exchange(other.head_, nullptr)),
        n_(other.n_) {}

  void push_value(NodeID value, priority_level p) {
    if (head_ == nullptr) {
      head_ = new ChunkT(value, p);
      n_++;
      return;
    }

    if (!head_->full() && head_->end == 0) {
      head_->push_back(value);
      return;
    }

    // Chunk is full or is a node chunk
    auto chunk = new ChunkT(value, p, head_);
    head_ = chunk;
    n_++;
    return;
  }

  void push_chunk(ChunkT* chunk) {
    if (head_ == nullptr || head_->full()) {
      chunk->next = std::exchange(head_, chunk);
      n_++;
      return;
    }

    chunk->next = std::exchange(head_->next, chunk);
    n_++;
  }

  node_prio pop_value() {
    node_prio temp = {head_->pop_front(), head_->priority};
    if (head_->empty()) {
      delete std::exchange(head_, head_->next);
      n_--;
    }
    return temp;
  }

  // pre-condition: !(*this).empty()
  ChunkT* pop_chunk() {
    auto temp = std::exchange(head_, head_->next);
    n_--;
    return temp;
  }

  bool empty() const {
    return head_ == nullptr;
  }

  std::size_t size() const {
    return n_;
  }

  void clear() noexcept {
    head_ = nullptr;
    n_ = 0;
  }

private:
  ChunkT* head_ = nullptr;
  std::size_t n_ = 0;
};

} // namespace bucketing

#endif