#ifndef BUCKETS_BITMAP_H
#define BUCKETS_BITMAP_H

#include <vector>

#include "../bucketing/bucket.h"

class buckets_bitmap {
public:
  explicit buckets_bitmap(std::size_t level1_size = 64)
      : level1_(level1_size), level2_(BITS_PER_BLOCK * level1_size) {}

  void set_bucket(bucketing::priority_level p) {
    if (p >= level2_.size() * BITS_PER_BLOCK) {
      resize_to_fit(p);
    }

    std::size_t l2_block_index = p / BITS_PER_BLOCK;
    std::size_t l2_bit_index = p % BITS_PER_BLOCK;

    uint64_t old = level2_[l2_block_index];
    level2_[l2_block_index] |= (1UL << l2_bit_index);

    if (old == 0) {
      std::size_t l1_block_index = l2_block_index / BITS_PER_BLOCK;
      std::size_t l1_bit_index = l2_block_index % BITS_PER_BLOCK;
      level1_[l1_block_index] |= (1UL << l1_bit_index);
    }
  }

  void reset_bucket(bucketing::priority_level p) {
    if (p >= level2_.size() * BITS_PER_BLOCK) {
      return;
    }

    std::size_t l2_block_index = p / BITS_PER_BLOCK;
    std::size_t l2_bit_index = p % BITS_PER_BLOCK;

    level2_[l2_block_index] &= ~(1UL << l2_bit_index);

    if (level2_[l2_block_index] == 0) {
      std::size_t l1_block_index = l2_block_index / BITS_PER_BLOCK;
      std::size_t l1_bit_index = l2_block_index % BITS_PER_BLOCK;
      level1_[l1_block_index] &= ~(1UL << l1_bit_index);
    }
  }

  bucketing::priority_level first_nonempty() {
    for (std::size_t i = 0; i < level1_.size(); i++) {
      if (level1_[i] == 0)
        continue;

      std::size_t l1_bit_index = __builtin_ctzl(level1_[i]);
      std::size_t l2_block_index = i * BITS_PER_BLOCK + l1_bit_index;
      std::size_t l2_bit_index = __builtin_ctzl(level2_[l2_block_index]);

      return l2_block_index * BITS_PER_BLOCK + l2_bit_index;
    }
    return bucketing::EMPTY_BUCKETS;
  }

private:
  static constexpr std::size_t BITS_PER_BLOCK = sizeof(uint64_t) * 8;
  std::vector<uint64_t> level1_;
  std::vector<uint64_t> level2_;

  void resize_to_fit(bucketing::priority_level p) {
    std::size_t new_l1_size = ((p / BITS_PER_BLOCK) / BITS_PER_BLOCK) + 1;

    level1_.resize(new_l1_size);
    level2_.resize(BITS_PER_BLOCK * new_l1_size);
  }
};

#endif