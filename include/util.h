// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef UTIL_H_
#define UTIL_H_

#include <cinttypes>
#include <memory>
#include <stdio.h>
#include <string>

/*
GAP Benchmark Suite
Author: Scott Beamer

Miscellaneous helpers that don't fit into classes
*/

#define _TEST_HI(v) (((v) >> 31) & 1) != 0
#define _TEST_LO(v) ((v) & 1) != 0

#define _SET_HI(v) ((v) | (1 << 31))
#define _SET_LO(v) ((v) | 1)

#define IS_POW2(x)

inline bool is_pow2(std::uint32_t x) {
  return ((x & (x - 1)) == 0);
}

inline std::uint32_t next_pow2(std::uint32_t x) {
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x++;
  return x;
}

template <typename T>
bool test_low_and_clear(T& value) {
  bool temp = (value & 1) != 0;
  value &= ~1;
  return temp;
}

static const int64_t kRandSeed = 27491095;

inline void PrintLabel(const std::string& label, const std::string& val) {
  printf("%-21s%7s\n", (label + ":").c_str(), val.c_str());
  fflush(stdout);
}

inline void PrintTime(const std::string& s, double seconds) {
  printf("%-21s%3.5lf\n", (s + ":").c_str(), seconds);
  fflush(stdout);
}

inline void PrintStep(const std::string& s, int64_t count) {
  printf("%-14s%14" PRId64 "\n", (s + ":").c_str(), count);
  fflush(stdout);
}

inline void PrintStep(const std::string& s, double seconds, int64_t count = -1) {
  if (count != -1)
    printf("%5s%11" PRId64 "  %10.5lf\n", s.c_str(), count, seconds);
  else
    printf("%5s%23.5lf\n", s.c_str(), seconds);
  fflush(stdout);
}

inline void PrintStep(int step, double seconds, int64_t count = -1) {
  PrintStep(std::to_string(step), seconds, count);
  fflush(stdout);
}

// Runs op and prints the time it took to execute labelled by label
#define TIME_PRINT(label, op)       \
  {                                 \
    Timer t_;                       \
    t_.Start();                     \
    (op);                           \
    t_.Stop();                      \
    PrintTime(label, t_.Seconds()); \
  }

template <typename T_>
class RangeIter {
  T_ x_;

public:
  explicit RangeIter(T_ x) : x_(x) {}
  bool operator!=(RangeIter const& other) const { return x_ != other.x_; }
  T_ const& operator*() const { return x_; }
  RangeIter& operator++() {
    ++x_;
    return *this;
  }
};

template <typename T_>
class Range {
  T_ from_;
  T_ to_;

public:
  explicit Range(T_ to) : from_(0), to_(to) {}
  Range(T_ from, T_ to) : from_(from), to_(to) {}
  RangeIter<T_> begin() const { return RangeIter<T_>(from_); }
  RangeIter<T_> end() const { return RangeIter<T_>(to_); }
};

template <class T>
std::unique_ptr<T> make_unique_for_overwrite(std::size_t n) {
  return std::unique_ptr<T>(new std::remove_extent_t<T>[n]);
}

inline std::string GetSuffix(const std::string filename) {
  std::size_t suff_pos = filename.rfind('.');
  if (suff_pos == std::string::npos) {
    printf("Couldn't find suffix of %s\n", filename.c_str());
    fflush(stdout);
    std::exit(-1);
  }
  return filename.substr(suff_pos);
}

#endif // UTIL_H_
