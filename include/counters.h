#ifndef COUNTERS_H_
#define COUNTERS_H_

#include "timer.h"

class counters {
public:
  struct thread_counter {
    CumulativeTimer push_timer;
    CumulativeTimer pop_timer;
    CumulativeTimer steal_timer;
  };

  static void init(int num_threads) {
    ctrs = new thread_counter[num_threads];
  }

  static thread_counter& get(int tid) { return ctrs[tid]; }

private:
  static thread_counter* ctrs;
};

#endif