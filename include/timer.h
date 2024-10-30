// Copyright (c) 2024, Queen's University Belfast
// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef TIMER_H_
#define TIMER_H_

#include <chrono>

#include "omp.h"

/*
GAP Benchmark Suite
Class:  Timer
Authors: Scott Beamer, Michael Sutton

Simple timer that wraps std::chrono
*/

class Timer {
public:
  Timer() {}

  void Start() {
    elapsed_time_ = start_time_ = std::chrono::high_resolution_clock::now();
  }

  void Stop() {
    elapsed_time_ = std::chrono::high_resolution_clock::now();
  }

  double Seconds() const {
    return std::chrono::duration_cast<std::chrono::duration<double>>(elapsed_time_ - start_time_).count();
  }

  double Millisecs() const {
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(elapsed_time_ - start_time_).count();
  }

  double Microsecs() const {
    return std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(elapsed_time_ - start_time_).count();
  }

private:
  std::chrono::high_resolution_clock::time_point start_time_, elapsed_time_;
};

class CumulativeTimer {
public:
  CumulativeTimer() {}

  void Start() {
    start_time_ = std::chrono::high_resolution_clock::now();
  }

  void Stop() {
    total_ += std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start_time_);
  }

  double Seconds() const {
    return total_.count();
  }

  double Millisecs() const {
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(total_).count();
  }

  double Microsecs() const {
    return std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(total_).count();
  }

private:
  std::chrono::high_resolution_clock::time_point start_time_;
  std::chrono::duration<double> total_{};
};

class OMPTimer {
public:
  OMPTimer() {}

  void Start() {
    elapsed_time_ = start_time_ = omp_get_wtime();
  }

  void Stop() {
    elapsed_time_ = omp_get_wtime();
  }

  double Seconds() const {
    return elapsed_time_ - start_time_;
  }

private:
  double start_time_, elapsed_time_;
};

// Times op's execution using the timer t
#define TIME_OP(t, op) \
  {                    \
    t.Start();         \
    (op);              \
    t.Stop();          \
  }

#endif // TIMER_H_
