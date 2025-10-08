// Copyright (c) 2024, Queen's University Belfast
// See LICENSE.txt for license details

#ifndef PROFILING_H_
#define PROFILING_H_

#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <ratio>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "omp.h"

class stats {
  using clock = std::chrono::high_resolution_clock;
  using timepoint = std::chrono::time_point<clock>;

  stats() : init_time(clock::now()) {}
  ~stats() = default;

  stats(stats&) = delete;
  stats& operator=(const stats&) = delete;
  stats(stats&&) = delete;
  stats& operator=(stats&&) = delete;

  struct holder_base {
    bool per_thread;
    virtual ~holder_base() = default;
  };

  template <typename T>
  struct global_holder : holder_base {
    std::vector<std::pair<timepoint, T>> values;
    global_holder() { per_thread = false; }
  };

  template <typename T>
  struct thread_holder : holder_base {
    std::vector<std::vector<std::pair<timepoint, T>>> values;
    thread_holder(int num_threads) : values(num_threads) { per_thread = true; }
  };

  std::map<std::string, std::unique_ptr<holder_base>> stats_map;
  timepoint init_time;

  static stats& get_instance() {
    static stats s;
    return s;
  }

public:
  /*  Used to initialize the statistics, this is not thread-safe.
      It should be only invoked by the main thread.  */
  template <typename StatT>
  static void register_stat(const std::string& name, bool per_thread) {
    static int num_threads = omp_get_max_threads();
    auto& s = get_instance();

    if (s.stats_map.count(name)) {
      throw std::runtime_error("register_stat: stat already registered: " + name);
    }

    if (per_thread)
      s.stats_map[name] = std::make_unique<thread_holder<StatT>>(num_threads);
    else
      s.stats_map[name] = std::make_unique<global_holder<StatT>>();
  }

  /*  Pushing non-per-thread stats is not thread-safe, it should only be performed in single-thread regions.  */
  template <typename StatT>
  static void push_stat(const std::string& name, StatT value) {
    thread_local int tid = omp_get_thread_num();
    auto& s = get_instance();

    auto it = s.stats_map.find(name);
    if (it == s.stats_map.end())
      throw std::runtime_error("push_stat: stat not registered " + name);

    bool per_thread = it->second->per_thread;
    if (per_thread) {
      auto h = dynamic_cast<thread_holder<StatT>*>(it->second.get());
      if (!h)
        throw std::runtime_error("push_stat: type mismatch for stat " + name);

      h->values[tid].push_back({clock::now(), value});
    } else { // not thread-safe, only push in a single-thread region
      auto h = dynamic_cast<global_holder<StatT>*>(it->second.get());
      if (!h)
        throw std::runtime_error("push_stat: type mismatch for stat " + name);
      h->values.push_back({clock::now(), value});
    }
  }

  template <typename StatT, bool ThreadStat>
  static std::conditional_t<
      ThreadStat,
      std::vector<std::vector<std::pair<timepoint, StatT>>>&,
      std::vector<std::pair<timepoint, StatT>>&>
  get_stat(const std::string& name) {
    auto& s = get_instance();

    auto it = s.stats_map.find(name);
    if (it == s.stats_map.end())
      throw std::runtime_error("get_stat: stat not registered " + name);

    bool per_thread = s.stats_map[name]->per_thread;
    if (ThreadStat != per_thread)
      throw std::runtime_error("get_stat: global/thread mismatch for stat " + name);

    if constexpr (ThreadStat) {
      auto h = dynamic_cast<thread_holder<StatT>*>(it->second.get());
      if (!h) throw std::runtime_error("get_stat: type mismatch for stat " + name);
      return h->values;
    } else {
      auto h = dynamic_cast<global_holder<StatT>*>(it->second.get());
      if (!h) throw std::runtime_error("get_stat: type mismatch for stat " + name);
      return h->values;
    }
  }

  template <typename StatT, bool ThreadStat>
  static std::conditional_t<ThreadStat, std::vector<double>, double> get_stat_average(const std::string& name) {
    auto accumulate_on_second = [](StatT& acc, std::pair<timepoint, StatT>& pair) {
      return acc + pair.second;
    };

    if constexpr (ThreadStat) {
      const auto& per_thread_values = get_stat<StatT, true>(name);
      std::vector<double> avgs(per_thread_values.size());

      for (std::size_t i = 0; i < per_thread_values.size(); ++i) {
        const auto& thread_values = per_thread_values[i];
        if (thread_values.empty()) {
          avgs[i] = 0.0;
          continue;
        }

        StatT sum = std::accumulate(thread_values.begin(), thread_values.end(), static_cast<StatT>(0), accumulate_on_second);
        avgs[i] = static_cast<double>(sum) / thread_values.size();
      }
      return avgs;
    } else {
      const auto& global_values = get_stat<StatT, false>(name);

      if (global_values.empty()) return 0.0;

      StatT sum = std::accumulate(global_values.begin(), global_values.end(), static_cast<StatT>(0), accumulate_on_second);
      return static_cast<double>(sum) / global_values.size();
    }
  }

  template <typename StatT, bool ThreadStat>
  static void print_stat(const std::string& name) {
    auto& stats = stats::get_instance();
    if constexpr (ThreadStat) {
      auto stat = stats::get_stat<std::size_t, true>(name);

      std::cout << "thread_stat: " << name << " " << omp_get_max_threads() << std::endl;
      for (int tid = 0; tid < omp_get_max_threads(); tid++) {
        std::cout << "\ttid: " << tid << "  " << stat[tid].size() << std::endl;

        for (size_t i = 0; i < stat[tid].size(); i++) {
          const auto duration = stat[tid][i].first - stats.init_time;
          std::cout << duration.count() << " ";
        }
        std::cout << std::endl;

        for (size_t i = 0; i < stat[tid].size(); i++) {
          std::cout << stat[tid][i].second << " ";
        }
        std::cout << std::endl;
      }

    } else {
      auto stat = stats::get_stat<std::size_t, false>(name);

      std::cout << "global_stat: " << name << " " << stat.size() << std::endl;
      for (size_t i = 0; i < stat.size(); i++) {
        const auto duration = stat[i].first - stats.init_time;
        std::cout << duration.count() << " ";
      }
      std::cout << std::endl;

      for (size_t i = 0; i < stat.size(); i++) {
        std::cout << stat[i].second << " ";
      }
      std::cout << std::endl;
    }
  }
};

#endif