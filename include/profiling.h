// Copyright (c) 2024, Queen's University Belfast
// See LICENSE.txt for license details

#ifndef PROFILING_H_
#define PROFILING_H_

#include <initializer_list>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "omp.h"

#include "types.h"

unsigned long get_thread_num(void) {
  return (unsigned long)omp_get_thread_num();
}

namespace profiling {

template <typename... Types>
class threads_info_map {
public:
  threads_info_map(int num_threads, std::initializer_list<std::string> names, std::initializer_list<std::string> units)
      : num_threads(num_threads) {
    if ((sizeof...(Types) != names.size()) || (sizeof...(Types) != units.size()))
      throw std::invalid_argument("threads_info_map number of types should match number of info names and info units");

    auto it_names = names.begin();
    auto it_units = units.begin();
    ([&] {
      map[*it_names] = std::vector<std::vector<Types>>(num_threads);
      info_names.push_back(*it_names);
      unit_names.push_back(*it_units);

      it_names++;
      it_units++;
    }(),
     ...);
  }

  template <typename T>
  void push_tid_info(std::string info_name, int tid, T value) {
    std::get<std::vector<std::vector<T>>>(map[info_name])[tid].push_back(value);
  }

  template <typename T>
  std::vector<T>& get_tid_info(std::string info_name, int tid) {
    return std::get<std::vector<std::vector<T>>>(map[info_name])[tid];
  }

  void print() {
    std::cout << "prof:thread_num " << num_threads << std::endl;

    for (int tid = 0; tid < num_threads; tid++) {
      std::cout << "prof:thread " << tid << std::endl;

      std::size_t i = 0;
      ([&] {
        std::cout << "\t" << info_names[i] << " " << unit_names[i] << " ";

        auto& values = get_tid_info<Types>(info_names[i], tid);
        std::cout << values.size() << " ";

        for (auto& val : values)
          std::cout << val << " ";
        std::cout << std::endl;

        i++;
      }(),
       ...);
    }
  }

private:
  int num_threads;
  std::map<std::string, unique_variant<std::vector<std::vector<Types>>...>> map;
  std::vector<std::string> info_names;
  std::vector<std::string> unit_names;
};

template <typename... Types>
class global_info_map {
public:
  global_info_map(std::initializer_list<std::string> names, std::initializer_list<std::string> units) {
    if ((sizeof...(Types) != names.size()) || (sizeof...(Types) != units.size()))
      throw std::invalid_argument("threads_info_map number of types should match number of info names and info units");

    auto it_names = names.begin();
    auto it_units = units.begin();
    ([&] {
      map[*it_names] = std::vector<Types>();
      info_names.push_back(*it_names);
      unit_names.push_back(*it_units);

      it_names++;
      it_units++;
    }(),
     ...);
  }

  template <typename T>
  void push_info(std::string info_name, T value) {
    std::get<std::vector<T>>(map[info_name]).push_back(value);
  }

  template <typename T>
  std::vector<T>& get_info(std::string info_name) {
    return std::get<std::vector<T>>(map[info_name]);
  }

  void print() {
    std::cout << "prof:global" << std::endl;
    std::size_t i = 0;
    ([&] {
      std::cout << "\t" << info_names[i] << " " << unit_names[i] << " ";

      auto& values = get_info<Types>(info_names[i]);
      std::cout << values.size() << " ";

      for (auto& val : values)
        std::cout << val << " ";
      std::cout << std::endl;

      i++;
    }(),
     ...);
  }

private:
  std::map<std::string, unique_variant<std::vector<Types>...>> map;
  std::vector<std::string> info_names;
  std::vector<std::string> unit_names;
};

} // namespace profiling

#endif