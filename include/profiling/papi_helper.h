#ifndef PAPI_HELPER_
#define PAPI_HELPER_

#include <string>
#include <vector>

#include "omp.h"
#include "papi.h"

class papi_helper {
private:
  static unsigned long get_thread_num(void) {
    return (unsigned long)omp_get_thread_num();
  }

  papi_helper() {
    int retval;
    if ((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT) {
      std::cerr << "PAPI_library_init error: " << retval << std::endl;
      exit(1);
    }

    if ((retval = PAPI_thread_init(get_thread_num)) != PAPI_OK) {
      std::cerr << "PAPI_thread_init error: " << retval << std::endl;
      exit(1);
    }

    char code_name[PAPI_MAX_STR_LEN];
    int default_events[] = {PAPI_TOT_CYC, PAPI_TOT_INS, PAPI_L1_DCM, PAPI_L2_DCM};

    for (auto i = 0; i < std::size(default_events); i++) {
      if ((retval = PAPI_event_code_to_name(default_events[i], code_name)) != PAPI_OK) {
        std::cerr << "PAPI_event_code_to_name error: " << retval << std::endl;
        exit(1);
      }

      if ((retval = PAPI_query_event(default_events[i]) != PAPI_OK)) {
        std::cerr << "PAPI event not supported: " << code_name << std::endl;
        continue;
      }

      supported_events_.push_back(default_events[i]);
      supported_names_.push_back(code_name);
    }

    if (supported_events_.empty()) {
      std::cerr << "No supported PAPI events found" << std::endl;
    }

    thread_values_.resize(omp_get_max_threads());
    for (auto i = 0; i < thread_values_.size(); i++) {
      thread_values_[i].resize(supported_events_.size());
    }
  }

  ~papi_helper() {}

  std::vector<int> supported_events_;
  std::vector<std::string> supported_names_;
  std::vector<std::vector<long long>> thread_values_;

  static papi_helper& get_instance() {
    static papi_helper instance;
    return instance;
  }

public:
  static void initialize() {
    get_instance();
  }

  static int* get_events() {
    return get_instance().supported_events_.data();
  }

  static int get_num_events() {
    return get_instance().supported_events_.size();
  }

  static long long* get_thread_values(int tid) {
    return get_instance().thread_values_[tid].data();
  }

  static void print_values() {
    auto& instance = get_instance();
    for (auto i = 0; i < instance.thread_values_.size(); i++) {
      std::cout << "Thread " << i << std::endl;
      for (auto j = 0; j < instance.thread_values_[i].size(); j++) {
        std::cout << instance.supported_names_[j] << ": " << instance.thread_values_[i][j] << std::endl;
      }
    }
  }
};

#endif