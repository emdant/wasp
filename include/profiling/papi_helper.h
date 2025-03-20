#ifndef PAPI_HELPER_
#define PAPI_HELPER_

#include <string>
#include <vector>

#include "omp.h"
#ifdef PAPI_PROFILE
#include "papi.h"
#endif

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

    const PAPI_hw_info_t* hw_info = nullptr;
    if ((hw_info = PAPI_get_hardware_info()) == NULL) {
      std::cerr << "PAPI_get_hardware_info error: " << retval << std::endl;
      exit(1);
    }

    bool intel = false, amd = false;
    std::string vendor = hw_info->vendor_string;
    if (vendor.find("Intel") != std::string::npos)
      intel = true;
    else if (vendor.find("AMD") != std::string::npos)
      amd = true;
    else {
      std::cerr << "CPU vendor not supported" << std::endl;
      exit(1);
    }

    char code_name[PAPI_MAX_STR_LEN];
    int default_events[] = {
        PAPI_TOT_CYC,
        PAPI_TOT_INS,
        PAPI_L1_DCM,
        PAPI_L2_DCM
    };

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

    if (intel) {
      const char* intel_native_events[] = {
          // "LONGEST_LAT_CACHE.MISS",
          // "LLC_MISSES",
          // "LLC-LOAD-MISSES",
          // "LLC-STORE-MISSES",
          "MEM_LOAD_L3_MISS_RETIRED:LOCAL_DRAM",
          "MEM_LOAD_L3_MISS_RETIRED:REMOTE_DRAM",
          "MEM_LOAD_L3_MISS_RETIRED:REMOTE_FWD",
          "MEM_LOAD_L3_MISS_RETIRED:REMOTE_HITM",
      };
      for (auto i = 0; i < std::size(intel_native_events); i++) {
        int event_code;
        if ((retval = PAPI_event_name_to_code(intel_native_events[i], &event_code)) != PAPI_OK) {
          std::cerr << "PAPI_event_name_to_code error: " << retval << " for native event " << intel_native_events[i] << std::endl;
          exit(1);
        }

        if ((retval = PAPI_query_event(event_code) != PAPI_OK)) {
          std::cerr << "PAPI event not supported: " << intel_native_events[i] << std::endl;
          continue;
        }

        supported_events_.push_back(event_code);
        supported_names_.push_back(intel_native_events[i]);
      }

      derived_events_.push_back("MEM_LOAD_L3_MISS_RETIRED");
    }

    if (amd) {
      // const char* amd_native_events[] = {
      //    "LLC-LOAD-MISSES",
      //    "NODE-LOAD-MISSES",
      //    "NODE-STORE-MISSES",
      // };
      // for (auto i = 0; i < std::size(amd_native_events); i++) {
      //   int event_code;
      //   if ((retval = PAPI_event_name_to_code(amd_native_events[i], &event_code)) != PAPI_OK) {
      //     std::cerr << "PAPI_event_name_to_code error: " << retval << " for native event " << amd_native_events[i] << std::endl;
      //     exit(1);
      //   }

      //   if ((retval = PAPI_query_event(event_code) != PAPI_OK)) {
      //     std::cerr << "PAPI event not supported: " << amd_native_events[i] << std::endl;
      //     continue;
      //   }

      //   supported_events_.push_back(event_code);
      //   supported_names_.push_back(amd_native_events[i]);
      // }
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
  std::vector<std::string> derived_events_;

  static papi_helper& get_instance() {
    static papi_helper instance;
    return instance;
  }

  static std::vector<std::vector<long long>> derived_thread_values() {
    auto& instance = get_instance();

    // Derived values for each thread
    std::vector<std::vector<long long>> derived_values;
    derived_values.resize(instance.thread_values_.size());

    // For each thread id...
    for (auto i = 0; i < instance.thread_values_.size(); i++) {
      // Derived values for each derived event
      derived_values[i].resize(instance.derived_events_.size());

      // Calculate derived values for each derived event
      for (auto j = 0; j < instance.derived_events_.size(); j++) {
        auto derived_event = instance.derived_events_[j];

        // Find supported events that contain the derived event...
        std::vector<std::size_t> indices;
        for (auto k = 0; k < instance.supported_names_.size(); k++) {
          auto event = instance.supported_names_[k];
          if (event.compare(0, derived_event.length(), derived_event) == 0) {
            indices.push_back(k);
          }
        }

        // Calculate and store derived event
        if (indices.size() > 0) {
          long long sum = 0;
          for (auto k : indices)
            sum += instance.thread_values_[i][k];
          derived_values[i][j] = sum; // i = thread index, j = derived event index
        }
      }
    }
    return derived_values;
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

  static void print_perthread_values() {
    auto& instance = get_instance();

    auto d_thead_values = derived_thread_values();
    // For each thread id...
    for (auto i = 0; i < instance.thread_values_.size(); i++) {

      // Print all supported events
      std::cout << "Thread " << i << std::endl;
      for (auto j = 0; j < instance.thread_values_[i].size(); j++) {
        std::cout << instance.supported_names_[j] << ": " << instance.thread_values_[i][j] << std::endl;
      }
      for (auto j = 0; j < d_thead_values[i].size(); j++) {
        std::cout << instance.derived_events_[j] << ": " << d_thead_values[i][j] << std::endl;
      }
    }
  }

  static void print_values() {
    auto& instance = get_instance();
    std::vector<long long> avg_supported_values(instance.supported_events_.size(), 0);

    for (auto i = 0; i < instance.thread_values_.size(); i++) {
      for (auto j = 0; j < instance.thread_values_[i].size(); j++) {
        avg_supported_values[j] += instance.thread_values_[i][j];
      }
    }
    for (auto i = 0; i < avg_supported_values.size(); i++) {
      double i_value = (double)avg_supported_values[i] / instance.thread_values_.size();
      std::cout << instance.supported_names_[i] << " avg: " << i_value << std::endl;
    }

    std::vector<long long> avg_derived_values(instance.derived_events_.size(), 0);

    auto d_thread_values = derived_thread_values();
    for (auto i = 0; i < d_thread_values.size(); i++) {
      for (auto j = 0; j < d_thread_values[i].size(); j++) {
        avg_derived_values[j] += d_thread_values[i][j];
      }
    }
    for (auto i = 0; i < avg_derived_values.size(); i++) {
      double i_value = (double)avg_derived_values[i] / d_thread_values.size();
      std::cout << instance.derived_events_[i] << " avg: " << i_value << std::endl;
    }
  }
};

#endif