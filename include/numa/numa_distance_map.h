#ifndef NUMA_DISTANCE_MAP_H_
#define NUMA_DISTANCE_MAP_H_

#include <algorithm>
#include <numa.h>
#include <omp.h>
#include <random>
#include <sched.h>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

class numa_distance_map {
public:
  // Get thread IDs at a specific distance from the given thread ID
  static std::vector<int> get_threads_at_distance(int tid, int distance) {
    auto& instance = numa_distance_map::instance();
    if (!instance.is_initialized_) {
      throw std::runtime_error("NUMA distance map not initialized. Call initialize() first.");
    }

    if (tid < 0 || tid >= static_cast<int>(instance.tid_to_numa_.size())) {
      throw std::out_of_range("Thread ID out of range");
    }

    if (distance < 0 || distance >= instance.num_distances_) {
      throw std::out_of_range("Distance out of range");
    }

    return instance.thread_distance_map_[tid][distance];
  }

  // Get the number of distance levels
  static int get_num_distances() {
    auto& instance = numa_distance_map::instance();
    if (!instance.is_initialized_) {
      throw std::runtime_error("NUMA distance map not initialized. Call initialize() first.");
    }
    return instance.num_distances_;
  }

  // Manual initialization method - can be called from anywhere
  static void initialize() {
    auto& instance = numa_distance_map::instance();
    if (instance.is_initialized_) {
      return; // Already initialized
    }

    if (numa_available() == -1) {
      throw std::runtime_error("NUMA is not available on this system");
    }

    int num_threads = omp_get_max_threads();
    instance.tid_to_numa_.resize(num_threads, -1);
    std::unordered_map<int, std::vector<int>> numa_to_tids;

// First, map threads to NUMA nodes
#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      int node = numa_node_of_cpu(sched_getcpu());
      instance.tid_to_numa_[tid] = node;

#pragma omp critical
      {
        numa_to_tids[node].push_back(tid);
      }
    }

    // Get all unique NUMA nodes that have threads
    std::vector<int> used_nodes;
    for (const auto& pair : numa_to_tids) {
      used_nodes.push_back(pair.first);
    }

    // Build a map of distances between all nodes
    std::vector<std::vector<int>> node_distances(used_nodes.size(), std::vector<int>(used_nodes.size()));
    for (size_t i = 0; i < used_nodes.size(); i++) {
      for (size_t j = 0; j < used_nodes.size(); j++) {
        if (i == j) {
          node_distances[i][j] = 0; // Same node distance is 0
        } else {
          node_distances[i][j] = numa_distance(used_nodes[i], used_nodes[j]);
        }
      }
    }

    // Find all unique distance values and sort them
    std::set<int> unique_distances;
    for (const auto& row : node_distances) {
      for (int dist : row) {
        if (dist > 0) { // Ignore self-distance (0)
          unique_distances.insert(dist);
        }
      }
    }

    // Convert set to vector and sort
    std::vector<int> distance_values(unique_distances.begin(), unique_distances.end());
    std::sort(distance_values.begin(), distance_values.end());

    // Store the number of distances (+1 for same-node/distance-0)
    instance.num_distances_ = distance_values.size() + 1;

    // Create a mapping from distance value to distance level
    std::unordered_map<int, int> distance_to_level;
    for (size_t i = 0; i < distance_values.size(); i++) {
      distance_to_level[distance_values[i]] = i + 1; // Level 0 is for same node
    }

    // Create random number generator
    std::random_device rd;
    std::mt19937 g(rd());

    // Build the thread distance map
    instance.thread_distance_map_.resize(num_threads);
    for (int tid = 0; tid < num_threads; tid++) {
      int source_node = instance.tid_to_numa_[tid];
      int source_node_index = std::find(used_nodes.begin(), used_nodes.end(), source_node) - used_nodes.begin();

      // Initialize array of vectors for each distance level
      instance.thread_distance_map_[tid].resize(instance.num_distances_);

      // Process each thread
      for (int other_tid = 0; other_tid < num_threads; other_tid++) {
        if (other_tid == tid)
          continue; // Skip self

        int target_node = instance.tid_to_numa_[other_tid];
        int target_node_index = std::find(used_nodes.begin(), used_nodes.end(), target_node) - used_nodes.begin();

        int distance = node_distances[source_node_index][target_node_index];

        if (distance == 0) {
          // Same node, distance level 0
          instance.thread_distance_map_[tid][0].push_back(other_tid);
        } else {
          // Different node, find the level based on distance
          int level = distance_to_level[distance];
          instance.thread_distance_map_[tid][level].push_back(other_tid);
        }
      }

      // Randomize the order of thread IDs at each distance level
      for (auto& threads : instance.thread_distance_map_[tid]) {
        std::shuffle(threads.begin(), threads.end(), g);
      }
    }

    instance.is_initialized_ = true;
  }

  // Check if initialized
  static bool is_initialized() {
    return numa_distance_map::instance().is_initialized_;
  }

private:
  numa_distance_map() : is_initialized_(false), num_distances_(0) {}

  static numa_distance_map& instance() {
    static numa_distance_map instance;
    return instance;
  }

  // Prevent copying and assignment
  numa_distance_map(const numa_distance_map&) = delete;
  numa_distance_map& operator=(const numa_distance_map&) = delete;
  numa_distance_map(numa_distance_map&&) = delete;
  numa_distance_map& operator=(numa_distance_map&&) = delete;

  std::vector<int> tid_to_numa_; // Still needed internally
  std::vector<std::vector<std::vector<int>>> thread_distance_map_;
  bool is_initialized_;
  int num_distances_;
};

#endif