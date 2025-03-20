#include <iostream>
#include <numa.h>
#include <omp.h>
#include <sched.h>
#include <string>
#include <vector>

using namespace std;

static char* cpuset_to_cstr(cpu_set_t* mask, char* str) {
  char* ptr = str;
  int i, j, entry_made = 0;
  for (i = 0; i < CPU_SETSIZE; i++) {
    if (CPU_ISSET(i, mask)) {
      int run = 0;
      entry_made = 1;
      for (j = i + 1; j < CPU_SETSIZE; j++) {
        if (CPU_ISSET(j, mask))
          run++;
        else
          break;
      }
      if (!run)
        sprintf(ptr, "%d,", i);
      else if (run == 1) {
        sprintf(ptr, "%d,%d,", i, i + 1);
        i++;
      } else {
        sprintf(ptr, "%d-%d,", i, i + run);
        i += run;
      }
      while (*ptr != 0)
        ptr++;
    }
  }
  ptr -= entry_made;
  *ptr = 0;
  return (str);
}

int main() {

  int max_threads = omp_get_max_threads();
  cout << "Max threads: " << max_threads << endl;
  vector<string> threads_info(max_threads);
  vector<string> numa_info(max_threads);

  int numa_nodes = numa_num_configured_nodes();
  cout << "NUMA nodes: " << numa_nodes << endl;

  for (int i = 0; i < numa_nodes; i++) {
    for (int j = i + 1; j < numa_nodes; j++)
      cout << "(" << i << " " << j << ") distance: " << numa_distance(i, j) << endl;
  }

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int cpu = sched_getcpu();

    cpu_set_t cpu_set;
    int affinity = sched_getaffinity(0, sizeof(cpu_set_t), &cpu_set);
    char set_str[1024];
    cpuset_to_cstr(&cpu_set, set_str);
    threads_info[tid] = "Thread " + to_string(tid) + " is running on CPU " + to_string(cpu) + " with affinity " + set_str + " NUMA node " + to_string(numa_node_of_cpu(cpu));
  }

  for (int i = 0; i < max_threads; i++) {
    cout << threads_info[i] << endl;
  }
}