#include "doctest.h"

#include <thread>
#include <vector>

#include "parallel/deque.h"

using namespace std;

TEST_CASE("deque: single thread pop_bottom") {
  parallel::deque<int*> deque;
  int* a = new int(1);

  deque.push(a);
  auto val = deque.pop();

  CHECK(*val == 1);
}

TEST_CASE("deque: single thread steal") {
  parallel::deque<int*> deque;
  int* a = new int(1);

  deque.push(a);
  auto val = deque.steal();

  CHECK(*val == 1);
}

TEST_CASE("deque: one thread pushes, one thread steals") {
  parallel::deque<int*> deque;

  thread worker([&]() {
    int* a = new int(1);
    deque.push(a);
  });

  thread stealer([&]() {
    auto val = deque.steal();
    while (!val)
      val = deque.steal();
    CHECK(*val == 1);
  });

  worker.join();
  stealer.join();
}

TEST_CASE("deque: steal ring") {
  int num_threads = 4;
  vector<parallel::deque<int*>> deques(4);
  vector<thread> threads;

  for (auto i = 0; i < num_threads; i++) {
    threads.emplace_back([&, i]() {
      int* a = new int(i);
      deques[i].push(a);
      auto val = deques[(i + 1) % num_threads].steal();

      while (!val)
        val = deques[(i + 1) % num_threads].steal();

      CHECK(*val == (i + 1) % num_threads);
    });
  }

  for (auto& t : threads)
    t.join();
}
