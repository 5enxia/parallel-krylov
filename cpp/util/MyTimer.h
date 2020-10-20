#include <chrono>

namespace MyTimer {
  using namespace std;

  chrono::system_clock::time_point start() {
    chrono::system_clock::time_point start;
    start = std::chrono::system_clock::now();
    return start;
  }

  long end(chrono::system_clock::time_point start) {
    chrono::system_clock::time_point end;
    end = std::chrono::system_clock::now();
    return chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
  }
} 