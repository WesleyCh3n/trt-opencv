#ifndef __UTILS_HPP__
#define __UTILS_HPP__
#include <chrono>
#include <sstream>
#include <string>

// clang-format off
#ifdef _OPENMP
  #include <omp.h> // This line won't add the library if you don't compile with -fopenmp option.
  #define OMP_FOR _Pragma("omp parallel for")
  #define OMP_FOR_NUM(n) _Pragma("omp parallel for num_threads(n)")
#else
  #define OMP_FOR
  #define OMP_FOR_NUM(n)
#endif
// clang-format on

template <typename OstreamImpl> std::string ostm2str(OstreamImpl v) {
  std::stringstream ss;
  ss << v;
  return ss.str();
}

class Timer {
public:
  /**
   * @brief Construct a new Timer. record the construct time
   */
  Timer() { start_time_ = std::chrono::high_resolution_clock::now(); };

  /**
   * @brief Destroy the Timer
   * @details calculate lap time since last start_time_
   */
  ~Timer() { lap_time(); };

  /**
   * @brief reset the timer start_time_
   */
  void reset() { start_time_ = std::chrono::high_resolution_clock::now(); };

  /**
   * @brief get the lap time since last start_time_
   * @return the lap time in nanoseconds
   * */
  std::chrono::nanoseconds lap_time() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now() - start_time_); // nanoseconds
  };

private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};

#endif // __UTILS_HPP__
