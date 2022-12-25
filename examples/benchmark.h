#include <iostream>
#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

template<typename... P>
void benchmark(void(*f)(P...), const P&... p, std::size_t n) {
    auto t1 = high_resolution_clock::now();

    for (std::size_t i = 0; i < n; i++) {
        f(p...);
    }

    auto t2 = high_resolution_clock::now();

    double ms = static_cast<duration<double, std::milli>>(t2 - t1).count();
    double average = ms / static_cast<double>(n); 

    std::cout << "Benchmarked " << n << " times: " << ms << "ms total, " << average << "ms on average" << std::endl;
}
