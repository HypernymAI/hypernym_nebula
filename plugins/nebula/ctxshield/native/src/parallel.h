// Fixed fan-out over contiguous chunks (build doc §9.2.5): no locks on the hot path; the first
// worker exception is re-raised on the calling thread. threads <= 0 → P-core count.
#pragma once
#include <algorithm>
#include <cstddef>
#include <exception>
#include <thread>
#include <vector>

#include "shield.h"

namespace ctxshield {

template <typename Fn>
void parallel_chunks(size_t n, int threads, Fn fn) {
    int T = threads > 0 ? threads : default_threads();
    T = std::max(1, std::min<int>(T, static_cast<int>(n > 0 ? n : 1)));
    if (T == 1) { fn(size_t(0), n); return; }   // no thread spawn for one worker
    std::vector<std::thread> pool;
    std::vector<std::exception_ptr> errs(static_cast<size_t>(T));
    const size_t chunk = (n + static_cast<size_t>(T) - 1) / static_cast<size_t>(T);
    for (int t = 0; t < T; ++t) {
        const size_t a = std::min(n, static_cast<size_t>(t) * chunk), b = std::min(n, a + chunk);
        pool.emplace_back([&, t, a, b]() {
            try { fn(a, b); } catch (...) { errs[static_cast<size_t>(t)] = std::current_exception(); }
        });
    }
    for (auto& th : pool) th.join();
    for (auto& e : errs) if (e) std::rethrow_exception(e);
}

// One scratch per OS thread, allocated once and reused across calls.
inline Scratch& thread_scratch() {
    thread_local Scratch sc;
    return sc;
}

}  // namespace ctxshield
