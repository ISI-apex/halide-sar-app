#pragma once

#include <cstdint>
#include <string>

#if defined(WITH_HBS)
#include <heartbeats-simple.h>
#endif // WITH_HBS

/**
 * C++ wrapper for a heartbeat_acc_pow if it's available, otherwise functions
 * are essentially no-ops.
 */
class HeartbeatAccPow {
public:
    HeartbeatAccPow(uint64_t window_len = 1,
                    std::string log = std::string(),
                    bool overwrite = false);
    ~HeartbeatAccPow();

    void heartbeat(uint64_t id, uint64_t work,
                   uint64_t start_ns, uint64_t stop_ns,
                   uint64_t accuracy,
                   uint64_t start_uj, uint64_t end_uj);
    uint64_t get_window_time();
    uint64_t get_window_work();
    double get_window_perf();
    uint64_t get_window_accuracy();
    double get_window_accuracy_rate();
    uint64_t get_window_energy();
    double get_window_power();

private:
#if defined(WITH_HBS)
    heartbeat_acc_pow_container hb;
#endif // WITH_HBS
};
