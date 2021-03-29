#pragma once

#include <chrono>
#include <cstdint>
#include <string>

#include "Actuator.h"
#include "AdaptiveController.h"
#include "EnergyMon.h"
#include "Heartbeat.h"

/**
 * Manages an Observe-Decide-Act (ODA) loop
 */
class ControlLoop {
public:
    ControlLoop(uint64_t window_len = 1,
                std::string hb_log = std::string(),
                bool overwrite = false);

    /**
     * Start a loop iteration.
     */
    void iteration_begin();

    /**
     * End a loop iteration.
     * When a window period completes, performs the Decide and Act steps.
     */
    void iteration_end(uint64_t work, uint64_t accuracy);

    uint64_t get_last_iteration_ms();

private:
    EnergyMon em;
    HeartbeatAccPow hb;
    std::unique_ptr<AdaptiveController> ac;
    std::unique_ptr<HalideThreadActuator> actuator;
    const uint64_t window_len;
    uint64_t count;
    std::chrono::time_point<std::chrono::steady_clock> start_time;
    std::chrono::time_point<std::chrono::steady_clock> stop_time;
    uint64_t start_uj;
    uint64_t stop_uj;

    /**
     * Observes each loop iteration.
     * When a window period completes, performs the Decide and Act steps.
     */
    void iteration(uint64_t work, uint64_t accuracy,
                   std::chrono::time_point<std::chrono::steady_clock> start_time,
                   std::chrono::time_point<std::chrono::steady_clock> stop_time,
                   uint64_t start_uj, uint64_t stop_uj);
};
