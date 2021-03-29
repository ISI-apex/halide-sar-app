#pragma once

#include <iostream>
#include <stdexcept>
#include <vector>

#include "AdaptiveController.h"

template <typename ActuatorConfig>
class Actuator {
public:
    Actuator(const std::vector<ActuatorConfig> configs, size_t start_idx) :
        configs(configs),
        id_last(start_idx) {}

    virtual void actuate(const ActuatorConfig &config) = 0;

    void set_schedule(const ControlWindowSchedule &sched) {
        if (sched.cfg_lower >= configs.size() || sched.cfg_upper >= configs.size()) {
            throw std::runtime_error("Schedule configs not in range [0, configs.size())");
        }
        this->sched = sched;
    }

    // May actuate, depending on the schedule
    // When a window periods expires, call this AFTER set_schedule
    void iteration() {
        size_t id;
        if (sched.t_lower > 0) {
            id = sched.cfg_lower;
            sched.t_lower--;
        } else {
            id = sched.cfg_upper;
        }
        if (id >= configs.size()) {
            // NOOP: occurs when configs is empty and/or no schedule is set yet and start_idx is out of range
            return;
        }
        if (id != this->id_last) {
            std::cout << "Actuator: actuating config: " << id << std::endl;
            this->id_last = id;
            actuate(configs[id]);
        }
    }

private:
    std::vector<ActuatorConfig> configs;
    ControlWindowSchedule sched;
    size_t id_last;
};

typedef struct HalideThreadActuatorConfig {
    int n_threads;
} HalideThreadActuatorConfig;

class HalideThreadActuator : public Actuator<HalideThreadActuatorConfig> {
public:
    HalideThreadActuator();
    HalideThreadActuator(const std::vector<HalideThreadActuatorConfig> configs, size_t start_idx);
    virtual void actuate(const HalideThreadActuatorConfig &config);
};
