#pragma once

#include <cstdint>
#include <vector>

typedef struct ControlState {
    // rates (like performance or power), normalized to lowest config
    double xup;
    double cost_value;
} ControlState;

typedef enum ControlObjective {
    MINIMIZE, // cost function
    MAXIMIZE  // value function
} ControlObjective;

typedef struct ControlWindowSchedule {
    size_t cfg_lower;
    size_t cfg_upper;
    uint64_t t_lower; // iterations to spend in cfg_lower: 0 <= t_lower <= window_len
} ControlWindowSchedule;

class AdaptiveController {
public:
    AdaptiveController();
    AdaptiveController(const std::vector<ControlState> model, size_t cfg_start, double constraint,
                       uint64_t window_len);

    ControlWindowSchedule adapt(double constraint_measurement);

protected:
    double estimate_base_workload(double current_workload, double last_xup);
    void calculate_xup(double achieved, double w);
    void schedule_pair(size_t cfg_upper, size_t cfg_lower, uint64_t *low_state_iters, double *cost);
    virtual ControlWindowSchedule find_schedule() {
        throw std::runtime_error("find_schedule() not implemented");
    };

    // Represents state of kalman filter
    struct filter_state {
        double x_hat_minus;
        double x_hat;
        double p_minus;
        double h;
        double k;
        double p;
        // constants
        double q;
        double r;
    };

    // Container for controller state
    struct calc_xup_state {
        double u;
        double e;
        // constants
        double p1;
    };

    std::vector<ControlState> model;
    double constraint;
    uint64_t window_len;
    struct filter_state fs;
    struct calc_xup_state cxs;
};

class OptimalAdaptiveController : public AdaptiveController {
public:
    OptimalAdaptiveController(const std::vector<ControlState> model, size_t cfg_start, double constraint,
                              ControlObjective goal, uint64_t window_len);

protected:
    virtual ControlWindowSchedule find_schedule();
    
    ControlObjective goal;
};

class HeuristicAdaptiveController : public AdaptiveController {
public:
    HeuristicAdaptiveController(const std::vector<ControlState> model, size_t cfg_start, double constraint,
                                uint64_t window_len);

protected:
    virtual ControlWindowSchedule find_schedule();
};
