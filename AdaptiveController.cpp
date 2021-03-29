#include <cfloat>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "AdaptiveController.h"

using namespace std;

// filter constants
static const double X_HAT_MINUS_START  =   0.0;
static const double X_HAT_START        =   0.2;
static const double P_START            =   1.0;
static const double P_MINUS_START      =   0.0;
static const double H_START            =   0.0;
static const double K_START            =   0.0;
static const double Q_DEFAULT          =   0.00001;
static const double R_DEFAULT          =   0.01;

// xup constants
static const double P1_DEFAULT         =   0.0;
static const double E_START            =   0.0;

static const std::vector<ControlState> MODEL_NOOP {
    {
        .xup = 1.0,
        .cost_value = 1.0
    }
};

AdaptiveController::AdaptiveController() : AdaptiveController(MODEL_NOOP, 0, 0.0, 1) {}

AdaptiveController::AdaptiveController(const vector<ControlState> model, size_t cfg_start, double constraint,
                                       uint64_t window_len) :
                                       model(model),
                                       constraint(constraint),
                                       window_len(window_len) {
    if (model.empty()) {
        throw runtime_error("model is empty");
    }
    for (size_t i = 1; i < model.size(); i++) {
        if (model[i].xup < model[i - 1].xup) {
            throw runtime_error("model not sorted by xup");
        }
    }
    if (fabs(model[0].xup - 1.0) >= numeric_limits<double>::epsilon()) {
        throw runtime_error("model xup values not normalized");
    }
    if (cfg_start >= model.size()) {
        throw runtime_error("cfg_start not in model range");
    }

    // initialize variables used in the Kalman filter
    fs.x_hat_minus = X_HAT_MINUS_START;
    fs.x_hat = X_HAT_START;
    fs.p_minus = P_MINUS_START;
    fs.h = H_START;
    fs.k = K_START;
    fs.p = P_START;
    fs.q = Q_DEFAULT;
    fs.r = R_DEFAULT;

    // initialize variables used for calculating xup
    cxs.u = model[cfg_start].xup;
    cxs.e = E_START;
    cxs.p1 = P1_DEFAULT;
}

ControlWindowSchedule AdaptiveController::adapt(double constraint_measurement) {
    const double workload = estimate_base_workload(constraint_measurement, cxs.u);
    calculate_xup(constraint_measurement, workload);
    cout << "AdaptiveController::adapt: workload=" << workload << ", xup=" << cxs.u << endl;
    return find_schedule();
}

double AdaptiveController::estimate_base_workload(double current_workload, double last_xup) {
    fs.x_hat_minus = fs.x_hat;
    fs.p_minus = fs.p + fs.q;
    fs.h = last_xup;
    fs.k = (fs.p_minus * fs.h) / ((fs.h * fs.p_minus * fs.h) + fs.r);
    fs.x_hat = fs.x_hat_minus + (fs.k * (current_workload - (fs.h * fs.x_hat_minus)));
    fs.p = (1.0 - (fs.k * fs.h)) * fs.p_minus;
    return 1.0 / fs.x_hat;
}

/*
 * Calculate the xup necessary to achieve the target constraint
 */
void AdaptiveController::calculate_xup(double achieved, double w) {
    // compute error
    cxs.e = this->constraint - achieved;
    // Calculate xup
    cxs.u = cxs.u + ((1 - cxs.p1) * (w * cxs.e));
    // Prevent windup: xup less than 1 has no effect, greater than the maximum is not achievable
    cxs.u = min(max(1.0, cxs.u), model.back().xup);
}

/*
 * Calculate the time division between the two configurations and the schedule's cost/value.
 */
void AdaptiveController::schedule_pair(size_t cfg_upper, size_t cfg_lower, uint64_t *t_lower, double *cv) {
    const double upper_xup = model[cfg_upper].xup;
    const double upper_xup_cv = model[cfg_upper].cost_value;
    const double lower_xup = model[cfg_lower].xup;
    const double lower_xup_cv = model[cfg_lower].cost_value;
    // If lower rate and upper rate are equal, no need for time division
    uint64_t low_state_iters;
    if (fabs(upper_xup - lower_xup) < numeric_limits<double>::epsilon()) {
        low_state_iters = 0;
    } else {
        const double target_xup = cxs.u;
        // x represents the percentage of iterations spent in the first (lower) configuration
        // Conversely, (1 - x) is the percentage of iterations in the second (upper) configuration
        // This equation ensures the time period of the combined rates is equal to the time period of the target rate
        // 1 / Target rate = X / (lower rate) + (1 - X) / (upper rate)
        const double x = ((upper_xup * lower_xup) - (target_xup * lower_xup)) /
                         ((upper_xup * target_xup) - (target_xup * lower_xup));
        // Num of iterations (in lower state) = x * (controller period)
        low_state_iters = (uint64_t) round(this->window_len * x);
    }
    // calculate actual cost/value
    *t_lower = low_state_iters;
    *cv = ((low_state_iters / lower_xup) * lower_xup_cv) +
          (((this->window_len - low_state_iters) / upper_xup) * upper_xup_cv);
}

OptimalAdaptiveController::OptimalAdaptiveController(const std::vector<ControlState> model, size_t cfg_start,
                                                     double constraint, ControlObjective goal, uint64_t window_len) :
                                                     AdaptiveController(model, cfg_start, constraint, window_len),
                                                     goal(goal)
{}

/**
 * Check all pairs of states that can achieve the target and choose the pair
 * with the lowest cost. Uses an O(n^2) algorithm.
 * See POET: https://doi.org/10.1109/RTAS.2015.7108419
 */
ControlWindowSchedule OptimalAdaptiveController::find_schedule() {
    ControlWindowSchedule sched = { 0 };
    double cv_best;
    switch (this->goal) {
        case MINIMIZE:
            cv_best = DBL_MAX;
            break;
        case MAXIMIZE:
        default:
            cv_best = DBL_MIN;
            break;
    }
    // i is the "upper" configuration, j is the "lower" configuration
    for (size_t i = 0; i < model.size(); i++) {
        if (model[i].xup < cxs.u) {
            continue;
        }
        for (size_t j = 0; j < model.size(); j++) {
            if (model[j].xup > cxs.u) {
                continue;
            }
            uint64_t j_iters;
            double cv;
            bool is_best;
            // find time for both configurations
            schedule_pair(i, j, &j_iters, &cv);
            switch (this->goal) {
                case MINIMIZE:
                    is_best = cv < cv_best;
                    break;
                case MAXIMIZE:
                default:
                    is_best = cv > cv_best;
                    break;
            }
            if (is_best) {
                sched.cfg_lower = j;
                sched.cfg_upper = i;
                sched.t_lower = j_iters;
                cv_best = cv;
            }
        }
    }
    return sched;
}

HeuristicAdaptiveController::HeuristicAdaptiveController(const std::vector<ControlState> model, size_t cfg_start,
                                                         double constraint, uint64_t window_len) :
                                                         AdaptiveController(model, cfg_start, constraint, window_len)
{}

/**
 * Ignores the objective and finds the pair of states that clamp the target xup.
 * Works well when this pair's schedule is close to optimal cost/value.
 * Performance is O(n), but could be O(log(n)) if we cared to be smarter.
 * See PTRADE's "proportional" translator: https://doi.org/10.1109/EMSOFT.2013.6658597
 */
ControlWindowSchedule HeuristicAdaptiveController::find_schedule() {
    ControlWindowSchedule sched = { 0 };
    // i is the "upper" configuration
    for (size_t i = 0; i < model.size(); i++) {
        if (model[i].xup < cxs.u) {
            continue;
        }
        sched.cfg_upper = i;
        if (i == 0) {
            sched.cfg_lower = sched.cfg_upper; // not really necessary
            sched.t_lower = 0;
        } else {
            double cv; // ignored
            sched.cfg_lower = sched.cfg_upper - 1;
            schedule_pair(sched.cfg_upper, sched.cfg_lower, &sched.t_lower, &cv);
        }
        break;
    }
    return sched;
}
