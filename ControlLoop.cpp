#include <cfloat>
#include <cstdlib>
#include <thread>

#include "Actuator.h"
#include "AdaptiveController.h"
#include "ControlLoop.h"

using namespace std;
using namespace std::chrono;

// Performance/power models from Summit for different thread counts
#define PERF_PWR_CONFIG \
    { 1 }, \
    { 2 }, \
    { 4 }, \
    { 8 }, \
    { 16 }, \
    { 21 }, \
    { 42 }, \
    { 84 }, \
    /* To keep model sorted, we can't use the maximum thread count */ \
    /* { 168 } */
// Sandia dataset, base config: perf = 0.000936, pwr = 448.081067
#define PERF_PWR_MODEL \
    { 1, 1 }, \
    { 1.705128205, 1.014495319 }, \
    { 3.507478632, 1.045825833 }, \
    { 7.012820513, 1.106682508 }, \
    { 13.50747863, 1.228656934 }, \
    { 17.60683761, 1.302667747 }, \
    { 29.66452991, 1.504471005 }, \
    { 35.78418803, 1.523596937 }, \
    /* To keep model sorted, we can't use the maximum thread count */ \
    /* { 33.77991453, 1.536675996 } */

#define ENV_PERF_GOAL "SARBP_PERFORMANCE_CONSTRAINT"

ControlLoop::ControlLoop(uint64_t window_len, string hb_log, bool overwrite) :
    hb(window_len, hb_log, overwrite),
    ac(),
    actuator(),
    window_len(window_len),
    count(0) {
    cout << "ControlLoop: window_len=" << window_len << endl;

#if 0
    cout << "ControlLoop: using hard-coded model" << endl;
    vector<ControlState> model = { PERF_PWR_MODEL };
    vector<HalideThreadActuatorConfig> configs = { PERF_PWR_CONFIG };
#else
    cout << "ControlLoop: using naive model" << endl;
    // build a naive thread performance/cost model
    vector<ControlState> model = { { 1.0, 1.0 } };
    vector<HalideThreadActuatorConfig> configs = { { 1 } };
    const int max_concurrency = thread::hardware_concurrency(); // could be 0
    for (int i = 2; i <= max_concurrency; i++) {
        model.push_back({ (double)i, (double)i });
        configs.push_back({ i });
    }
#endif

    // determine the starting configuration
    const char *env_num_threads = getenv("HL_NUM_THREADS");
    const int threads_start = env_num_threads ? max(atoi(env_num_threads), 1) : configs.back().n_threads;
    size_t configs_start_idx = 0;
    bool start_idx_found = false;
    for (size_t i = 0; i < configs.size(); i++) {
        if (configs[i].n_threads == threads_start) {
            start_idx_found = true;
            configs_start_idx = i;
            cout << "ControlLoop: configs_start_idx=" << i << ", threads=" << configs[i].n_threads << endl;
            break;
        }
    }
    if (!start_idx_found) {
        cerr << "ControlLoop: Failed to determine starting configuration" << endl;
        configs_start_idx = configs.size() - 1;
        cerr << "ControlLoop: Will force maximum known configuration at index: " << configs_start_idx << endl;
    }

    // get the performance goal
    const char *env_perf_goal = getenv(ENV_PERF_GOAL);
    const double perf_goal = env_perf_goal ? atof(env_perf_goal) : DBL_MAX;
    cout << "ControlLoop: perf_goal=" << perf_goal << endl;
    // construct the controller and actuator
#if 0
    cout << "ControlLoop: using optimal controller" << endl;
    this->ac = make_unique<OptimalAdaptiveController>(model, configs_start_idx, perf_goal, MINIMIZE, window_len);
#else
    cout << "ControlLoop: using heuristic controller" << endl;
    this->ac = make_unique<HeuristicAdaptiveController>(model, configs_start_idx, perf_goal, window_len);
#endif
    this->actuator = make_unique<HalideThreadActuator>(configs, configs_start_idx);
    if (!start_idx_found) {
        actuator->actuate(configs[configs_start_idx]);
    }
}

void ControlLoop::iteration(uint64_t work, uint64_t accuracy,
                            time_point<steady_clock> start_time,
                            time_point<steady_clock> stop_time,
                            uint64_t start_uj, uint64_t stop_uj) {
    // Observe: capture runtime metrics
    uint64_t start_ns = time_point_cast<nanoseconds>(start_time).time_since_epoch().count();
    uint64_t stop_ns = time_point_cast<nanoseconds>(stop_time).time_since_epoch().count();
    hb.heartbeat(count, work, start_ns, stop_ns, accuracy, start_uj, stop_uj);
    if ((count + 1) % window_len == 0) {
        cout << "Performance: " << hb.get_window_perf() << endl;
        // Decide: make control decision(s)
        ControlWindowSchedule sched = ac->adapt(hb.get_window_perf());
        cout << "New schedule: " << sched.cfg_lower << "," << sched.cfg_upper << "," << sched.t_lower << endl;
        actuator->set_schedule(sched);
    }
    // Act: perform control action(s) according to the schedule
    // there's no schedule until at least one window period completes
    if (count + 1 >= window_len) {
        actuator->iteration();
    }
    count++;
}

void ControlLoop::iteration_begin() {
    start_time = steady_clock::now();
    start_uj = em.read_total_uj();
}

void ControlLoop::iteration_end(uint64_t work, uint64_t accuracy) {
    stop_uj = em.read_total_uj();
    stop_time = steady_clock::now();
    iteration(work, accuracy, start_time, stop_time, start_uj, stop_uj);
}

uint64_t ControlLoop::get_last_iteration_ms() {
    return duration_cast<milliseconds>(stop_time - start_time).count();
}
