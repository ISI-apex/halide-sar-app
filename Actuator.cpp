#include <Halide.h>

#include "Actuator.h"

using namespace std;

HalideThreadActuator::HalideThreadActuator() : HalideThreadActuator({}, 0) {}

HalideThreadActuator::HalideThreadActuator(const vector<HalideThreadActuatorConfig> configs, size_t start_idx) :
    Actuator(configs, start_idx) {}

void HalideThreadActuator::actuate(const HalideThreadActuatorConfig &config) {
    cout << "HalideThreadActuator::actuate: threads=" << config.n_threads << endl;
    halide_shutdown_thread_pool();
    halide_set_num_threads(config.n_threads);
}
