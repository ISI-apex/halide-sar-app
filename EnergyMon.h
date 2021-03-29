#pragma once

#include <cstdint>
#include <string>

#if defined(WITH_ENERGYMON)
#include <energymon.h>
#endif // WITH_ENERGYMON

/**
 * C++ wrapper for an energymon if it's available, otherwise functions are
 * essentially no-ops.
 */
class EnergyMon {
public:
    EnergyMon();
    ~EnergyMon();

    uint64_t read_total_uj();
    std::string get_source();
    uint64_t get_interval_us();
    uint64_t get_precision_uj();
    bool is_exclusive();

private:
#if defined(WITH_ENERGYMON)
    energymon em;
#endif // WITH_ENERGYMON
};
