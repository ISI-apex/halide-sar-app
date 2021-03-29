#include <stdexcept>

#if defined(WITH_ENERGYMON)
#include <energymon-default.h>
#endif // WITH_ENERGYMON

#include "EnergyMon.h"

using namespace std;

EnergyMon::EnergyMon() {
#if defined(WITH_ENERGYMON)
    if (energymon_get_default(&em)) {
        throw runtime_error("EnergyMon::energymon_get_default");
    }
    if (em.finit(&em)) {
        throw runtime_error("EnergyMon::finit");
    }
#endif // WITH_ENERGYMON
}

EnergyMon::~EnergyMon() {
#if defined(WITH_ENERGYMON)
    if (em.ffinish(&em)) {
        perror("~EnergyMon::ffinish");
    }
#endif // WITH_ENERGYMON
}

uint64_t EnergyMon::read_total_uj() {
#if defined(WITH_ENERGYMON)
    return em.fread(&em);
#else
    return 0;
#endif // WITH_ENERGYMON
}

string EnergyMon::get_source() {
#if defined(WITH_ENERGYMON)
    char source[64] = { 0 };
    if (!em.fsource(source, sizeof(source))) {
        throw runtime_error("EnergyMon::fsource");
    }
    return string(source);
#else
    return "";
#endif // WITH_ENERGYMON
}

uint64_t EnergyMon::get_interval_us() {
#if defined(WITH_ENERGYMON)
    return em.finterval(&em);
#else
    return 0;
#endif // WITH_ENERGYMON
}

uint64_t EnergyMon::get_precision_uj() {
#if defined(WITH_ENERGYMON)
    return em.fprecision(&em);
#else
    return 0;
#endif // WITH_ENERGYMON
}


bool EnergyMon::is_exclusive() {
#if defined(WITH_ENERGYMON)
    return em.fexclusive();
#else
    return 0;
#endif // WITH_ENERGYMON
}
