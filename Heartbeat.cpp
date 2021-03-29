#include <cstring>
#include <iostream>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "Heartbeat.h"

using namespace std;

static bool file_exists(const string& name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    }
    return false;
}

HeartbeatAccPow::HeartbeatAccPow(uint64_t window_len, string log, bool overwrite) {
    if (!window_len) {
        throw range_error("Window length must be > 0");
    }
#if defined(WITH_HBS)
    int fd = -1;
    if (!log.empty()) {
        if (!overwrite && file_exists(log)) {
            throw ios_base::failure(log + ": " + strerror(EEXIST));
        }
        if ((fd = open(log.c_str(),
                       O_CREAT | O_WRONLY | O_TRUNC,
                       S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)) < 0) {
            throw ios_base::failure(log + ": open: " + strerror(errno));
        }
        if (hb_acc_pow_log_header(fd)) {
            int e = errno;
            close(fd);
            throw ios_base::failure(string("hb_acc_pow_log_header: ") + strerror(e));
        }
    }
    if (heartbeat_acc_pow_container_init_context(&hb, window_len, fd, NULL)) {
        int e = errno;
        if (fd >= 0) {
            close(fd);
        }
        throw ios_base::failure(string("heartbeat_acc_container_init_context: ") + strerror(e));
    }
#endif // WITH_HBS
}

HeartbeatAccPow::~HeartbeatAccPow() {
#if defined(WITH_HBS)
    int fd = hb_acc_pow_get_log_fd(&hb.hb);
    // flush any remaining log entries
    if (fd >= 0 && hb_acc_pow_log_window_buffer(&hb.hb, fd)) {
        perror("hb_acc_pow_log_window_buffer");
    }
    heartbeat_acc_pow_container_finish(&hb);
    if (fd >= 0 && close(fd)) {
        perror("close");
    }
#endif // WITH_HBS
}

void HeartbeatAccPow::heartbeat(uint64_t id, uint64_t work,
                                uint64_t start_ns, uint64_t stop_ns,
                                uint64_t accuracy,
                                uint64_t start_uj, uint64_t end_uj) {
#if defined(WITH_HBS)
    heartbeat_acc_pow(&hb.hb, id, work, start_ns, stop_ns, accuracy, start_uj, end_uj);
#endif // WITH_HBS
}

uint64_t HeartbeatAccPow::get_window_time() {
#if defined(WITH_HBS)
    return hb_acc_pow_get_window_time(&hb.hb);
#else
    return 0;
#endif // WITH_HBS
}

uint64_t HeartbeatAccPow::get_window_work() {
#if defined(WITH_HBS)
    return hb_acc_pow_get_window_work(&hb.hb);
#else
    return 0;
#endif // WITH_HBS
}

double HeartbeatAccPow::get_window_perf() {
#if defined(WITH_HBS)
    return hb_acc_pow_get_window_perf(&hb.hb);
#else
    return 0;
#endif // WITH_HBS
}

uint64_t HeartbeatAccPow::get_window_accuracy() {
#if defined(WITH_HBS)
    return hb_acc_pow_get_window_accuracy(&hb.hb);
#else
    return 0;
#endif // WITH_HBS
}

double HeartbeatAccPow::get_window_accuracy_rate() {
#if defined(WITH_HBS)
    return hb_acc_pow_get_window_accuracy_rate(&hb.hb);
#else
    return 0;
#endif // WITH_HBS
}

uint64_t HeartbeatAccPow::get_window_energy() {
#if defined(WITH_HBS)
    return hb_acc_pow_get_window_energy(&hb.hb);
#else
    return 0;
#endif // WITH_HBS
}

double HeartbeatAccPow::get_window_power() {
#if defined(WITH_HBS)
    return hb_acc_pow_get_window_power(&hb.hb);
#else
    return 0;
#endif // WITH_HBS
}
