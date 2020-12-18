#pragma once

#include <unistd.h>

void dft_init_fftw(size_t N_fft);
void dft_destroy_fftw(void);
