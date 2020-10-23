#include <stdio.h>
#include <stdlib.h>

#include <Halide.h>
#include <halide_image_io.h>

#include <cnpy.h>

#include "img_output_u8.h"

using namespace std;
using Halide::Runtime::Buffer;

int main(int argc, char **argv) {
    if (argc < 5) {
        cerr << "Usage: " << argv[0] << " <npy_dB_file> <dB_min> <dB_max> <output_file>" << endl;
        return 1;
    }
    string f_in = argv[1];
    double dB_min = atof(argv[2]);
    double dB_max = atof(argv[3]);
    string f_out = argv[4];

    // load NumPy dB data
    cnpy::NpyArray npydata = cnpy::npy_load(f_in);
    if (npydata.shape.size() != 2) {
        cerr << "Bad shape!" << endl;
        return 1;
    }
    double *data = static_cast<double *>(malloc(npydata.shape[0] * npydata.shape[1] * sizeof(double)));
    memcpy(data, npydata.data<double>(), npydata.shape[0] * npydata.shape[1] * sizeof(double));
    Buffer<double, 2> buf_data(data, npydata.shape[1], npydata.shape[0]);

    // create image and save
    Buffer<uint8_t, 2> buf_out(npydata.shape[1], npydata.shape[0]);
    int rv = img_output_u8(buf_data, dB_min, dB_max, buf_out);
    if (!rv) {
        Halide::Tools::convert_and_save_image(buf_out, f_out);
    }
    return rv;
}
