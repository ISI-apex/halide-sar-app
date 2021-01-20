# SAR Data

This directory contains example SAR data.

## Resizing data

The `resize.py` script supports *some* data resizing.
For example, to double the number of data pulses and samples per pulse:

```sh
./resize.py ./AFRL/pass1/HH_npy AFRL_p2_s2 -p 2 -s 2
```

Depending on the data size, changes to backprojection parameters may be needed to produce a good output image.
For the above example, set `--res-factor=0.5` when running `sarbp` (in addition to `--platform-dir=../data/AFRL_p2_s2`).
