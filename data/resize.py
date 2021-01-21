#!/usr/bin/env python3
"""Resize RITSAR-formatted NumPy data."""

import argparse
import os
import numpy as np
from skimage import transform

__author__ = "Connor Imes"
__date__ = "2021-01-15"


# This currently only handles the fields we require, not all platform data!
def resize(idir, odir, p, s, scale_p=1.0, scale_s=1.0):
    """Resize data from idir and save to odir."""

    delta_r = np.load(os.path.join(idir, 'delta_r.npy'))
    k_r = np.load(os.path.join(idir, 'k_r.npy'))
    npulses = np.load(os.path.join(idir, 'npulses.npy'))
    nsamples = np.load(os.path.join(idir, 'nsamples.npy'))
    phs = np.load(os.path.join(idir, 'phs.npy'))
    pos = np.load(os.path.join(idir, 'pos.npy'))
    R_c = np.load(os.path.join(idir, 'R_c.npy'))

    # p takes precedence over scale_p
    if p is not None and p > 0:
        scale_p = p / npulses
    else:
        p = int(npulses * scale_p)
        if p <= 0:
            raise ValueError('Scaling pulses resulted in count <= 0')
    print('pulses = ' + str(p))

    # s takes precedence over scale_s
    if s is not None and s > 0:
        scale_s = s / nsamples
    else:
        s = int(nsamples * scale_s)
        if s <= 0:
            raise ValueError('Scaling samples resulted in count <= 0')
    print('samples = ' + str(s))

    # delta_r value is a function of freq, so value must be adjusted
    delta_r = delta_r / scale_s
    # k_r values are a function of freq, so both values and shape must be adjusted
    k_r = transform.resize(k_r * scale_s, (s,), preserve_range=True)
    npulses = p
    nsamples = s
    # transform.resize does not work for complex types
    phs_r = transform.resize(np.real(phs), (p, s), preserve_range=True)
    phs_i = transform.resize(np.imag(phs), (p, s), preserve_range=True)
    phs = phs_r + 1j * phs_i
    pos = transform.resize(pos, (p, pos.shape[1]), preserve_range=True)
    # Nothing to do for R_c

    os.mkdir(odir)
    np.save(os.path.join(odir, 'delta_r.npy'), delta_r)
    np.save(os.path.join(odir, 'k_r.npy'), k_r)
    np.save(os.path.join(odir, 'npulses.npy'), npulses)
    np.save(os.path.join(odir, 'nsamples.npy'), nsamples)
    np.save(os.path.join(odir, 'phs.npy'), phs)
    np.save(os.path.join(odir, 'pos.npy'), pos)
    np.save(os.path.join(odir, 'R_c.npy'), R_c)


if __name__ == '__main__':
    def positive(v):
        if v <= 0:
            raise argparse.ArgumentTypeError("Expected value > 0, got %s" % v)
        return v

    def positive_int(v):
        return positive(int(v))

    def positive_float(v):
        return positive(float(v))

    parser = argparse.ArgumentParser()
    parser.add_argument('indir', help='Data input directory')
    parser.add_argument('outdir', help='Data output directory')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-P', '--pulses', type=positive_int,
                       help='The number of pulses')
    group.add_argument('-p', '--scale-pulses', default=1, type=positive_float,
                       help='The scale factor for number of pulses')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-S', '--samples', type=positive_int,
                       help='The number of samples')
    group.add_argument('-s', '--scale-samples', default=1, type=positive_float,
                       help='The scale factor for number of samples')
    args = parser.parse_args()

    resize(args.indir, args.outdir, args.pulses, args.samples,
           scale_p=args.scale_pulses, scale_s=args.scale_samples)
