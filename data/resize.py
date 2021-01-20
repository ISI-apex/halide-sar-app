#!/usr/bin/env python3
"""Resize RITSAR-formatted NumPy data."""

import argparse
import os
import numpy as np
from skimage import transform

__author__ = "Connor Imes"
__date__ = "2021-01-15"


# This currently only handles the fields we require, not all platform data!
def resize(idir, odir, scale_pulses, scale_samples):
    """Resize data from idir and save to odir."""

    delta_r = np.load(os.path.join(idir, 'delta_r.npy'))
    k_r = np.load(os.path.join(idir, 'k_r.npy'))
    npulses = np.load(os.path.join(idir, 'npulses.npy'))
    nsamples = np.load(os.path.join(idir, 'nsamples.npy'))
    phs = np.load(os.path.join(idir, 'phs.npy'))
    pos = np.load(os.path.join(idir, 'pos.npy'))
    R_c = np.load(os.path.join(idir, 'R_c.npy'))

    # delta_r value is a function of freq, so value must be adjusted
    delta_r = delta_r / scale_samples
    # k_r values are a function of freq, so both values and shape must be adjusted
    k_r = transform.resize(k_r * scale_samples,
                           (int(k_r.shape[0] * scale_samples),),
                           preserve_range=True)
    npulses = int(npulses * scale_pulses)
    nsamples = int(nsamples * scale_samples)
    # transform.resize does not work for complex types
    phs_r = transform.resize(np.real(phs),
                             (int(phs.shape[0] * scale_pulses), int(phs.shape[1] * scale_samples)),
                             preserve_range=True)
    phs_i = transform.resize(np.imag(phs),
                             (int(phs.shape[0] * scale_pulses), int(phs.shape[1] * scale_samples)),
                             preserve_range=True)
    phs = phs_r + 1j * phs_i
    pos = transform.resize(pos,
                           (int(pos.shape[0] * scale_pulses), pos.shape[1]),
                           preserve_range=True)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('indir', help='Data input directory')
    parser.add_argument('outdir', help='Data output directory')
    parser.add_argument('-p', '--scale-pulses', default=1, type=float,
                        help='The scale factor for number of pulses')
    parser.add_argument('-s', '--scale-samples', default=1, type=float,
                        help='The scale factor for number of samples')
    args = parser.parse_args()

    resize(args.indir, args.outdir, args.scale_pulses, args.scale_samples)
