# Summit Experiments

This directory contains launch scripts for evaluating the application on Summit at OLCF.


## Prerequisites

The `sarbp` application needs to be compiled and available on `$PATH`.
However, nodes sometimes fail to find the application on `$PATH`, so it's actually best to just modify the scripts to point directly (use the full path) to the `sarbp` application you want to use.

If you used different `gcc` and `cuda` module versions, modify the job scripts appropriately.

The input data directories must be available in project directory at `${PROJWORK}/csc436/SAR-data/`.
If they are not available, use the resize script in this project's `data` directory to produce the required input directories with the correct naming convention.
On Summit, you may need to use the Anaconda python to create an environment to run the resize script (the more traditional venv failed to compile all package dependencies):

```sh
module load python/3.7.0-anaconda3-5.3.0
conda create -p env-conda
conda init bash
. ~/.bashrc
conda activate env-conda
conda install numpy
conda install scikit-image
# Now you can use the python resize script
conda deactivate
```

For larger data sizes, the login nodes may kill the resize script.
In this case, run like above in an interactive job session, e.g.:

```sh
bsub -P CSC436 -q debug -Is -W 2:00 -nnodes 1 bash
```


## Executing

To submit a job, use `bsub`, e.g.:

```sh
export PATH=/path/to/halide-sar-app/build:$PATH
bsub sar-AFRL-4096x4096-1.lsf
```

Monitor queued and running jobs with `bjobs`.

Output images will be written to your user's directory at `${MEMBERWORK}/csc436/`.


## Additional Resources

[Summit User Guide](https://docs.olcf.ornl.gov/systems/summit_user_guide.html)
