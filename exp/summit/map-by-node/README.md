# Summit Experiments

The job scripts in this directory use an entire node in each resource set, where each node has two processor sockets.
This results in one MPI rank per node.

All 4 SMT cores are allocated to a task, per Summit's default allocation settings, but Halide threads are limited to one thread per available physical CPU.
This is similar to if the job scripts had specified `#BSUB -alloc_flags smt1`, but since we don't, threads aren't necessarily bound to the physical CPU.
