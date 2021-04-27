# Summit Experiments

The job scripts in this directory use one processor socket in each resource set, where two resource sets are used per node since each node has two processor sockets.
This results in two MPI ranks per node.

All 4 SMT cores are allocated to a task, per Summit's default allocation settings.
Halide uses all available hardware threads.
