#!/bin/bash

set -e

# make output directories
mkdir -p vis vis_tnl

# dbg suffix for binaries
#dbg=""
dbg="-dbg"

# problem size: N=n^2
n=100

# disable OpenMP for reproducibility
export OMP_NUM_THREADS=1

make_glvis() {
    local dir="$1"
    local sol="$dir/ex5.sol"

    echo "FiniteElementSpace" > "$sol"
    echo "FiniteElementCollection: H1_2D_P1" >> "$sol"
    echo "VDim: 1" >> "$sol"
    echo "Ordering: 0" >> "$sol"
    echo "" >> "$sol"
    find "$dir" -name "ex5.sol.??????" | sort | xargs cat >> "$sol"
}

verify() {
    make_glvis vis
    make_glvis vis_tnl
    diff "vis/ex5.sol" "vis_tnl/ex5.sol"
    rm -f vis/ex5.sol* vis_tnl/ex5.sol*
}

# BoomerAMG
mpirun hypre-ex5$dbg -vis -n $n -solver 0
mpirun tnl-hypre-ex5$dbg -vis -n $n -solver 0
verify

# PCG
mpirun hypre-ex5$dbg -vis -n $n -solver 50
mpirun tnl-hypre-ex5$dbg -vis -n $n -solver 50
verify

# PCG with AMG
mpirun hypre-ex5$dbg -vis -n $n -solver 1
mpirun tnl-hypre-ex5$dbg -vis -n $n -solver 1
verify

# PCG with ParaSails
mpirun hypre-ex5$dbg -vis -n $n -solver 8
mpirun tnl-hypre-ex5$dbg -vis -n $n -solver 8
verify

# FlexGMRES with AMG
mpirun hypre-ex5$dbg -vis -n $n -solver 61
mpirun tnl-hypre-ex5$dbg -vis -n $n -solver 61
verify
