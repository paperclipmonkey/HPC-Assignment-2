#!/bin/bash --login
module purge
module load compiler/intel
module load mpi/intel

mpiicc mpi.c -o mpi