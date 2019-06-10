#!/bin/bash

#SBATCH -J mpitest
#SBATCH -n 64                # Number of cores
#SBATCH -t 0-00:50           # Runtime in D-HH:MM
#SBATCH -p general           # Partition to submit to
#SBATCH --output=mpitest.out
#SBATCH --mail-type=END      # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=burhanpurkar@college.harvard.edu  # Email to which notifications will be sent

module purge
module load gcc/7.1.0-fasrc01
module load openmpi/2.1.0-fasrc02

mpicxx testmod3d.cpp -o parallel -lm
srun -n 64 --mpi=pmi2 ./parallel
