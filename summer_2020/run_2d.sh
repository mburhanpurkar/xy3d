#!/bin/bash

#SBATCH -J full
#SBATCH -n 9                # Number of cores
#SBATCH -t 0-00:85           # Runtime in D-HH:MM
#SBATCH -p shared            # Partition to submit to
#SBATCH --output=full_output
#SBATCH --mail-user=maya.burhanpurkar@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

module purge
module load python/2.7.14-fasrc02
module load gcc/7.1.0-fasrc01 openmpi/2.1.0-fasrc02
source activate ody

mpirun ./full_2d.py
