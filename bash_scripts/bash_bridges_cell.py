#!/bin/bash

export DATE=$(date +%Y%m%d_%H%M%S)
export LABEL="$1"
export JOB_NAME=optimize_CA1_inter_"$LABEL"_"$DATE"

export ANIMAL_ID="$2"
export RANKS="$3"
export NUM_CELLS="$4"
LOOP_MAX=$((NUM_CELLS - 1))

# Thread limits
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Log dir
export SCRATCH_DIR="/ocean/projects/bio240068p/mfinch/CA1-inter/logs"
mkdir -p ${SCRATCH_DIR}/logs/CA1-inter

# Submit job
sbatch <<EOT
#!/bin/bash
#SBATCH -J $JOB_NAME
#SBATCH -o ${SCRATCH_DIR}/logs/CA1-inter/$JOB_NAME.%j.out
#SBATCH -e ${SCRATCH_DIR}/logs/CA1-inter/$JOB_NAME.%j.err
#SBATCH -p RM
#SBATCH -A bio240068p
#SBATCH --ntasks=20
#SBATCH --time=3:00:00
#SBATCH --mail-user=mike.finch@rutgers.edu
#SBATCH --mail-type=ALL

cd \$HOME/CA1-interneuron-GLM
set -x

for i in \$(seq 0 $LOOP_MAX); do
    srun --exclusive -N1 -n1 python cell_mse_x00_positive_min_max.py \$i $ANIMAL_ID $RANKS \
     > ${SCRATCH_DIR}/logs/CA1-inter/cell_\$i.out 2>&1 &


done

wait
EOT
