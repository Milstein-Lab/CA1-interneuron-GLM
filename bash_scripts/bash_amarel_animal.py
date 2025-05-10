
export DATE=$(date +%Y%m%d_%H%M%S)
export LABEL="$1"
export JOB_NAME=optimize_CA1_inter_"$LABEL"_"$DATE"


export NUM_ANIMALS="$2"
export RANKS="$3"

ARRAY_MAX=$(($NUM_ANIMALS - 1))



# Set environment variables for thread limits
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Create necessary directories on the scratch space
mkdir -p /scratch/${USER}/data/CA1-inter
mkdir -p /scratch/${USER}/logs/CA1-inter

# Submit the job to SLURM
sbatch <<EOT
#!/bin/bash
#SBATCH -J $JOB_NAME
#SBATCH -o /scratch/${USER}/logs/CA1-inter/$JOB_NAME.%j.o
#SBATCH -e /scratch/${USER}/logs/CA1-inter/$JOB_NAME.%j.e
#SBATCH -p main
#SBATCH --requeue
#SBATCH --array=0-$ARRAY_MAX
#SBATCH --nodes=1
#SBATCH --ntasks=1       # Request 10 parallel tasks
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1:30:00
#SBATCH --mail-user=mike.finch@rutgers.edu
#SBATCH --mail-type=ALL

set -x
cd $HOME/CA1-interneuron-GLM

# Run the optimization script with the SLURM task ID as the animal index
srun python get_00x.py \$SLURM_ARRAY_TASK_ID $RANKS
EOT
