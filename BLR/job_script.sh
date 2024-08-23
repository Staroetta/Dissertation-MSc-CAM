#!/bin/sh
# Grid Engine options (lines prefixed with #$)
#$ -N BLR_T=5             
#$ -cwd                  
#$ -l h_rt=4:00:00 
#$ -l h_vmem=6G
#$ -pe interactivemem 16
#  These options are:
#  job name: -N
#  use the current working directory: -cwd
#  runtime limit: -l h_rt
#  memory limit: -l h_vmem
#  parallel environment: -pe interactivemem

# Initialise the environment modules
. /etc/profile.d/modules.sh

# Load Julia
module load roslin/julia/1.9.0

# Run the program
julia -p 16 BLR.jl

