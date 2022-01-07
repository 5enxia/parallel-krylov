#!/bin/bash


# ito
#-----------------------------------------------------
# sub-system
#PJM -L "rscunit=ito-b"

# resource group
#PJM -L "rscgrp=ito-g-4-dbg"

# number of virtual nodes
#PJM -L "vnode=1"

# number of cores per virtual node
#PJM -L "vnode-core=36"

# elapsed-time limit
#PJM -L "elapse=3:00"

# Output standard error to the same file that standard output
#PJM -j

# Send Mail
#PJM --mail-list "g212004284@edu.teu.ac.jp"
#PJM -m e
#-----------------------------------------------------


# python
#-----------------------------------------------------
module load python/3.6.2
#-----------------------------------------------------


# module
#-----------------------------------------------------
## mvapich
#module load mvapich/2.3-nocuda-intel18.3
#module load mvapich/gdr-2.3-cuda10.1-gcc4.8.5

## openmpi
#module load openmpi/3.1.3-nocuda-intel18.3
module load openmpi/3.1.3-cuda9.1-intel18.3

#module load cuda/10.1
#-----------------------------------------------------


# date
#-----------------------------------------------------
date
#-----------------------------------------------------


# record info 
#-----------------------------------------------------
module list
#-----------------------------------------------------


# run
# -----------------------------------------------------
python3 test.py
# -----------------------------------------------------

