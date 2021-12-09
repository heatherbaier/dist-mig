# Jobs

**Single Node Distributed:** job8  
**Multi-Node Distributed:** Steps to to:

**Step1:** Launch an interactive job on EVERY machine using:

```
qsub -I -l walltime=1:00:00 -l nodes=1:ppn=2
```

**Step 2:** Then, in the terminal for EVERY checked out machine, run:

```
source "/usr/local/anaconda3-2021.05/etc/profile.d/conda.csh"
module load anaconda3/2021.05
unsetenv PYTHONPATH
conda activate dhsrl4

torchrun --nnodes=2 --nproc_per_node=2 --rdzv_id=790876 --rdzv_backend=c10d --rdzv_endpoint=128.239.59.1:29500 /sciclone/home20/hmbaier/test_rpc/test.py
```

Repeat steps one and two above in a new terminal (which will reserve a new machine/cluster node)