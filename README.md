

'''bash

# upload code to Unity
rsync -rtlp <DIR_NAME> unity:~

# SSH into Unity
ssh unity

# submit job
sbatch path/train.sbatch

# check queue
squeue -u <username>

# cancel job
scancel <job-id>