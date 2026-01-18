@echo off
wsl bash -l -c "cd '/mnt/d/OneDrive - University of Surrey/InTheWildRelight/lumigaussSun2/output' && tensorboard --logdir . --load_fast true"