@echo off
wsl bash -l -c "cd '/mnt/d/PhD/InTheWildRelight/lumigaussSun2/output' && tensorboard --logdir . --load_fast true"