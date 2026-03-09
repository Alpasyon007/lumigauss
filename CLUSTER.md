# Running on the AISurrey Compute Cluster

This guide covers deploying lumigaussSun2 training on the University of Surrey AISurrey HPC cluster, which uses **Slurm** for job scheduling and **Apptainer** (formerly Singularity) for containers.

---

## Prerequisites

- SSH access to the submit node: `aisurrey-submit01.surrey.ac.uk`
- Access to the container registry: `container-registry.surrey.ac.uk`
- Your university username (referred to as `YOUR_USERNAME` below)

---

## 1. Build & Push the Docker Image

The cluster runs Apptainer, which can convert Docker images. You first need to push your Docker image to the Surrey container registry.

### Build locally

```bash
docker build -t lumigauss_sun2 .
```

### Tag for the Surrey registry

```bash
docker tag lumigauss_sun2 container-registry.surrey.ac.uk/YOUR_USERNAME/lumigauss_sun2:latest
```

> Replace `YOUR_USERNAME` with your GitLab/registry namespace (e.g., your university username or a shared project namespace).

### Log in & push

```bash
docker login container-registry.surrey.ac.uk
docker push container-registry.surrey.ac.uk/YOUR_USERNAME/lumigauss_sun2:latest
```

Then update the `CONTAINER_IMAGE` variable in `slurm_train.sh` to match your registry path.

---

## 2. Upload Data to the Cluster

Your dataset lives locally at:
```
D:/OneDrive - University of Surrey/Datasets/LumiGaussData/data
```

You need to upload it to the cluster's fast WEKA storage.

### Using rsync (recommended — resumable, shows progress)

From WSL or Git Bash:
```bash
rsync -avP --progress \
  "/mnt/d/OneDrive - University of Surrey/Datasets/LumiGaussData/data/" \
  YOUR_USERNAME@aisurrey-submit01.surrey.ac.uk:/mnt/fast/nobackup/users/YOUR_USERNAME/data/
```

### Using scp

From PowerShell:
```powershell
scp -r "D:\OneDrive - University of Surrey\Datasets\LumiGaussData\data\*" `
  YOUR_USERNAME@aisurrey-submit01.surrey.ac.uk:/mnt/fast/nobackup/users/YOUR_USERNAME/data/
```

### Check data size first

```powershell
Get-ChildItem "D:\OneDrive - University of Surrey\Datasets\LumiGaussData\data" -Recurse |
  Measure-Object -Property Length -Sum |
  Select-Object @{N='Size_GB';E={[math]::Round($_.Sum/1GB,2)}}
```

### Storage options

| Path | Quota | Retention |
|------|-------|-----------|
| `/mnt/fast/nobackup/users/YOUR_USERNAME/` | 200 GB | Persistent |
| `/mnt/fast/nobackup/scratch4weeks/YOUR_USERNAME/` | Large | Auto-deleted after 28 days of no access |

If your data exceeds 200 GB, use scratch storage and update `DATA_DIR` in `slurm_train.sh`.

---

## 3. Upload Code to the Cluster

```bash
rsync -avP --progress \
  --exclude '.git' \
  --exclude 'data' \
  --exclude 'output' \
  --exclude '__pycache__' \
  --exclude '*.pth' \
  "/mnt/d/PhD/InTheWildRelight/lumigaussSun2/" \
  YOUR_USERNAME@aisurrey-submit01.surrey.ac.uk:/mnt/fast/nobackup/users/YOUR_USERNAME/lumigaussSun2/
```

---

## 4. Submit a Training Job

SSH into the submit node and submit the job:

```bash
ssh YOUR_USERNAME@aisurrey-submit01.surrey.ac.uk
cd /mnt/fast/nobackup/users/$USER/lumigaussSun2
sbatch slurm_train.sh
```

---

## 5. Available GPU Partitions

| Partition | GPU | VRAM | Max Walltime | Notes |
|-----------|-----|------|-------------|-------|
| `debug` | A5000 | 24 GB | 4 hours | Quick tests only |
| `2080ti` | RTX 2080 Ti | 11 GB | 3 days | Limited VRAM |
| `3090` | RTX 3090 | 24 GB | 3 days | Good default |
| `a100` | A100 | 80 GB | 3 days | High-end, may have longer queue |
| `l40s` | L40S | 48 GB | 3 days | If available |

Change the partition in `slurm_train.sh` by editing the `#SBATCH --partition=` line.

---

## 6. Monitor Jobs

```bash
# Check job status
squeue -u $USER

# Watch job output in real time
tail -f slurm_JOBID.out

# Cancel a job
scancel JOBID

# Check job details after completion
sacct -j JOBID --format=JobID,JobName,Partition,Elapsed,State,MaxRSS,MaxVMSize

# Check your running/pending jobs
squeue -u $USER -o "%.10i %.20j %.10P %.6D %.4C %.10m %.12l %.8T %.10M"
```

---

## 7. Interactive Jobs (for debugging)

Request an interactive GPU session:

```bash
srun --partition=debug --gpus-per-node=1 --cpus-per-task=4 --mem=32G --time=1:00:00 --pty bash
```

Then run Apptainer manually:

```bash
SIF_FILE="/mnt/fast/nobackup/users/${USER}/containers/lumigauss_sun2.sif"
WORKSPACE="/mnt/fast/nobackup/users/${USER}/lumigaussSun2"
DATA_DIR="/mnt/fast/nobackup/users/${USER}/data"

apptainer exec --nv \
    --bind "${WORKSPACE}:/workspace" \
    --bind "${DATA_DIR}:/workspace/data" \
    --pwd /workspace \
    "${SIF_FILE}" \
    bash
```

This drops you into a shell inside the container where you can run commands interactively.

---

## 8. Download Results

After training completes, download the output back to your local machine.

From WSL or Git Bash:
```bash
rsync -avP --progress \
  YOUR_USERNAME@aisurrey-submit01.surrey.ac.uk:/mnt/fast/nobackup/users/YOUR_USERNAME/lumigaussSun2/output/ \
  "/mnt/d/PhD/InTheWildRelight/lumigaussSun2/output/"
```

From PowerShell:
```powershell
scp -r YOUR_USERNAME@aisurrey-submit01.surrey.ac.uk:/mnt/fast/nobackup/users/YOUR_USERNAME/lumigaussSun2/output/* `
  "D:\PhD\InTheWildRelight\lumigaussSun2\output\"
```

---

## 9. Slurm Script Reference

The main job script is [`slurm_train.sh`](slurm_train.sh). Key variables to configure:

| Variable | Description | Default |
|----------|-------------|---------|
| `CONTAINER_IMAGE` | Docker registry path | `container-registry.surrey.ac.uk/shared-containers/lumigauss_sun2:latest` |
| `SIF_FILE` | Local path for converted .sif file | `/mnt/fast/nobackup/users/$USER/containers/lumigauss_sun2.sif` |
| `WORKSPACE` | Code directory on cluster | `/mnt/fast/nobackup/users/$USER/lumigaussSun2` |
| `DATA_DIR` | Dataset directory on cluster | `/mnt/fast/nobackup/users/$USER/data` |

### Running different scenes

Edit the training command in `slurm_train.sh`. Examples:

**St scene:**
```bash
python ./train.py \
    -s=./data/st_colmap/undistorted \
    --sun_json_path=./data/st_colmap/sun_directions_blender.json \
    -m=./output/st_sun_shadow_pbr_sun_cal_optimize_casts_shadow \
    --use_sun --use_residual_sh --shadow_method shadow_map \
    --shadow_map_resolution=2048 \
    --sky_mask_path=./data/st_colmap/dynamic_masks/sky \
    --use_sun_cal --sky_sh_degree=3 --optimize_casts_shadow
```

**LK2 scene:**
```bash
python ./train.py \
    -s=./data/lk2_colmap/undistorted \
    --sun_json_path=./data/lk2_colmap/sun_directions_blender.json \
    -m=./output/lk2_sun_shadow_pbr_sun_cal \
    --use_sun --use_residual_sh --shadow_method shadow_map \
    --shadow_map_resolution=8192 \
    --sky_mask_path=./data/lk2_colmap/dynamic_masks/sky \
    --full_pbr --use_sun_cal
```

**LWP scene:**
```bash
python ./train.py \
    -s=./data/lwp_colmap/undistorted \
    --sun_json_path=./data/lwp_colmap/sun_directions_blender.json \
    -m=./output/lwp_sun_shadow_pbr_sun_cal \
    --use_sun --use_residual_sh --shadow_method shadow_map \
    --shadow_map_resolution=8192 \
    --sky_mask_path=./data/lwp_colmap/dynamic_masks/sky \
    --full_pbr --use_sun_cal
```

---

## 10. Troubleshooting

| Issue | Solution |
|-------|----------|
| `apptainer pull` fails with auth error | Run `apptainer remote login --username YOUR_USERNAME docker://container-registry.surrey.ac.uk` first |
| Out of GPU memory | Switch to `a100` partition (80 GB), or reduce `--shadow_map_resolution` |
| Job killed at walltime | Add checkpointing; use `--checkpoint_iterations` to save periodically |
| `--constraint=fs_weka` causes pending | Remove the constraint (some nodes use NFS instead, which is slower) |
| Permission denied on `/mnt/fast/nobackup/users/` | Create your directory first: `mkdir -p /mnt/fast/nobackup/users/$USER` |
| Container .sif file corrupted | Delete it and let it re-pull: `rm $SIF_FILE` then resubmit |
