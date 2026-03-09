#!/bin/bash
#SBATCH --job-name=lumigauss              # Job name shown in squeue
#SBATCH --partition=3090                  # GPU partition (3090=24GB, a100=80GB, 2080ti=11GB)
#SBATCH --gpus-per-node=1                # Number of GPUs
#SBATCH --cpus-per-task=8                # CPU cores
#SBATCH --mem=64G                        # RAM
#SBATCH --time=3-00:00:00               # Max walltime (3 days)
#SBATCH --output=slurm_%j.out           # stdout log (%j = job ID)
#SBATCH --error=slurm_%j.err            # stderr log
#SBATCH --constraint=fs_weka            # Use fast WEKA storage nodes

# ============================================================
# PATHS — adjust these to match your cluster layout
# ============================================================
CONTAINER_IMAGE="container-registry.surrey.ac.uk/shared-containers/lumigauss_sun2:latest"
SIF_FILE="/mnt/fast/nobackup/users/${USER}/containers/lumigauss_sun2.sif"
WORKSPACE="/mnt/fast/nobackup/users/${USER}/lumigaussSun2"
DATA_DIR="/mnt/fast/nobackup/users/${USER}/data"

# ============================================================
# Pull/convert container image (only if .sif doesn't exist)
# ============================================================
mkdir -p "$(dirname ${SIF_FILE})"
if [ ! -f "${SIF_FILE}" ]; then
    echo "Converting Docker image to Apptainer .sif ..."
    apptainer pull "${SIF_FILE}" "docker://${CONTAINER_IMAGE}"
fi

# ============================================================
# Run training inside the container
# ============================================================
# --nv            : enable NVIDIA GPU support
# --bind          : mount host directories into container
# --pwd           : set working directory inside container
#
# Bind mounts:
#   $WORKSPACE  -> /workspace        (your code)
#   $DATA_DIR   -> /workspace/data   (your datasets)

apptainer exec --nv \
    --bind "${WORKSPACE}:/workspace" \
    --bind "${DATA_DIR}:/workspace/data" \
    --pwd /workspace \
    "${SIF_FILE}" \
    python ./train.py \
        -s=./data/st_colmap/undistorted \
        --sun_json_path=./data/st_colmap/sun_directions_blender.json \
        -m=./output/st_sun_shadow_pbr_sun_cal_optimize_casts_shadow \
        --use_sun \
        --use_residual_sh \
        --shadow_method shadow_map \
        --shadow_map_resolution=2048 \
        --sky_mask_path=./data/st_colmap/dynamic_masks/sky \
        --use_sun_cal \
        --sky_sh_degree=3 \
        --optimize_casts_shadow
