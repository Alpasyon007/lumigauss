#FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

#ENV DEBIAN_FRONTEND=noninteractive
#ENV TORCH_CUDA_ARCH_LIST="3.5 5.0 6.0 6.1 7.0 7.5 8.0 8.6"

## Install basic dependencies
#RUN set -xe \
#	&& apt-get update -y \
#	&& apt-get install -y curl gnupg2 wget git

## Install Python 3.10
#RUN apt-get install software-properties-common -y \
#	&& add-apt-repository ppa:deadsnakes/ppa -y \
#	&& apt-get install -y python3.10 \
#	&& apt-get install -y python3.10-dev \
#	&& apt-get install -y python3.10-tk

#RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.10 get-pip.py

## Create workspace
#WORKDIR /workspace
#COPY ./submodules ./submodules
#COPY ./requirements.txt ./requirements.txt

## Install Gaussian Splatting Requirements
#RUN python3.10 -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116
#RUN python3.10 -m pip install submodules/diff-gaussian-rasterization submodules/simple-knn

FROM nvcr.io/nvidia/pytorch:25.01-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="5.0 6.0 6.1 7.0 7.5 8.0 8.6 9.0 12.0"

# Install basic dependencies
RUN set -xe \
	&& apt-get update -y \
	&& apt-get install -y curl gnupg2 wget unzip git dos2unix libx11-6 libglib2.0-0 libxrender1 libxext6 libgl1 libgomp1

# Create workspace
WORKDIR /workspace
COPY ./requirements.txt ./requirements.txt

# Install Gaussian Splatting Requirements
RUN python -m pip install -r requirements.txt

COPY ./submodules ./submodules
RUN python -m pip install submodules/diff-surfel-rasterization submodules/simple-knn

# Copy all necessary files
COPY . /workspace