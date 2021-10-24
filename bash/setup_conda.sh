#!/bin/bash


# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "The 'conda' command could not be found. Exiting..."
    exit
fi


# Configure env
read -rp "Enter conda environment name: " env_name
python_version="3.8"
cuda_version="10.2"


# Create env
conda create -y -n "$env_name" python="$python_version"


# Install pytorch + cuda
if [ "$cuda_version" == "none" ]; then
    conda install -n "$env_name" -y pytorch torchvision torchaudio cpuonly -c pytorch
elif [ "$cuda_version" == "10.2" ]; then
    conda install -n "$env_name" pytorch torchvision torchaudio cudatoolkit=$cuda_version -c pytorch
elif [ "$cuda_version" == "11.1" ]; then
    conda install -n "$env_name" pytorch torchvision torchaudio cudatoolkit=$cuda_version -c pytorch -c nvidia
else
    echo "Incorrect cuda version. Exiting..."
    exit
fi

conda install pytorch cudatoolkit=10.2 -c pytorch
conda install -c anaconda swig
