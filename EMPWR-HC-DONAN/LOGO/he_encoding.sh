#!/bin/bash

# Function to print usage
usage() {
    echo "Usage: $0 --wsi_path <path> --save_dir <dir> --tile_size <size> --resize_factor <factor> --tile_path <path> --encoding_save_dir <dir>"
    exit 1
}

# Parse input arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --wsi_path) wsi_path="$2"; shift ;;
        --save_dir) save_dir="$2"; shift ;;
        --tile_size) tile_size="$2"; shift ;;
        --resize_factor) resize_factor="$2"; shift ;;
        --tile_path) tile_path="$2"; shift ;;
        --encoding_save_dir) encoding_save_dir="$2"; shift ;;
        *) usage ;; # unknown option
    esac
    shift
done

# Check if all required arguments are provided
if [ -z "$wsi_path" ] || [ -z "$save_dir" ] || [ -z "$tile_size" ] || [ -z "$resize_factor" ] || [ -z "$tile_path" ] || [ -z "$encoding_save_dir" ]; then
    echo "Missing arguments!"
    usage
fi

# Check if environment.yml exists and create conda environment from it
if [ -f "environment.yml" ]; then
    echo "Creating conda environment from environment.yml..."
    conda env create -f environment_vips.yml
fi

# Activate the environment
conda activate vips

# Run tiling.py with arguments
python -u tiling.py --wsi_path "$wsi_path" --save_dir "$save_dir" --tile_size "$tile_size" --resize_factor "$resize_factor"

# Deactivate the environment
conda deactivate

# Clone the repository
git clone https://github.com/prov-gigapath/prov-gigapath

# Navigate into the repository
cd prov-gigapath

# Create the conda environment from environment.yaml
conda env create -f environment.yaml

# Activate the gigapath environment
conda activate gigapath

# Install the package in editable mode
pip install -e .

# Run encoding.py with arguments
python -u encoding.py --tile_path "$tile_path" --save_dir "$encoding_save_dir"

# Deactivate the environment
conda deactivate