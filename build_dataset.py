"""Script to build the dataset and save to a specified directory."""

import tensorflow_datasets as tfds
from showlab_dataset_builder import PickNPlaceEE

# Build the dataset
data_dir = '/storage/danze/VLA/openvla/pick_and_place_orange_block_01_27/'

builder = PickNPlaceEE(data_dir=data_dir)
builder.download_and_prepare()

print(f"Dataset built successfully at: {data_dir}")
print(f"Dataset info:\n{builder.info}")
