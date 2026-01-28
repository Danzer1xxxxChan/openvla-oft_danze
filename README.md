# finetuning OpenVLA-OFT in ShowLab

## 🛠 Installation

First, set up a specialized conda environment for the project.

### 1. Create and Activate Environment

```bash
# Create and activate conda environment
conda create -n openvla-oft python=3.10 -y
conda activate openvla-oft

```

### 2. Install Core Dependencies

```bash
# Install PyTorch (Choose the command specific to your machine: https://pytorch.org/get-started/locally/)
pip3 install torch torchvision torchaudio

# Clone the repo and install dependencies
git clone https://github.com/Danzer1xxxxChan/openvla-oft_danze.git
cd openvla-oft_danze
pip install -e .

```

### 3. Install Flash Attention 2

This is required for efficient training.

```bash
# If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"
pip install "flash-attn==2.5.5" --no-build-isolation

# If the above fails, try installing from the specific wheel:
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

```

---

## 🚀 Quick Start

### Libero Setup

Clone and install the **LIBERO** repository and its required packages:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt  # Run from openvla-oft base dir

# If ModuleNotFoundError: No module named 'libero' occurs, export the path:
export PYTHONPATH=$PYTHONPATH:./LIBERO

```

### Run Inference

Use the following Python script to run inference using a pretrained OpenVLA-OFT checkpoint:

```python
import pickle
from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM

# Instantiate config
cfg = GenerateConfig(
    pretrained_checkpoint = "moojink/openvla-7b-oft-finetuned-libero-spatial",
    use_l1_regression = True,
    use_diffusion = False,
    use_film = False,
    num_images_in_input = 2,
    use_proprio = True,
    load_in_8bit = False,
    load_in_4bit = False,
    center_crop = True,
    num_open_loop_steps = NUM_ACTIONS_CHUNK,
    unnorm_key = "libero_spatial_no_noops",
)

# Load OpenVLA-OFT policy and processor
vla = get_vla(cfg)
processor = get_processor(cfg)
action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

# Load sample observation
with open("experiments/robot/libero/sample_libero_spatial_observation.pkl", "rb") as file:
    observation = pickle.load(file)

# Generate robot action chunk
actions = get_vla_action(cfg, vla, processor, observation, observation["task_description"], action_head, proprio_projector)

print("Generated action chunk:")
for act in actions:
    print(act)

```

---

## 📊 Build Dataset

### 1. Dedicated Environment for Data Generation

Isolate data building from training to avoid dependency conflicts.

```bash
conda create -n vla-data-gen python=3.10 -y
conda activate vla-data-gen

# Install base packages for TFDS, HDF5, and Image processing
pip install tensorflow tensorflow-datasets tensorflow-hub h5py pillow scipy numpy

# Verify Installation
python -c "import tensorflow as tf; import tensorflow_datasets as tfds; import tensorflow_hub as hub; print('Environment Ready')"

```

### 2. Convert Data (HDF5 to RLDS)

Each episode takes approximately **2 minutes** to process.

```bash
python showlab_dataset_builder.py \
  --input_path "/scratch/Projects/CFP-01/CFP01-CF-033/danze/VLA/data/orange_block_hdf5/pick_and_place/episode_*.hdf5" \
  --data_dir "/scratch/Projects/CFP-01/CFP01-CF-033/danze/VLA/data/orange_block_rlds/" \
  --tfhub_cache "/scratch/Projects/CFP-01/CFP01-CF-033/danze/VLA/models/tfhub_cache/"

```

* `--input_path`: Wildcard path to HDF5 files (use double quotes).
* `--data_dir`: Target directory for RLDS output.
* `--tfhub_cache`: Local cache for the Universal Sentence Encoder model.

### 3. Submit via HPC Queue (PBS Script)

File: `openvla_oft.sh`

```bash
#!/bin/bash
#PBS -N openvla_oft
#PBS -l select=1:ngpus=1:ncpus=12
#PBS -l walltime=05:00:00
#PBS -j oe
#PBS -k oed
#PBS -o /scratch/Projects/CFP-01/CFP01-CF-033/danze/projects_log/openvla_oft.txt
#PBS -P CFP01-CF-033
#PBS -q auto

echo 'job start...'
nvidia-smi

source /hpctmp/e1546981/miniconda/bin/activate vla-data-gen
export CUDA_VISIBLE_DEVICES=0
cd /scratch/Projects/CFP-01/CFP01-CF-033/danze/VLA/openvla-oft-xk

python showlab_dataset_builder.py \
  --input_path "/scratch/Projects/CFP-01/CFP01-CF-033/danze/VLA/data/orange_block_hdf5/pick_and_place/episode_*.hdf5" \
  --data_dir "/scratch/Projects/CFP-01/CFP01-CF-033/danze/VLA/data/orange_block_rlds/" \
  --tfhub_cache "/scratch/Projects/CFP-01/CFP01-CF-033/danze/VLA/models/tfhub_cache/"

echo "Task Complete"

```

Submit with: `qsub openvla_oft.sh`

---

## 🏋️ Training

### Fine-tuning Script

Use `torchrun` for distributed training.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_P2P_DISABLE=1
export NCCL_P2P_LEVEL=2
export NCCL_IB_TIMEOUT=22
export TORCH_NCCL_BLOCKING_WAIT=0

# Force Python to use local prismatic code
export PYTHONPATH=/scratch/Projects/CFP-01/CFP01-CF-033/danze/VLA/openvla-oft-xk:$PYTHONPATH

torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/finetune_real_world.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /scratch/Projects/CFP-01/CFP01-CF-033/danze/VLA/data/orange_block_rlds/ \
  --dataset_name pick_n_place_ee \
  --run_root_dir /scratch/Projects/CFP-01/CFP01-CF-033/danze/VLA/models/ckpt_log_orange_block_01_28 \
  --use_l1_regression False \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio False \
  --batch_size 16 \
  --learning_rate 5e-5 \
  --num_steps_before_decay 10000 \
  --max_steps 10005 \
  --save_freq 1000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_project "OpenVLA-OFT-Orange-Block" \
  --merge_lora_during_training True \
  --run_id_note orange_block_noq99_nonorm_nol1_parallel_dec--24_acts_chunk--l1_reg--3rd_person_img--no_aug \
  --shuffle_buffer_size 10000

```

> **Note:** Remember to turn off `center_crop` during evaluation!
