# echo "alias conda='/mnt/bn/vgfm2/test_dit/zechen/customized_conda_env/condabin/conda'" >> ~/.bashrc

# Create and activate conda environment
#conda create -n openvla-oft python=3.10 -y
#conda activate openvla-oft

cd ../openvla-oft/
pip install -e .

# Install Flash Attention 2 for training (https://github.com/Dao-AILab/flash-attention)
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Verify Ninja --> should return exit code "0"

pip3 install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121

pip install "flash-attn==2.5.5" --no-build-isolation

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt
pip install transformers@git+https://github.com/moojink/transformers-openvla-oft.git

pip install numpy==1.26.4 peft==0.15.0 # PyOpenGL==3.1.1a1
conda install -c conda-forge -y libegl-devel
sudo apt install -y libosmesa6 libosmesa6-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev libglew-dev    # work together with export MUJOCO_GL="osmesa"

# for training
pip install peft==0.11.1 diffusers==0.33.0