"""
Quick OpenVLA-OFT single-step inference experiment.
This script performs a single inference step with a given image and end effector pose.

CRITICAL - Proprioception Format:
=================================
For the 'pick_n_place_ee' dataset, the model expects 8-dimensional END-EFFECTOR POSE:
  - Dimensions 0-5: EEF_state (end-effector Cartesian position and orientation)
    - [0-2]: x, y, z position (meters)
    - [3-5]: roll, pitch, yaw orientation (radians, Euler angles)
  - Dimensions 6-7: gripper_state (2D gripper positions)

This is NOT joint angles!
- If you have joint state (7 joint angles + gripper), you MUST first convert it to 
  end-effector Cartesian pose using forward kinematics for your robot.
- The raw dataset may contain joint angles, but the libero_dataset_transform extracts
  the EEF pose from state[:, :6], meaning the raw state is expected to have EEF data.
"""

import os
import sys
import numpy as np
import torch
from PIL import Image
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from experiments.robot.openvla_utils import (
    get_vla,
    get_processor,
    get_action_head,
    get_proprio_projector,
    prepare_images_for_vla,
    normalize_proprio,
    DEVICE,
    OPENVLA_IMAGE_SIZE,
)


class SimpleConfig:
    """Simple configuration object for inference."""
    def __init__(self):
        # Model checkpoint - change this to your desired checkpoint
        # self.pretrained_checkpoint = "/users/xiaokangliu/robotics/openvla-oft-xk/ckpt_log/openvla-7b+pick_n_place_ee+b16+lr-0.0005+lora-r32+dropout-0.0--nol1_parallel_dec--24_acts_chunk--l1_reg--3rd_person_img--no_aug--20000_chkpt"
        self.pretrained_checkpoint = "/users/xiaokangliu/robotics/openvla-oft-xk/ckpt_log_tube/openvla-7b+pick_n_place_ee+b16+lr-5e-05+lora-r32+dropout-0.1--image_aug--tube_noq99_nonorm_nol1_parallel_dec--24_acts_chunk--l1_reg--3rd_person_img--no_aug--6000_chkpt"
        
        # Model settings
        self.model_family = "openvla"
        self.load_in_8bit = False
        self.load_in_4bit = False
        self.lora_rank = 32
        
        # Input settings
        self.num_images_in_input = 1
        self.use_proprio = False  # Set to True if you want to use proprioception
        # NOTE: For 'pick_n_place_ee', proprio should be 8D:
        #   - EEF_state: 6D (x, y, z, roll, pitch, yaw)
        #   - gripper_state: 2D (two gripper position values)
        self.center_crop = False
        self.showlab_franka_arm = True
        
        # Action head settings
        self.use_l1_regression = False  # Using L1 regression action head
        self.use_diffusion = False
        self.use_film = False
        
        # Dataset/unnormalization key - this should match your training data
        self.unnorm_key = "pick_n_place_ee"  # Adjust if needed
        

def main():
    """Run a single inference step."""
    
    print("=" * 80)
    print("OpenVLA-OFT Quick Inference")
    print("=" * 80)
    
    # Initialize configuration
    cfg = SimpleConfig()
    
    # Input data
    image_path = "/users/xiaokangliu/robotics/openvla-oft-xk/episode_0.png"
    
    # ============================================================================
    # IMPORTANT: Proprioception Input Format
    # ============================================================================
    # The 'pick_n_place_ee' model expects 8D proprioception:
    #   - EEF_state: 6D (x, y, z, roll, pitch, yaw) - END-EFFECTOR CARTESIAN POSE
    #   - gripper_state: 2D (two gripper values)
    #
    # If you have JOINT STATE (7 joint angles + gripper), you MUST convert to
    # end-effector pose using forward kinematics before passing to the model!
    #
    # The dataset transform (libero_dataset_transform) expects the raw state to
    # already contain EEF pose in the first 6 dimensions.
    # ============================================================================
    
    # Option 1: You have END-EFFECTOR POSE (what the model expects)
    # End effector pose: [x, y, z, roll, pitch, yaw, gripper]
    end_effector_pose_7d = np.array([
        0.42615190090293326,
        -0.012967961119098183,
        0.39999344322374614,
        -3.101586492971949,
        0.0011933248705754817,
        -0.8015327060140238,
        0.0
    ], dtype=np.float32)
    
    # Convert to 8D format expected by the model (duplicate gripper value)
    end_effector_pose = np.concatenate([
        end_effector_pose_7d[:6],  # EEF_state (x, y, z, roll, pitch, yaw)
        end_effector_pose_7d[6:7], # gripper_state[0]
        end_effector_pose_7d[6:7], # gripper_state[1] (duplicated for 2D gripper)
    ])
    
    # Option 2: If you have JOINT STATE instead (uncomment and modify):
    # joint_state_8d = np.array([
    #     0.08104195445775986,      # joint 1
    #     -0.42638757824897766,     # joint 2
    #     -0.026782847940921783,    # joint 3
    #     -2.6759588718414307,      # joint 4
    #     -0.01819315366446972,     # joint 5
    #     2.251084566116333,        # joint 6
    #     0.8725829124450684,       # joint 7
    #     0.0                        # gripper
    # ], dtype=np.float32)
    # 
    # # YOU MUST CONVERT JOINTS TO END-EFFECTOR POSE HERE!
    # # Use forward kinematics with your robot's URDF/kinematic model
    # # from your_robot_kinematics import joint_to_ee_pose
    # # ee_pose_6d = joint_to_ee_pose(joint_state_8d[:7])  # Returns [x,y,z,r,p,y]
    # # end_effector_pose = np.concatenate([ee_pose_6d, joint_state_8d[7:8], joint_state_8d[7:8]])
    # raise NotImplementedError("Convert joint state to EEF pose using forward kinematics!")
    
    # Task description
    # task_label = "Pick up the grey cup and place it on the green plate."
    task_label = "Pick up the test tube with the orange cap and put it in the grey cup."
    
    print(f"\n[1/5] Loading image from: {image_path}")
    # Load and prepare the image
    image_pil = Image.open(image_path).convert("RGB")
    print(f"  Original image size: {image_pil.size}")
    
    # Resize to OpenVLA expected size (224x224)
    if image_pil.size != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE):
        print(f"  Resizing image to {OPENVLA_IMAGE_SIZE}x{OPENVLA_IMAGE_SIZE}...")
        image_pil = image_pil.resize((OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE), Image.Resampling.BILINEAR)
    
    image_np = np.array(image_pil)
    print(f"  Final image shape: {image_np.shape}, dtype: {image_np.dtype}")
    
    print(f"\n[2/5] Loading VLA model from: {cfg.pretrained_checkpoint}")
    # Load the model
    vla = get_vla(cfg)
    print(f"  Model loaded successfully on device: {DEVICE}")
    
    print(f"\n[3/5] Loading processor...")
    # Load the processor
    processor = get_processor(cfg)
    print(f"  Processor loaded successfully")
    
    # Load action head if using L1 regression
    action_head = None
    if cfg.use_l1_regression:
        print(f"\n[4/5] Loading L1 regression action head...")
        llm_dim = vla.llm_dim
        action_head = get_action_head(cfg, llm_dim)
        print(f"  Action head loaded successfully")
    else:
        print(f"\n[4/5] Skipping action head (using default VLA output)")
    
    # Load proprioception projector if needed
    proprio_projector = None
    if cfg.use_proprio:
        print(f"\n  Loading proprioception projector...")
        llm_dim = vla.llm_dim
        proprio_dim = end_effector_pose.shape[0]
        proprio_projector = get_proprio_projector(cfg, llm_dim, proprio_dim)
        print(f"  Proprio projector loaded successfully")
    
    print(f"\n[5/5] Running inference...")
    # Prepare the observation dictionary
    obs = {
        "full_image": image_np,
        "state": end_effector_pose
    }
    
    # Run inference
    with torch.inference_mode():
        # Process the image
        processed_images = prepare_images_for_vla([obs["full_image"]], cfg)
        primary_image = processed_images[0]
        
        # Build the prompt
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
        # prompt = task_label
        print(f"  Prompt: {prompt}")
        
        # Process inputs
        inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)
        print(f"  Input processed, pixel_values shape: {inputs['pixel_values'].shape}")
        
        # Prepare proprioception if used
        proprio = None
        if cfg.use_proprio:
            proprio = obs["state"]
            proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
            proprio = normalize_proprio(proprio, proprio_norm_stats)
            print(f"  Normalized proprio: {proprio}")
        
        # Predict action
        if action_head is None:
            # Standard VLA output
            action, _ = vla.predict_action(
                **inputs,
                unnorm_key=cfg.unnorm_key,
                do_sample=False,
                proprio=proprio,
                proprio_projector=proprio_projector,
                use_film=cfg.use_film
            )
        else:
            # With action head
            action, _ = vla.predict_action(
                **inputs,
                unnorm_key=cfg.unnorm_key,
                do_sample=False,
                proprio=proprio,
                proprio_projector=proprio_projector,
                noisy_action_projector=None,
                action_head=action_head,
                use_film=cfg.use_film,
            )
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nInput:")
    print(f"  Image: {image_path}")
    print(f"  End effector pose (8D): {end_effector_pose}")
    print(f"    - EEF position (x,y,z): [{end_effector_pose[0]:.4f}, {end_effector_pose[1]:.4f}, {end_effector_pose[2]:.4f}]")
    print(f"    - EEF orientation (r,p,y): [{end_effector_pose[3]:.4f}, {end_effector_pose[4]:.4f}, {end_effector_pose[5]:.4f}]")
    print(f"    - Gripper state (2D): [{end_effector_pose[6]:.4f}, {end_effector_pose[7]:.4f}]")
    print(f"  Task: {task_label}")
    print(f"\nPredicted Action:")
    print(f"  Shape: {action.shape}")
    print(f"  Values: {action}")
    print(f"\nAction breakdown (assuming 7-DoF):")
    if action.shape[-1] >= 7:
        print(f"  Position (x, y, z): [{action[0]:.4f}, {action[1]:.4f}, {action[2]:.4f}]")
        print(f"  Rotation (rx, ry, rz): [{action[3]:.4f}, {action[4]:.4f}, {action[5]:.4f}]")
        print(f"  Gripper: {action[6]:.4f}")
        if action.shape[-1] > 7:
            print(f"  Additional dims: {action[7:]}")
    else:
        print(f"  Full action: {action}")
    
    print("\n" + "=" * 80)
    print("Inference completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
