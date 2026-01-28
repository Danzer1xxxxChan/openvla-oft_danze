"""Showlab Dataset EE: Robot manipulation demonstrations with end-effector pose actions."""

from typing import Iterator, Tuple, Any

import os
import glob
import numpy as np
import h5py
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from scipy.spatial.transform import Rotation as R

# Set TensorFlow Hub cache directory to use local models
os.environ['TFHUB_CACHE_DIR'] = '/storage/xiaokangliu/models/tfhub_cache'
tfds.core.utils.gcs_utils._is_gcs_disabled = True
os.environ['NO_GCE_CHECK'] = 'true'

def quaternion_to_rotation_6d(quat):
    """Convert quaternion to 6D rotation representation.
    
    Args:
        quat: Quaternion in (x, y, z, w) format
        
    Returns:
        6D rotation representation (first two columns of rotation matrix flattened)
    """
    rotation = R.from_quat(quat)
    rot_matrix = rotation.as_matrix()
    # Take first two columns and flatten
    rot_6d = rot_matrix[:, :2].T.flatten()
    return rot_6d


# Action normalization bounds (computed from dataset statistics with rotvec)
# These values normalize EE pose deltas to [-1, 1]

# 1.2.0 & 1.3.0
# ACTION_MIN = np.array([-0.02053109, -0.02344218, -0.01927894, 
#                        -0.03959246, -0.02551901, -0.05755476], dtype=np.float32)
# ACTION_MAX = np.array([0.02424264, 0.02225738, 0.01838928,
#                        0.02692719, 0.02277117, 0.07725246], dtype=np.float32)

# 1.4.0
ACTION_MIN = np.array([-0.01055485, -0.01778186, -0.01517773, 
                       -0.01637130, -0.01816135, -0.03264310], dtype=np.float32)
ACTION_MAX = np.array([0.02017602, 0.03290551, 0.02640338,
                       0.01911882, 0.01777281, 0.06029939], dtype=np.float32)

class PickNPlaceEE(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Showlab dataset with end-effector pose actions."""

    VERSION = tfds.core.Version('1.4.0')
    RELEASE_NOTES = {
      '1.4.0': 'Pick up only',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
        # /storage/xiaokangliu/models/google-t5/t5-11b/

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Front camera RGB observation.',
                        ),
                        'camera_left_image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Left camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                            doc='Robot state, consists of [3x EE position, 3x EE orientation (roll, pitch, yaw), '
                                '2x gripper positions (2 fingers)].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot action, consists of normalized delta end-effector pose and gripper action. '
                            '[3x position delta (dx, dy, dz), 3x orientation delta (droll, dpitch, dyaw), '
                            '1x gripper action (-1=open, 1=close)]. '
                            'Position deltas normalized to [-1, 1] using dataset min/max values. End-effector control.',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction (dummy for now).'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                    'episode_length': tfds.features.Scalar(
                        dtype=np.int32,
                        doc='Length of the episode in timesteps.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        # You can modify this to split your data into train/val
        # For now, we'll put 80% in train and 20% in val
        # We'll use the first 10 episodes for testing
        # all_files = sorted(glob.glob('/storage/xiaokangliu/data/showlab/data/success/episode_*.hdf5'))
        # all_files = sorted(glob.glob('/storage/xiaokangliu/data/VLAST-Data/empty_empty/episode_*.hdf5'))
        all_files = sorted(glob.glob('/storage/zhijun/real_franka/pick_and_place/episode_*.hdf5'))

        # Randomly pick 3 samples for validation
        import random
        random.seed(42)  # Set seed for reproducibility
        val_files = random.sample(all_files, min(3, len(all_files)))
        train_files = [f for f in all_files if f not in val_files]
        
        return {
            'train': self._generate_examples(file_list=train_files),
            'val': self._generate_examples(file_list=val_files),
        }

    def _generate_examples(self, file_list) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _rotvec_to_euler(rotvec):
            """Convert rotation vector to euler angles (roll, pitch, yaw)."""
            rotation = R.from_rotvec(rotvec)
            return rotation.as_euler('xyz', degrees=False)
        
        def _normalize_action(pos_delta, ori_delta):
            """Normalize position and orientation deltas to [-1, 1] using dataset min/max."""
            # Normalize position (first 3 dims of ACTION_MIN/MAX)
            pos_min = ACTION_MIN[:3]
            pos_max = ACTION_MAX[:3]
            pos_delta_clipped = np.clip(pos_delta, pos_min, pos_max)
            pos_range = pos_max - pos_min
            pos_normalized = 2 * (pos_delta_clipped - pos_min) / pos_range - 1
            
            # Normalize orientation (last 3 dims of ACTION_MIN/MAX)
            ori_min = ACTION_MIN[3:6]
            ori_max = ACTION_MAX[3:6]
            ori_delta_clipped = np.clip(ori_delta, ori_min, ori_max)
            ori_range = ori_max - ori_min
            ori_normalized = 2 * (ori_delta_clipped - ori_min) / ori_range - 1
            
            return pos_normalized.astype(np.float32), ori_normalized.astype(np.float32)

        def _parse_example(episode_path):
            try:
                # Load HDF5 file
                with h5py.File(episode_path, 'r') as f:
                    # Extract data from HDF5
                    gripper_action = f['joint_action'][:]  # Gripper command (0 or 0.08)
                    
                    # Observations
                    camera_front = f['observations/images/camera_front_color'][:]
                    camera_left = f['observations/images/camera_left_color'][:]
                    camera_wrist = f['observations/images/camera_wrist_color'][:]
                    joint_pos = f['observations/joint_pos'][:]
                    joint_vel = f['observations/joint_vel'][:]
                    ee_pose = f['observations/ee_pose'][:]
                    
                    episode_length = len(gripper_action)
                    
                    # Create language instruction (using dummy for now)
                    # TODO: Replace with actual task description if available
                    # language_instruction = 'Place the cube on the purple gear'
                    language_instruction = 'Pick up the grey cup and place it on the green plate'
                    language_embedding = self._embed([language_instruction])[0].numpy()
                    
                    # First pass: identify which timesteps to keep
                    # Threshold for considering motion as redundant
                    POS_DELTA_THRESHOLD = 1e-4
                    
                    # First, identify all redundant transitions
                    is_redundant_transition = []
                    for i in range(episode_length - 1):
                        # Calculate position delta
                        curr_pos = ee_pose[i, :3]
                        next_pos = ee_pose[i + 1, :3]
                        pos_delta = next_pos - curr_pos
                        
                        # Get current and next gripper state
                        curr_gripper = -1.0 if gripper_action[i] >= 0.07 else 1.0
                        next_gripper = -1.0 if gripper_action[i + 1] >= 0.07 else 1.0
                        
                        # Check if this is a redundant transition
                        # Redundant if: position delta is tiny AND gripper doesn't change
                        redundant = np.all(np.abs(pos_delta) < POS_DELTA_THRESHOLD) and curr_gripper == next_gripper
                        is_redundant_transition.append(redundant)
                    
                    # Find consecutive redundant sequences of length >= 2
                    indices_to_remove = set()
                    i = 0
                    while i < len(is_redundant_transition):
                        if is_redundant_transition[i]:
                            # Found start of redundant sequence
                            seq_start = i
                            seq_end = i
                            # Find end of consecutive redundant sequence
                            while seq_end < len(is_redundant_transition) and is_redundant_transition[seq_end]:
                                seq_end += 1
                            
                            seq_length = seq_end - seq_start
                            
                            # If sequence length >= 2, mark middle steps for removal
                            # Keep first step of sequence, remove all middle steps
                            if seq_length >= 2:
                                # Remove indices from seq_start+1 to seq_end (inclusive)
                                for idx in range(seq_start + 1, seq_end + 1):
                                    # Don't remove the last step of the episode
                                    if idx < episode_length - 1:
                                        indices_to_remove.add(idx)
                            
                            i = seq_end
                        else:
                            i += 1
                    
                    # Build list of indices to keep
                    indices_to_keep = [i for i in range(episode_length) if i not in indices_to_remove]
                    
                    # Always ensure first and last steps are included
                    if 0 not in indices_to_keep:
                        indices_to_keep.insert(0, 0)
                    if (episode_length - 1) not in indices_to_keep:
                        indices_to_keep.append(episode_length - 1)
                    
                    indices_to_keep = sorted(set(indices_to_keep))
                    
                    # Second pass: build episode with filtered steps
                    episode = []
                    for idx_in_filtered, i in enumerate(indices_to_keep):
                        # State: [ee_position (3D), ee_orientation_euler (3D), gripper_pos (2D)]
                        ee_pos = ee_pose[i, :3]
                        ee_rotvec = ee_pose[i, 3:6]  # rotation vector (3D)
                        ee_euler = _rotvec_to_euler(ee_rotvec)  # Convert to roll, pitch, yaw
                        gripper_pos = joint_pos[i, 7:9]  # Last 2 dimensions are gripper positions
                        
                        state = np.concatenate([
                            ee_pos,         # 3-dim (x, y, z)
                            ee_euler,       # 3-dim (roll, pitch, yaw)
                            gripper_pos,    # 2-dim (finger 1, finger 2)
                        ], axis=0).astype(np.float32)
                        
                        # Action: delta EE pose (position + euler angles) + gripper
                        # For the last timestep in filtered sequence, use zero delta and current gripper
                        if idx_in_filtered < len(indices_to_keep) - 1:
                            # Get next index in the filtered sequence
                            next_i = indices_to_keep[idx_in_filtered + 1]
                            
                            # Position delta
                            curr_pos = ee_pose[i, :3]
                            next_pos = ee_pose[next_i, :3]
                            pos_delta = next_pos - curr_pos
                            
                            # Orientation delta (euler angles)
                            curr_rotvec = ee_pose[i, 3:6]
                            next_rotvec = ee_pose[next_i, 3:6]
                            curr_euler = _rotvec_to_euler(curr_rotvec)
                            next_euler = _rotvec_to_euler(next_rotvec)
                            ori_delta = next_euler - curr_euler
                            
                            # Handle angle wrapping for orientation
                            ori_delta = np.arctan2(np.sin(ori_delta), np.cos(ori_delta))
                            
                            # branch ---

                            # --- Normalize position and orientation deltas ---
                            pos_delta_normalized, ori_delta_normalized = _normalize_action(pos_delta, ori_delta)
                            
                            # --- no normalization in raw dataset ---
                            # pos_delta_normalized = pos_delta.astype(np.float32)
                            # ori_delta_normalized = ori_delta.astype(np.float32)
                            
                            # branch end ---

                            # Gripper action: -1 for open (>= 0.07), 1 for close (< 0.07)
                            gripper = -1.0 if gripper_action[next_i] >= 0.07 else 1.0
                            
                            # Concatenate: [pos_delta (3), ori_delta (3), gripper (1)]
                            action = np.concatenate([
                                pos_delta_normalized,
                                ori_delta_normalized,
                                [gripper]
                            ], axis=0).astype(np.float32)
                        else:
                            # Last timestep: zero delta and current gripper
                            gripper = -1.0 if gripper_action[i] >= 0.07 else 1.0
                            action = np.concatenate([
                                np.zeros(3),   # position delta
                                np.zeros(3),   # orientation delta
                                [gripper]
                            ], axis=0).astype(np.float32)
                        
                        episode.append({
                            'observation': {
                                'image': camera_front[i],
                                'camera_left_image': camera_left[i],
                                'wrist_image': camera_wrist[i],
                                'state': state,
                            },
                            'action': action,
                            'discount': 1.0,
                            'reward': float(idx_in_filtered == (len(indices_to_keep) - 1)),
                            'is_first': idx_in_filtered == 0,
                            'is_last': idx_in_filtered == (len(indices_to_keep) - 1),
                            'is_terminal': idx_in_filtered == (len(indices_to_keep) - 1),
                            'language_instruction': language_instruction,
                            'language_embedding': language_embedding,
                        })
                    
                    # Create output data sample
                    filtered_length = len(indices_to_keep)
                    sample = {
                        'steps': episode,
                        'episode_metadata': {
                            'file_path': episode_path,
                            'episode_length': filtered_length,
                        }
                    }
                    
                    return episode_path, sample
                    
            except Exception as e:
                print(f"Error processing {episode_path}: {e}")
                return None

        # no filtering version
        def parse_example(episode_path):
            """Parse example without filtering redundant steps - processes entire sequence.
            
            Args:
                episode_path: Path to the HDF5 file containing the episode data.
                
            Returns:
                Tuple of (episode_path, sample) where sample contains all timesteps.
            """
            def _rotvec_to_euler(rotvec):
                """Convert rotation vector to euler angles (roll, pitch, yaw)."""
                rotation = R.from_rotvec(rotvec)
                return rotation.as_euler('xyz', degrees=False)
            
            def _normalize_action(pos_delta, ori_delta):
                """Normalize position and orientation deltas to [-1, 1] using dataset min/max."""
                # Normalize position (first 3 dims of ACTION_MIN/MAX)
                pos_min = ACTION_MIN[:3]
                pos_max = ACTION_MAX[:3]
                pos_delta_clipped = np.clip(pos_delta, pos_min, pos_max)
                pos_range = pos_max - pos_min
                pos_normalized = 2 * (pos_delta_clipped - pos_min) / pos_range - 1
                
                # Normalize orientation (last 3 dims of ACTION_MIN/MAX)
                ori_min = ACTION_MIN[3:6]
                ori_max = ACTION_MAX[3:6]
                ori_delta_clipped = np.clip(ori_delta, ori_min, ori_max)
                ori_range = ori_max - ori_min
                ori_normalized = 2 * (ori_delta_clipped - ori_min) / ori_range - 1
                
                return pos_normalized.astype(np.float32), ori_normalized.astype(np.float32)
            
            try:
                # Load HDF5 file
                with h5py.File(episode_path, 'r') as f:
                    # Extract data from HDF5
                    gripper_action = f['joint_action'][:]  # Gripper command (0 or 0.08)
                    
                    # Observations
                    camera_front = f['observations/images/camera_front_color'][:]
                    camera_left = f['observations/images/camera_left_color'][:]
                    camera_wrist = f['observations/images/camera_wrist_color'][:]
                    joint_pos = f['observations/joint_pos'][:]
                    joint_vel = f['observations/joint_vel'][:]
                    ee_pose = f['observations/ee_pose'][:]
                    
                    episode_length = len(gripper_action)
                    
                    # Create language instruction
                    language_instruction = 'Pick up the test tube with the orange cap and put it in the grey cup.'
                    language_embedding = self._embed([language_instruction])[0].numpy()
                    
                    # Build episode with ALL timesteps (no filtering)
                    episode = []
                    for i in range(episode_length):
                        # State: [ee_position (3D), ee_orientation_euler (3D), gripper_pos (2D)]
                        ee_pos = ee_pose[i, :3]
                        ee_rotvec = ee_pose[i, 3:6]  # rotation vector (3D)
                        ee_euler = _rotvec_to_euler(ee_rotvec)  # Convert to roll, pitch, yaw
                        gripper_pos = joint_pos[i, 7:9]  # Last 2 dimensions are gripper positions
                        
                        state = np.concatenate([
                            ee_pos,         # 3-dim (x, y, z)
                            ee_euler,       # 3-dim (roll, pitch, yaw)
                            gripper_pos,    # 2-dim (finger 1, finger 2)
                        ], axis=0).astype(np.float32)
                        
                        # Action: delta EE pose (position + euler angles) + gripper
                        # For the last timestep, use zero delta and current gripper
                        if i < episode_length - 1:
                            # Position delta
                            curr_pos = ee_pose[i, :3]
                            next_pos = ee_pose[i + 1, :3]
                            pos_delta = next_pos - curr_pos
                            
                            # Orientation delta (euler angles)
                            curr_rotvec = ee_pose[i, 3:6]
                            next_rotvec = ee_pose[i + 1, 3:6]
                            curr_euler = _rotvec_to_euler(curr_rotvec)
                            next_euler = _rotvec_to_euler(next_rotvec)
                            ori_delta = next_euler - curr_euler
                            
                            # Handle angle wrapping for orientation
                            ori_delta = np.arctan2(np.sin(ori_delta), np.cos(ori_delta))
                            
                            # Normalize position and orientation deltas
                            # pos_delta_normalized, ori_delta_normalized = _normalize_action(pos_delta, ori_delta)
                            pos_delta_normalized, ori_delta_normalized = pos_delta, ori_delta # no normalization
                            
                            # Gripper action: -1 for open (>= 0.07), 1 for close (< 0.07)
                            gripper = -1.0 if gripper_action[i + 1] >= 0.07 else 1.0
                            
                            # Concatenate: [pos_delta (3), ori_delta (3), gripper (1)]
                            action = np.concatenate([
                                pos_delta_normalized,
                                ori_delta_normalized,
                                [gripper]
                            ], axis=0).astype(np.float32)
                        else:
                            # Last timestep: zero delta and current gripper
                            gripper = -1.0 if gripper_action[i] >= 0.07 else 1.0
                            action = np.concatenate([
                                np.zeros(3),   # position delta
                                np.zeros(3),   # orientation delta
                                [gripper]
                            ], axis=0).astype(np.float32)
                        
                        episode.append({
                            'observation': {
                                'image': camera_front[i],
                                'camera_left_image': camera_left[i],
                                'wrist_image': camera_wrist[i],
                                'state': state,
                            },
                            'action': action,
                            'discount': 1.0,
                            'reward': float(i == (episode_length - 1)),
                            'is_first': i == 0,
                            'is_last': i == (episode_length - 1),
                            'is_terminal': i == (episode_length - 1),
                            'language_instruction': language_instruction,
                            'language_embedding': language_embedding,
                        })
                    
                    # Create output data sample
                    sample = {
                        'steps': episode,
                        'episode_metadata': {
                            'file_path': episode_path,
                            'episode_length': episode_length,
                        }
                    }
                    
                    return episode_path, sample
                    
            except Exception as e:
                print(f"Error processing {episode_path}: {e}")
                return None

        # Parse all episodes
        for episode_path in file_list:
            # result = _parse_example(episode_path)
            result = parse_example(episode_path)
            if result is not None:
                yield result

        # For large datasets, use beam to parallelize data parsing
        # Uncomment the following lines and comment out the loop above
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(file_list)
        #         | beam.Map(_parse_example)
        #         | beam.Filter(lambda x: x is not None)
        # )
