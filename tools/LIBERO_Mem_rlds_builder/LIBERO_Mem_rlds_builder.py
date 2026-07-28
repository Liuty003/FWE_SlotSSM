from typing import Iterator, Tuple, Any

import os
import h5py
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
import json
from .conversion_utils import MultiThreadedDatasetBuilder

def convert2texts(reasoning, image_size=256):
    revised_reasoning = []
    for datum in reasoning:
        datum_reasoning = []
        # datum is a dictionary
        for key, values in datum.items():
            seg_ind, bbox, itrn_cnt = values
            bbox = (np.array(bbox)*image_size).astype(np.int32)
            re_key_values = key+':'+str(seg_ind)+','+str(bbox)+','+str(itrn_cnt)
            datum_reasoning.append(re_key_values)
        datum_reasoning = '#'.join(datum_reasoning)
        revised_reasoning.append(datum_reasoning)
    return revised_reasoning

def _generate_examples(paths, split, ratio=1.0) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for list of data paths."""
    # the line below needs to be *inside* generate_examples so that each worker creates it's own model
    # creating one shared model outside this function would cause a deadlock

    def _parse_example(episode_path, episode_meta_data, demo_id):
        # load raw data
        with h5py.File(episode_path, "r") as F:
            if f"demo_{demo_id}" not in F['data'].keys():
                return None # skip episode if the demo doesn't exist (e.g. due to failed demo)
            actions = F['data'][f"demo_{demo_id}"]["actions"][()]
            states = F['data'][f"demo_{demo_id}"]["obs"]["ee_states"][()]
            gripper_states = F['data'][f"demo_{demo_id}"]["obs"]["gripper_states"][()]
            joint_states = F['data'][f"demo_{demo_id}"]["obs"]["joint_states"][()]
            images = F['data'][f"demo_{demo_id}"]["obs"]["agentview_rgb"][()]
            wrist_images = F['data'][f"demo_{demo_id}"]["obs"]["eye_in_hand_rgb"][()]
            depth_images = F['data'][f"demo_{demo_id}"]["obs"]["agentview_depth"][()]
            depth_wrist_images = F['data'][f"demo_{demo_id}"]["obs"]["eye_in_hand_depth"][()]
            seg_images = F['data'][f"demo_{demo_id}"]["obs"]["agentview_seg"][()]
            seg_wrist_images = F['data'][f"demo_{demo_id}"]["obs"]["eye_in_hand_seg"][()]

        object_data = episode_meta_data[f"demo_{demo_id}"]
        image_reasonings = object_data['exo_boxes']
        image_reasonings = convert2texts(image_reasonings)
        wrist_image_reasonings = object_data['ego_boxes']
        wrist_image_reasonings = convert2texts(wrist_image_reasonings)
        command_nouns = object_data['task_nouns']
        command = object_data['task_description']


        command_nouns = '. '.join(command_nouns)

        # assemble episode --> here we're assuming demos so we set reward to 1 at the end
        episode = []
        for i in range(actions.shape[0]):
            episode.append({
                'observation': {
                    'image': images[i][:,:],
                    'wrist_image': wrist_images[i][:,:],
                    'image_depth': depth_images[i][:,:][..., np.newaxis],
                    'wrist_image_depth': depth_wrist_images[i][:,:][..., np.newaxis],
                    'image_seg': seg_images[i][:,:][..., np.newaxis], 
                    'wrist_image_seg': seg_wrist_images[i][:,:][..., np.newaxis],
                    'image_reasoning': image_reasonings[i], # object 1:@segid,@bbox,@interaction#object 2:@segid,@bbox,@interaction  
                    'wrist_image_reasoning': wrist_image_reasonings[i],

                    'state': np.asarray(np.concatenate((states[i], gripper_states[i]), axis=-1), np.float32),
                    'joint_state': np.asarray(joint_states[i], dtype=np.float32),
                },
                'action': np.asarray(actions[i], dtype=np.float32),
                'discount': 1.0,
                'reward': float(i == (actions.shape[0] - 1)),
                'is_first': i == 0,
                'is_last': i == (actions.shape[0] - 1),
                'is_terminal': i == (actions.shape[0] - 1),
                'language_instruction': command,
                'language_instruction_nouns': command_nouns,
            })

        # create output data sample
        sample = {
            'steps': episode,
            'episode_metadata': {
                'file_path': episode_path
            }
        }

        # if you want to skip an example for whatever reason, simply return None
        return episode_path + f"_{demo_id}", sample

    # get the meta data
    meta_path = os.path.join(os.path.dirname(paths[0]), 'metainfo.json')
    with open(meta_path, 'r') as file:
        meta_data = json.load(file)

    # for smallish datasets, use single-thread parsing
    for sample in paths:
        with h5py.File(sample, "r") as F:
            sample_demo_ids = [key.replace('demo_', '') for key in F['data'].keys()]
    
        task_name = os.path.basename(sample).split('/')[-1][:-10]
        if task_name not in meta_data:
            continue
        task_meta_data = meta_data[task_name]
        task_meta_demo_ids = [key.replace('demo_', '') for key in task_meta_data.keys()]
        # Keep conversion deterministic.  The original set-to-list conversion
        # produced a different episode order on different Python processes.
        demo_ids = sorted(
            set(sample_demo_ids) & set(task_meta_demo_ids),
            key=lambda value: int(value),
        )
        n_demos = len(demo_ids)

        idx = 0
        tv_splitpoint = int(ratio * n_demos)
        # train_data += tv_splitpoint
        # val_data += n_demos - tv_splitpoint
        # print('Train size', train_data, '--- Val size', val_data)
        
        while idx < n_demos:
            ret = _parse_example(sample, task_meta_data, demo_ids[idx])
            assert(ret is not None)
            idx += 1
            if (split == 'train') and (idx > tv_splitpoint):
                continue
            elif (split == 'val') and (idx <= tv_splitpoint):
                continue
            yield ret


class LIBEROMem(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    # Four workers balance conversion speed and disk pressure for the ten
    # large task files. The conversion helper filters empty worker chunks, so
    # the same builder still works for a one-task smoke dataset.
    N_WORKERS = 4
    MAX_PATHS_IN_MEMORY = 60   # number of paths converted & stored in memory before writing to disk
                               # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                               # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples      # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(256, 256, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Wrist camera RGB observation.',
                        ),

                        # depths
                        'image_depth': tfds.features.Image(
                            shape=(256, 256, 1),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera depth observation.',
                        ),
                        'wrist_image_depth': tfds.features.Image(
                            shape=(256, 256, 1),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Wrist camera depth observation.',
                        ),

                        # depths
                        'image_seg': tfds.features.Image(
                            shape=(256, 256, 1),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera segmentation observation.',
                        ),
                        'wrist_image_seg': tfds.features.Image(
                            shape=(256, 256, 1),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera segmentation observation.',
                        ),

                        # object-centric bboxes and seg indices
                        'image_reasoning': tfds.features.Text(
                            doc='scene objects as dictionary of bbox and seg index in Main camera.'
                        ),
                        'wrist_image_reasoning': tfds.features.Text(
                            doc='scene objects as dictionary of bbox and seg index in Wrist camera.'
                        ),

                        'state': tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                            doc='Robot EEF state (6D pose, 2D gripper).',
                        ),
                        'joint_state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot joint angles.',
                        )

                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Robot EEF action.',
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
                        doc='Language Instruction.'
                    ),
                    'language_instruction_nouns': tfds.features.Text(
                        doc='Language Instruction Nouns.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_paths(self):
        """Define filepaths for data splits."""
        # Do not depend on the process working directory.  This is especially
        # important for `tfds build`, which may stage the builder elsewhere.
        data_dir = os.environ.get("LIBERO_MEM_DATA_DIR", "./LIBERO-Mem")
        train_files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
        if not train_files:
            raise FileNotFoundError(
                f"No LIBERO-Mem HDF5 files found under {data_dir!r}. "
                "Set LIBERO_MEM_DATA_DIR to the downloaded dataset directory."
            )
        print(f"Building LIBERO-Mem RLDS from {len(train_files)} task files in {data_dir}")
        return {
            "train": train_files,
        }

# 961
