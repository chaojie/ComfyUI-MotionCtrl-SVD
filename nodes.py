import argparse
import os
import tempfile

import numpy as np
import torch
from glob import glob
from torchvision.transforms import CenterCrop, Compose, Resize

import sys
current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_path)

from .gradio_utils.camera_utils import CAMERA_MOTION_MODE, create_relative

from .gradio_utils.utils import vis_camera
from .gradio_utils.motionctrl_cmcm_gradio import build_model, motionctrl_sample

import json
import folder_paths
from PIL import Image

os.environ['KMP_DUPLICATE_LIB_OK']='True'

MOTION_CAMERA_OPTIONS = ["U", "D", "L", "R", "O", "O_0.2x", "O_0.4x", "O_1.0x", "O_2.0x", "O_0.2x", "O_0.2x", "Round-RI", "Round-RI_90", "Round-RI-120", "Round-ZoomIn", "SPIN-ACW-60", "SPIN-CW-60", "I", "I_0.2x", "I_0.4x", "I_1.0x", "I_2.0x", "1424acd0007d40b5", "d971457c81bca597", "018f7907401f2fef", "088b93f15ca8745d", "b133a504fc90a2d1"]

def process_camera(camera_pose_str,frame_length):
    RT=json.loads(camera_pose_str)
    for i in range(frame_length):
        if len(RT)<=i:
            RT.append(RT[len(RT)-1])
    
    if len(RT) > frame_length:
        RT = RT[:frame_length]
    
    RT = np.array(RT).reshape(-1, 3, 4)
    return RT

class LoadMotionCtrlSVDCameraPreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "motion_camera": (MOTION_CAMERA_OPTIONS,),
            }
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("POINTS",)
    FUNCTION = "load_motion_camera_preset"
    CATEGORY = "motionctrl"
    
    def load_motion_camera_preset(self, motion_camera):
        data="[]"
        comfy_path = os.path.dirname(folder_paths.__file__)
        with open(f'{comfy_path}/custom_nodes/ComfyUI-MotionCtrl-SVD/examples/camera_poses/test_camera_{motion_camera}.json') as f:
            data = f.read()
        
        return (data,)
      
    
# MODE = ["camera motion control", "object motion control", "camera + object motion control"]
MODE = ["control camera poses", "control object trajectory", "control both camera and object motion"]
RESIZE_MODE = ['Center Crop To 576x1024', 'Keep original spatial ratio']
DIY_MODE = ['Customized Mode 1: First A then B', 
            'Customized Mode 2: Both A and B', 
            'Customized Mode 3: RAW Camera Poses']

## load default model
num_frames = 14
num_steps = 25
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device {device}")



traj_list = [] 
camera_dict = {
                "motion":[],
                "mode": "Customized Mode 1: First A then B",  # "First A then B", "Both A and B", "Custom"
                "speed": 1.0,
                "complex": None
                }  

def process_input_image(input_image, resize_mode):
    width, height = 1024, 576 
    if resize_mode == RESIZE_MODE[0]:
        height = 576
        width = 1024
        w, h = input_image.size
        h_ratio = h / height
        w_ratio = w / width

        if h_ratio > w_ratio:
            h = int(h / w_ratio)
            if h < height:
                h = height
            input_image = Resize((h, width))(input_image)
            
        else:
            w = int(w / h_ratio)
            if w < width:
                w = width
            input_image = Resize((height, w))(input_image)

        transformer = Compose([
            # Resize(width),
            CenterCrop((height, width)),
        ])

        input_image = transformer(input_image)
    else:
        w, h = input_image.size
        if h > w:
            height = 576
            width = int(w * height / h)
        else:
            width = 1024
            height = int(h * width / w)

        input_image = Resize((height, width))(input_image)
        # print(f'input_image size: {input_image.size}')
    return input_image

class MotionctrlSVDLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"default": "motionctrl_svd.ckpt"}),
                "frame_length": ("INT", {"default": 14}),
                "steps": ("INT", {"default": 25}),
            }
        }
        
    RETURN_TYPES = ("MOTIONCTRLSVD",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_checkpoint"
    CATEGORY = "motionctrl"

    def load_checkpoint(self, ckpt_name, frame_length, steps):
        global device

        comfy_path = os.path.dirname(folder_paths.__file__)
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)

        config_path = os.path.join(comfy_path, 'custom_nodes/ComfyUI-MotionCtrl-SVD/configs/inference/config_motionctrl_cmcm.yaml')
        
        if not os.path.exists(ckpt_path):
            os.system(f'wget https://huggingface.co/TencentARC/MotionCtrl/resolve/main/motionctrl_svd.ckpt?download=true -P .')
            os.system(f'mv motionctrl_svd.ckpt?download=true {ckpt_path}')
        model = build_model(config_path, ckpt_path, device, frame_length, steps)

        return (model,)

class MotionctrlSVDSampleSimple:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MOTIONCTRLSVD",),
                "camera": ("STRING", {"multiline": True, "default":"[[1,0,0,0,0,1,0,0,0,0,1,0.2]]"}),
                "image": ("IMAGE",),
                "resize_mode" :(RESIZE_MODE,),
                "seed": ("INT", {"default": 1234}),
                "fps_id": ("INT", {"default": 6, "min": 5, "max": 30}),
                "frame_length": ("INT", {"default": 14}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run_inference"
    CATEGORY = "motionctrl"

    def run_inference(self,model,camera,image,resize_mode,seed,fps_id,frame_length):
        global device
        RT = process_camera(camera,frame_length).reshape(-1,12)
        image = 255.0 * image[0].cpu().numpy()
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        image=process_input_image(image,resize_mode)
        return motionctrl_sample(
            model=model,
            image=image,
            RT=RT,
            num_frames=frame_length,
            fps_id=fps_id,
            decoding_t=1,
            seed=seed,
            sample_num=1,
            device=device
        )


NODE_CLASS_MAPPINGS = {
    "Motionctrl-SVD Sample Simple":MotionctrlSVDSampleSimple,
    "Load Motionctrl-SVD Camera Preset":LoadMotionCtrlSVDCameraPreset,
    "Load Motionctrl-SVD Checkpoint": MotionctrlSVDLoader,
}