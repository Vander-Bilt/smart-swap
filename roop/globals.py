from settings import Settings
from typing import List, Optional
from dataclasses import dataclass
from functools import lru_cache
import gradio as gr
from cryptography.fernet import Fernet

source_path = None
target_path = None
output_path = None
target_folder_path = None

frame_processors: List[str] = []
keep_fps = None
keep_frames = None
skip_audio = None
many_faces = None
reg_notion = True
trigger_vercel = True
use_batch = None
source_face_index = 0
target_face_index = 0
face_position = None
video_encoder = None
video_quality = None
max_memory = None
execution_providers: List[str] = []
execution_threads = None
headless = None
log_level = 'error'
selected_enhancer = None
face_swap_mode = None
blend_ratio = 0.5
distance_threshold = 0.65
default_det_size = True

processing = False 

FACE_ENHANCER = None

INPUT_FACES = []
TARGET_FACES = []

IMAGE_CHAIN_PROCESSOR = None
VIDEO_CHAIN_PROCESSOR = None
BATCH_IMAGE_CHAIN_PROCESSOR = None

CFG: Settings = None

source_face_2_file: gr.File = None
source_face_3_file: gr.File = None

# Key for encryption: set it
encryption_key: bytes = b'0cZ56f7w3ejcXzYA6yC1E2iKi3gOog8ROrT-bRfesG8='

def get_encryption_key() -> bytes:
    global encryption_key
    return encryption_key


@lru_cache(maxsize=None)
def get_face_analyser():
    pass


