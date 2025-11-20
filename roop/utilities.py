import glob
import mimetypes
import os
import platform
import shutil
import ssl
import subprocess
import sys
import urllib
import torch
import gradio
import tempfile


import requests
import uuid

from pathlib import Path
from typing import List, Any
from tqdm import tqdm
from scipy.spatial import distance

import roop.globals

TEMP_FILE = 'temp.mp4'
TEMP_DIRECTORY = 'temp'

# monkey patch ssl for mac
if platform.system().lower() == 'darwin':
    ssl._create_default_https_context = ssl._create_unverified_context


def run_ffmpeg(args: List[str]) -> bool:
    commands = ['ffmpeg', '-hide_banner', '-hwaccel', 'auto', '-y', '-loglevel', roop.globals.log_level]
    commands.extend(args)
    print (" ".join(commands))
    try:
        subprocess.check_output(commands, stderr=subprocess.STDOUT)
        return True
    except Exception:
        pass
    return False


def download_file_from_url(url: str, save_dir: str) -> str | None:
    """
    Downloads a file from a given URL and saves it to the specified directory.
    Returns the path to the saved file, or None if download fails.
    """
    if not url or not url.startswith(('http://', 'https://')):
        print(f"Invalid URL: {url}")
        return None
    try:
        response = requests.get(url, stream=True, timeout=30)  # Increased timeout
        response.raise_for_status()  # Raise an exception for bad status codes

        content_disposition = response.headers.get('content-disposition')
        filename = None
        if content_disposition:
            import re
            fname_match = re.findall('filename=["\']?([^"\']+)["\']?', content_disposition)
            if fname_match:
                filename = fname_match[0]

        if not filename:
            parsed_url = urllib.parse.urlparse(url)
            filename = Path(parsed_url.path).name
            if not filename: # if path ends with /, or no clear name
                 filename = str(uuid.uuid4())
        
        # Basic sanitization and ensure an extension
        filename = "".join(c if c.isalnum() or c in ('.', '_', '-') else '_' for c in filename.strip())
        if not Path(filename).suffix:
            content_type = response.headers.get('content-type')
            if content_type:
                import mimetypes
                ext = mimetypes.guess_extension(content_type.split(';')[0].strip())
                if ext:
                    filename += ext
            if not Path(filename).suffix: # Default if still no extension
                 filename += ".download"

        # Ensure save_dir exists
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, filename)
        
        # Ensure unique filename if it already exists
        counter = 1
        original_name_part, original_ext_part = os.path.splitext(filename)
        while os.path.exists(file_path):
            filename = f"{original_name_part}_{counter}{original_ext_part}"
            file_path = os.path.join(save_dir, filename)
            counter += 1
            if counter > 100: # Safety break for too many existing files
                print(f"Could not find a unique filename for {original_name_part} in {save_dir}")
                return None

        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"File downloaded successfully to {file_path}")
        return file_path
    except requests.exceptions.Timeout:
        print(f"Timeout error downloading file from URL {url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file from URL {url}: {e}")
        return None
    except IOError as e:
        print(f"Error saving file {filename if 'filename' in locals() else 'unknown'} to {save_dir}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}")
        return None


def detect_fps(target_path: str) -> float:
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', target_path]
    output = subprocess.check_output(command).decode().strip().split('/')
    try:
        numerator, denominator = map(int, output)
        return numerator / denominator
    except Exception:
        pass
    return 24.0

def cut_video(original_video: str, cut_video: str, start_frame: int, end_frame: int):
    fps = detect_fps(original_video)
    start_time = start_frame / fps
    num_frames = end_frame - start_frame

    run_ffmpeg(['-ss',  str(start_time), '-i', original_video, '-c:v', roop.globals.video_encoder, '-c:a', 'aac', '-frames:v', str(num_frames), cut_video])

def join_videos(videos: List[str], dest_filename: str):
    inputs = []
    filter = ''
    for i,v in enumerate(videos):
        inputs.append('-i')
        inputs.append(v)
        filter += f'[{i}:v:0][{i}:a:0]'
    run_ffmpeg([" ".join(inputs), '-filter_complex', f'"{filter}concat=n={len(videos)}:v=1:a=1[outv][outa]"', '-map', '"[outv]"', '-map', '"[outa]"', dest_filename])    

def extract_frames(target_path: str) -> None:
    create_temp(target_path)
    temp_directory_path = get_temp_directory_path(target_path)
    run_ffmpeg(['-i', target_path, '-q:v', '1', '-pix_fmt', 'rgb24', os.path.join(temp_directory_path, f'%04d.{roop.globals.CFG.output_image_format}')])
    return temp_directory_path


def create_video(target_path: str, dest_filename: str, fps: float = 24.0) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    run_ffmpeg(['-r', str(fps), '-i', os.path.join(temp_directory_path, f'%04d.{roop.globals.CFG.output_image_format}'), '-c:v', roop.globals.video_encoder, '-crf', str(roop.globals.video_quality), '-pix_fmt', 'yuv420p', '-vf', 'colorspace=bt709:iall=bt601-6-625:fast=1', '-y', dest_filename])
    return dest_filename


def create_gif_from_video(video_path: str, gif_path):
    from roop.capturer import get_video_frame

    fps = detect_fps(video_path)
    frame = get_video_frame(video_path)

    run_ffmpeg(['-i', video_path, '-vf', f'fps={fps},scale={frame.shape[0]}:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse', '-loop', '0', gif_path])


def restore_audio(intermediate_video: str, original_video: str, final_video: str) -> None:
    run_ffmpeg(['-i', intermediate_video, '-i', original_video, '-c:v', 'copy', '-map', '0:v:0', '-map', '1:a:0', '-y', final_video])


def get_temp_frame_paths(target_path: str) -> List[str]:
    temp_directory_path = get_temp_directory_path(target_path)
    return glob.glob((os.path.join(glob.escape(temp_directory_path), f'*.{roop.globals.CFG.output_image_format}')))


def get_temp_directory_path(target_path: str) -> str:
    target_name, _ = os.path.splitext(os.path.basename(target_path))
    target_directory_path = os.path.dirname(target_path)
    return os.path.join(target_directory_path, TEMP_DIRECTORY, target_name)


def get_temp_output_path(target_path: str) -> str:
    temp_directory_path = get_temp_directory_path(target_path)
    return os.path.join(temp_directory_path, TEMP_FILE)


def normalize_output_path(source_path: str, target_path: str, output_path: str) -> Any:
    if source_path and target_path:
        source_name, _ = os.path.splitext(os.path.basename(source_path))
        target_name, target_extension = os.path.splitext(os.path.basename(target_path))
        if os.path.isdir(output_path):
            return os.path.join(output_path, source_name + '-' + target_name + target_extension)
    return output_path


def get_destfilename_from_path(srcfilepath: str, destfilepath: str, extension: str) -> str:
    fn, ext = os.path.splitext(os.path.basename(srcfilepath))
    if '.' in extension:
        return os.path.join(destfilepath, f'{fn}{extension}')
    return os.path.join(destfilepath, f'{fn}{extension}{ext}')




def create_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    Path(temp_directory_path).mkdir(parents=True, exist_ok=True)


def move_temp(target_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(target_path)
    if os.path.isfile(temp_output_path):
        if os.path.isfile(output_path):
            os.remove(output_path)
        shutil.move(temp_output_path, output_path)


def clean_temp(target_path: str) -> None:
    temp_directory_path = get_temp_directory_path(target_path)
    parent_directory_path = os.path.dirname(temp_directory_path)
    if not roop.globals.keep_frames and os.path.isdir(temp_directory_path):
        shutil.rmtree(temp_directory_path)
    if os.path.exists(parent_directory_path) and not os.listdir(parent_directory_path):
        os.rmdir(parent_directory_path)

def delete_temp_frames(filename: str) -> None:
    dir = os.path.dirname(os.path.dirname(filename))
    shutil.rmtree(dir)
 



def has_image_extension(image_path: str) -> bool:
    return image_path.lower().endswith(('png', 'jpg', 'jpeg', 'webp'))

def has_extension(filepath: str, extensions: List[str]) -> bool:
    return filepath.lower().endswith(tuple(extensions))


def is_image(image_path: str) -> bool:
    if image_path and os.path.isfile(image_path):
        mimetype, _ = mimetypes.guess_type(image_path)
        return bool(mimetype and mimetype.startswith('image/'))
    return False


def is_video(video_path: str) -> bool:
    if video_path and os.path.isfile(video_path):
        mimetype, _ = mimetypes.guess_type(video_path)
        return bool(mimetype and mimetype.startswith('video/'))
    return False


def encrypt_file(file_path: str) -> None:
    key = roop.globals.get_encryption_key()
    fernet = Fernet(key)
    with open(file_path, 'rb') as file_to_encrypt:
        original = file_to_encrypt.read()
    encrypted = fernet.encrypt(original)
    with open(file_path, 'wb') as encrypted_file:
        encrypted_file.write(encrypted)


def decrypt_file(file_path: str) -> str:
    key = roop.globals.get_encryption_key()
    fernet = Fernet(key)
    with open(file_path, 'rb') as enc_file:
        encrypted = enc_file.read()
    decrypted = fernet.decrypt(encrypted)
    temp_decrypted_path = os.path.join(get_temp_directory_path(file_path), os.path.basename(file_path))
    with open(temp_decrypted_path, 'wb') as dec_file:
        dec_file.write(decrypted)
    return temp_decrypted_path


def conditional_download(download_directory_path: str, urls: List[str]) -> None:
    if not os.path.exists(download_directory_path):
        os.makedirs(download_directory_path)
    for url in urls:
        download_file_path = os.path.join(download_directory_path, os.path.basename(url))
        if not os.path.exists(download_file_path):
            request = urllib.request.urlopen(url)
            total = int(request.headers.get('Content-Length', 0))
            with tqdm(total=total, desc=f'Downloading {url}', unit='B', unit_scale=True, unit_divisor=1024) as progress:
                urllib.request.urlretrieve(url, download_file_path, reporthook=lambda count, block_size, total_size: progress.update(block_size))


def resolve_relative_path(path: str) -> str:
    if hasattr(sys, '_MEIPASS'):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', path))

def get_device() -> str:
    if len(roop.globals.execution_providers) < 1:
        roop.globals.execution_providers = ['CPUExecutionProvider']

    prov = roop.globals.execution_providers[0]
    if 'CUDAExecutionProvider' == prov:
        return 'cuda'
    if 'CoreMLExecutionProvider' == prov:
        return 'mps'
    return 'cpu'
    

# Taken from https://stackoverflow.com/a/68842705
def get_platform():
    if sys.platform == 'linux':
        try:
            proc_version = open('/proc/version').read()
            if 'Microsoft' in proc_version:
                return 'wsl'
        except:
            pass
    return sys.platform

def open_with_default_app(filename):
    if filename == None:
        return
    platform = get_platform()
    if platform == 'darwin':
        subprocess.call(('open', filename))
    elif platform in ['win64', 'win32']:
        os.startfile(filename.replace('/','\\'))
    elif platform == 'wsl':
        subprocess.call('cmd.exe /C start'.split() + [filename])
    else:                                   # linux variants
        subprocess.call('xdg-open', filename)

def prepare_for_batch(target_files):
    print("Preparing temp files")
    tempfolder = os.path.join(tempfile.gettempdir(), "rooptmp")
    if os.path.exists(tempfolder):
        shutil.rmtree(tempfolder)
    Path(tempfolder).mkdir(parents=True, exist_ok=True)
    for f in target_files:
        newname = os.path.basename(f.name)
        shutil.move(f.name, os.path.join(tempfolder, newname))
    return tempfolder


def open_folder(path:str):
    platform = get_platform()
    try:
        if platform == 'darwin':
            subprocess.call(('open', path))
        elif platform in ['win64', 'win32']:
            open_with_default_app(path)
        elif platform == 'wsl':
            subprocess.call('cmd.exe /C start'.split() + [path])
        else:                                   # linux variants
            subprocess.call('xdg-open', path)
    except Exception as e:
        print(e)
        pass
        #import webbrowser
        #webbrowser.open(url)

    

def create_version_html():
    python_version = ".".join([str(x) for x in sys.version_info[0:3]])
    versions_html = f"""
python: <span title="{sys.version}">{python_version}</span>
•
torch: {getattr(torch, '__long_version__',torch.__version__)}
•
gradio: {gradio.__version__}
"""
    return versions_html


def compute_cosine_distance(emb1, emb2):
    return distance.cosine(emb1, emb2)
