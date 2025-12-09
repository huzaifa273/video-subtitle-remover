import warnings
from enum import Enum, unique

warnings.filterwarnings('ignore')
import os
import torch
import logging
import platform
import stat
from fsplit.filesplit import Filesplit
import paddle

# ×××××××××××××××××××× [Do Not Change] start ××××××××××××××××××××
paddle.disable_signal_handler()
logging.disable(logging.DEBUG) 
logging.disable(logging.WARNING) 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LAMA_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'big-lama')
STTN_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sttn', 'infer_model.pth')
MODEL_VERSION = 'V4'
DET_MODEL_BASE = os.path.join(BASE_DIR, 'models')
DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det')

if 'big-lama.pt' not in (os.listdir(LAMA_MODEL_PATH)):
    fs = Filesplit()
    fs.merge(input_dir=LAMA_MODEL_PATH)

if 'inference.pdiparams' not in os.listdir(DET_MODEL_PATH):
    fs = Filesplit()
    fs.merge(input_dir=DET_MODEL_PATH)

sys_str = platform.system()
if sys_str == "Windows":
    ffmpeg_bin = os.path.join('win_x64', 'ffmpeg.exe')
elif sys_str == "Linux":
    ffmpeg_bin = os.path.join('linux_x64', 'ffmpeg')
else:
    ffmpeg_bin = os.path.join('macos', 'ffmpeg')
FFMPEG_PATH = os.path.join(BASE_DIR, '', 'ffmpeg', ffmpeg_bin)

if 'ffmpeg.exe' not in os.listdir(os.path.join(BASE_DIR, '', 'ffmpeg', 'win_x64')):
    fs = Filesplit()
    fs.merge(input_dir=os.path.join(BASE_DIR, '', 'ffmpeg', 'win_x64'))
os.chmod(FFMPEG_PATH, stat.S_IRWXU + stat.S_IRWXG + stat.S_IRWXO)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# ×××××××××××××××××××× [Do Not Change] end ××××××××××××××××××××


@unique
class InpaintMode(Enum):
    STTN = 'sttn'
    LAMA = 'lama'


# ×××××××××××××××××××× [RTX 3090 EXTREME CONFIG] start ××××××××××××××××××××

MODE = InpaintMode.STTN

# === 1. MASKING STRATEGY ===
# We will use this to aggressively "eat" the subtitle and surrounding compression artifacts.
# This value acts as the base padding.
SUBTITLE_AREA_DEVIATION_PIXEL = 50 

# Thresholds for detection logic
THRESHOLD_HEIGHT_WIDTH_DIFFERENCE = 10
THRESHOLD_HEIGHT_DIFFERENCE = 20
PIXEL_TOLERANCE_Y = 20
PIXEL_TOLERANCE_X = 20

# === 2. STTN ENGINE TUNING ===
STTN_SKIP_DETECTION = True

# NEIGHBOR_STRIDE: How "far" apart to look for clean pixels.
# Increased to 30. It will scan frames much further away to find a background 
# that doesn't have the text.
STTN_NEIGHBOR_STRIDE = 30

# REFERENCE_LENGTH: How many reference frames to hold in the attention mechanism.
# Increased to 30 for higher temporal stability.
STTN_REFERENCE_LENGTH = 30

# MAX_LOAD_NUM: The size of the "Chunk" loaded into VRAM.
# 200 frames @ 30fps is ~6.6 seconds of context.
# Your 24GB VRAM *should* handle this. If it Crashes, lower to 150.
STTN_MAX_LOAD_NUM = 200

# Safety clamp
if STTN_MAX_LOAD_NUM < max(STTN_NEIGHBOR_STRIDE, STTN_REFERENCE_LENGTH):
    STTN_MAX_LOAD_NUM = max(STTN_NEIGHBOR_STRIDE, STTN_REFERENCE_LENGTH)

# === LAMA SETTINGS ===
LAMA_SUPER_FAST = False
# ×××××××××××××××××××× [RTX 3090 EXTREME CONFIG] end ××××××××××××××××××××