"""
FFmpeg wrapper for Audora audio processing pipeline
Handles audio analysis, normalization, and format conversion
"""


import logging
import subprocess
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import ffmpeg
import numpy as np
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class AudioFormat(Enum):
    """Supported audio formats"""
    MP3_320 = "mp3_320"
    MP3_256 = "mp3_256" 
    MP3_192 = "mp3_192"
    FLAC_CD = "flac_16_44"
    FLAC_HIRES = "flac_hires"
    WAV = "wav"
    AIFF = "aiff"