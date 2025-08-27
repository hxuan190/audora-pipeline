"""
Simplified dual-version processor that works with your existing structure
"""

import os
import time
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

import ffmpeg

logger = logging.getLogger(__name__)


class DualVersionProcessor:
    """
    Simplified processor for MVP - generates FREE and PREMIUM versions
    Integrates with your existing FFmpegWrapper architecture
    """
    
    # MVP Standards
    TARGET_LUFS = -14.0
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB for MVP
    SUPPORTED_FORMATS = ['mp3', 'wav', 'flac', 'aiff']
    
    def __init__(self):
        self._validate_ffmpeg()
    
    def _validate_ffmpeg(self) -> None:
        """Quick FFmpeg validation"""
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        except Exception as e:
            raise RuntimeError(f"FFmpeg not available: {e}")
    
    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """Simple validation for MVP"""
        try:
            if not os.path.exists(file_path):
                return False, "File not found"
            
            if os.path.getsize(file_path) > self.MAX_FILE_SIZE:
                return False, f"File too large (max {self.MAX_FILE_SIZE // (1024*1024)}MB)"
            
            # Basic FFmpeg probe
            probe = ffmpeg.probe(file_path)
            audio = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
            
            if not audio:
                return False, "No audio stream found"
            
            codec = audio.get('codec_name', 'unknown').lower()
            if codec not in self.SUPPORTED_FORMATS:
                return False, f"Unsupported format: {codec}"
            
            duration = float(audio.get('duration', 0))
            if duration < 10:
                return False, f"Track too short: {duration:.1f}s"
            
            return True, "Valid"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def get_audio_info(self, file_path: str) -> Dict:
        """Extract basic audio info"""
        try:
            probe = ffmpeg.probe(file_path)
            audio = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
            
            return {
                'duration': float(audio.get('duration', 0)),
                'sample_rate': int(audio.get('sample_rate', 44100)),
                'channels': int(audio.get('channels', 2)),
                'codec': audio.get('codec_name', 'unknown'),
                'bitrate': int(audio.get('bit_rate', 0)) if audio.get('bit_rate') else None,
                'file_size': os.path.getsize(file_path)
            }
        except Exception as e:
            logger.error(f"Failed to get audio info: {e}")
            return {}
    
    def process_dual_versions(self, input_file: str, output_dir: str, track_name: str) -> Dict:
        """
        Process into FREE (192kbps) and PREMIUM (320kbps) versions
        
        Returns dict with both file paths and metadata
        """
        import subprocess
        
        try:
            # Step 1: Normalize to temporary WAV file
            normalized_temp = os.path.join(output_dir, f"{track_name}_normalized.wav")
            
            # Normalize using loudnorm filter
            normalize_cmd = [
                'ffmpeg', '-i', input_file,
                '-af', f'loudnorm=I={self.TARGET_LUFS}:TP=-1.0',
                '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2',
                '-y', normalized_temp
            ]
            
            result = subprocess.run(normalize_cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError(f"Normalization failed: {result.stderr}")
            
            # Step 2: Generate FREE version (192kbps)
            free_file = os.path.join(output_dir, f"{track_name}_free.mp3")
            free_cmd = [
                'ffmpeg', '-i', normalized_temp,
                '-acodec', 'libmp3lame', '-b:a', '192k',
                '-ar', '44100', '-ac', '2',
                '-metadata', 'comment=Audora Free - Upgrade for HD quality',
                '-y', free_file
            ]
            
            result = subprocess.run(free_cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                raise RuntimeError(f"Free version generation failed: {result.stderr}")
            
            # Step 3: Generate PREMIUM version (320kbps)
            premium_file = os.path.join(output_dir, f"{track_name}_premium.mp3")
            premium_cmd = [
                'ffmpeg', '-i', normalized_temp,
                '-acodec', 'libmp3lame', '-b:a', '320k',
                '-ar', '44100', '-ac', '2',
                '-metadata', 'comment=Audora Premium - High Definition Audio',
                '-y', premium_file
            ]
            
            result = subprocess.run(premium_cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                raise RuntimeError(f"Premium version generation failed: {result.stderr}")
            
            # Cleanup temp file
            if os.path.exists(normalized_temp):
                os.remove(normalized_temp)
            
            # Verify both files exist
            if not os.path.exists(free_file) or not os.path.exists(premium_file):
                raise RuntimeError("Output files missing after processing")
            
            # Return results
            return {
                'success': True,
                'versions': {
                    'free': {
                        'file_path': free_file,
                        'bitrate': '192kbps',
                        'file_size': os.path.getsize(free_file),
                        'tier': 'free'
                    },
                    'premium': {
                        'file_path': premium_file,
                        'bitrate': '320kbps',
                        'file_size': os.path.getsize(premium_file),
                        'tier': 'premium'
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Dual version processing failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
