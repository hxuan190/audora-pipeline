# app/audio/ffmpeg_wrapper.py
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


class ProcessingIntensity(Enum):
    """Processing intensity levels"""
    CONSERVATIVE = "conservative"  # -16 LUFS, 95% DR preservation
    STANDARD = "standard"         # -14 LUFS, 90% DR preservation  
    AGGRESSIVE = "aggressive"     # -12 LUFS, 85% DR preservation


@dataclass
class AudioSpecs:
    """Audio file specifications"""
    sample_rate: int
    bit_depth: int
    channels: int
    duration: float
    bitrate: Optional[int]
    codec: str
    file_size: int
    
    
@dataclass
class QualityMetrics:
    """Audio quality measurements"""
    lufs: float
    peak_db: float
    dynamic_range: float
    thd_plus_n: float
    noise_floor: float
    quality_score: int
    
    
class ProcessingResult(BaseModel):
    """Result of audio processing operation"""
    success: bool
    input_file: str
    output_files: Dict[str, str] = Field(default_factory=dict)
    quality_metrics: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    mastered_for_audora: bool = False
    quality_score: int = 0
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class FFmpegWrapper:
    """Core FFmpeg wrapper for Audora audio processing"""
    
    # Audora quality standards
    QUALITY_STANDARDS = {
        'lufs_target': -14.0,
        'lufs_tolerance': 0.1,
        'peak_limit': -1.0,
        'dynamic_range_min': 6.0,
        'thd_max': 0.05,
        'noise_floor_max': -60.0
    }
    
    # Processing profiles
    PROCESSING_PROFILES = {
        ProcessingIntensity.CONSERVATIVE: {
            'lufs_target': -16.0,
            'dr_preservation': 0.95,
            'compression_ratio': 2.0,
            'eq_boost': 0.5
        },
        ProcessingIntensity.STANDARD: {
            'lufs_target': -14.0,
            'dr_preservation': 0.90,
            'compression_ratio': 3.0,
            'eq_boost': 1.0
        },
        ProcessingIntensity.AGGRESSIVE: {
            'lufs_target': -12.0,
            'dr_preservation': 0.85,
            'compression_ratio': 4.0,
            'eq_boost': 1.5
        }
    }
    
    def __init__(self, temp_dir: Optional[str] = None):
        """Initialize FFmpeg wrapper"""
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self._validate_ffmpeg_installation()
        
    def _validate_ffmpeg_installation(self) -> None:
        """Verify FFmpeg is installed and accessible"""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode != 0:
                raise RuntimeError("FFmpeg not found or not working")
                
            logger.info("FFmpeg validation successful")
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise RuntimeError(f"FFmpeg validation failed: {e}")
    
    def analyze_audio_file(self, file_path: str) -> Tuple[AudioSpecs, QualityMetrics]:
        """
        Comprehensive audio file analysis
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of AudioSpecs and QualityMetrics
        """
        logger.info(f"Analyzing audio file: {file_path}")
        
        try:
            # Get basic file info
            probe = ffmpeg.probe(file_path)
            audio_stream = next(
                (stream for stream in probe['streams'] if stream['codec_type'] == 'audio'),
                None
            )
            
            if not audio_stream:
                raise ValueError("No audio stream found in file")
            
            # Extract basic specifications
            specs = AudioSpecs(
                sample_rate=int(audio_stream.get('sample_rate', 0)),
                bit_depth=self._get_bit_depth(audio_stream),
                channels=int(audio_stream.get('channels', 0)),
                duration=float(audio_stream.get('duration', 0)),
                bitrate=int(audio_stream.get('bit_rate', 0)) if audio_stream.get('bit_rate') else None,
                codec=audio_stream.get('codec_name', 'unknown'),
                file_size=os.path.getsize(file_path)
            )
            
            # Perform quality analysis
            quality_metrics = self._analyze_audio_quality(file_path, specs)
            
            logger.info(f"Analysis complete - Quality score: {quality_metrics.quality_score}")
            return specs, quality_metrics
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            raise
    
    def _get_bit_depth(self, stream: Dict) -> int:
        """Extract bit depth from audio stream info"""
        sample_fmt = stream.get('sample_fmt', '')
        
        # Map FFmpeg sample formats to bit depths
        bit_depth_map = {
            'u8': 8, 'u8p': 8,
            's16': 16, 's16p': 16,
            's32': 32, 's32p': 32,
            'flt': 32, 'fltp': 32,
            'dbl': 64, 'dblp': 64,
            's24': 24  # Less common
        }
        
        return bit_depth_map.get(sample_fmt, 16)  # Default to 16-bit
    
    def _analyze_audio_quality(self, file_path: str, specs: AudioSpecs) -> QualityMetrics:
        """
        Perform detailed audio quality analysis
        
        Args:
            file_path: Path to audio file
            specs: Basic audio specifications
            
        Returns:
            QualityMetrics object
        """
        # Analyze loudness (LUFS) using EBU R128
        lufs = self._measure_lufs(file_path)
        
        # Measure peak levels
        peak_db = self._measure_peak_level(file_path)
        
        # Calculate dynamic range
        dynamic_range = self._measure_dynamic_range(file_path)
        
        # Estimate THD+N (simplified)
        thd_plus_n = self._estimate_thd_plus_n(file_path)
        
        # Measure noise floor
        noise_floor = self._measure_noise_floor(file_path)
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            specs, lufs, peak_db, dynamic_range, thd_plus_n, noise_floor
        )
        
        return QualityMetrics(
            lufs=lufs,
            peak_db=peak_db,
            dynamic_range=dynamic_range,
            thd_plus_n=thd_plus_n,
            noise_floor=noise_floor,
            quality_score=quality_score
        )
    
    def _measure_lufs(self, file_path: str) -> float:
        """Measure integrated loudness using EBU R128"""
        try:
            cmd = [
                'ffmpeg', '-i', file_path,
                '-af', 'ebur128=metadata=1',
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            # Parse LUFS from stderr output
            for line in result.stderr.split('\n'):
                if 'I:' in line and 'LUFS' in line:
                    # Extract LUFS value (format: "I: -XX.X LUFS")
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'I:':
                            return float(parts[i + 1])
                            
            logger.warning("Could not extract LUFS measurement")
            return -23.0  # Default fallback
            
        except Exception as e:
            logger.error(f"LUFS measurement failed: {e}")
            return -23.0
    
    def _measure_peak_level(self, file_path: str) -> float:
        """Measure peak audio level"""
        try:
            cmd = [
                'ffmpeg', '-i', file_path,
                '-af', 'astats=metadata=1:reset=1',
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            # Parse peak level from output
            for line in result.stderr.split('\n'):
                if 'Peak level dB:' in line:
                    return float(line.split(':')[1].strip())
                    
            return 0.0  # Default if not found
            
        except Exception as e:
            logger.error(f"Peak level measurement failed: {e}")
            return 0.0
    
    def _measure_dynamic_range(self, file_path: str) -> float:
        """Calculate dynamic range (simplified DR measurement)"""
        try:
            # Use astats to get RMS and peak measurements
            cmd = [
                'ffmpeg', '-i', file_path,
                '-af', 'astats=metadata=1',
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            # This is a simplified DR calculation
            # Real DR measurement requires more sophisticated analysis
            return 12.0  # Placeholder - implement proper DR measurement
            
        except Exception as e:
            logger.error(f"Dynamic range measurement failed: {e}")
            return 8.0
    
    def _estimate_thd_plus_n(self, file_path: str) -> float:
        """Estimate THD+N (simplified)"""
        # Simplified estimation - real THD+N requires signal generation and analysis
        return 0.02  # Placeholder - implement proper THD+N measurement
    
    def _measure_noise_floor(self, file_path: str) -> float:
        """Measure noise floor level"""
        try:
            # Analyze quiet sections to estimate noise floor
            cmd = [
                'ffmpeg', '-i', file_path,
                '-af', 'astats=metadata=1',
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            # Parse minimum RMS level as noise floor estimate
            return -70.0  # Placeholder - implement proper noise floor measurement
            
        except Exception as e:
            logger.error(f"Noise floor measurement failed: {e}")
            return -60.0
    
    def _calculate_quality_score(
        self, 
        specs: AudioSpecs, 
        lufs: float, 
        peak_db: float, 
        dynamic_range: float,
        thd_plus_n: float, 
        noise_floor: float
    ) -> int:
        """
        Calculate Audora Quality Rating (AQR) 0-100
        
        Based on format, dynamic range, and technical quality
        """
        score = 0
        
        # Format scoring (40 points max)
        if specs.codec in ['flac', 'wav', 'aiff']:
            if specs.bit_depth >= 24:
                score += 40
            elif specs.bit_depth >= 16:
                score += 35
        elif specs.codec == 'mp3':
            if specs.bitrate and specs.bitrate >= 320000:
                score += 25
            elif specs.bitrate and specs.bitrate >= 256000:
                score += 20
            elif specs.bitrate and specs.bitrate >= 192000:
                score += 15
        
        # Sample rate scoring (10 points max)
        if specs.sample_rate >= 192000:
            score += 10
        elif specs.sample_rate >= 96000:
            score += 8
        elif specs.sample_rate >= 48000:
            score += 6
        elif specs.sample_rate >= 44100:
            score += 5
        
        # Dynamic range scoring (30 points max)
        if dynamic_range >= 20:
            score += 30
        elif dynamic_range >= 15:
            score += 25
        elif dynamic_range >= 10:
            score += 20
        elif dynamic_range >= 6:
            score += 15
        
        # Technical quality scoring (20 points max)
        if thd_plus_n < 0.01:
            score += 7
        elif thd_plus_n < 0.05:
            score += 5
        
        if peak_db < -1.0:
            score += 7
        elif peak_db < 0.0:
            score += 5
        
        if noise_floor < -60:
            score += 6
        elif noise_floor < -50:
            score += 3
        
        return min(score, 100)
    
    def normalize_audio(
        self, 
        input_file: str, 
        output_file: str,
        intensity: ProcessingIntensity = ProcessingIntensity.STANDARD
    ) -> ProcessingResult:
        """
        Normalize audio using EBU R128 with Audora standards
        
        Args:
            input_file: Path to input audio file
            output_file: Path for normalized output
            intensity: Processing intensity level
            
        Returns:
            ProcessingResult with operation details
        """
        logger.info(f"Starting normalization: {input_file} -> {output_file}")
        
        profile = self.PROCESSING_PROFILES[intensity]
        target_lufs = profile['lufs_target']
        
        try:
            # Create FFmpeg processing chain
            input_stream = ffmpeg.input(input_file)
            
            # Build audio filter chain
            filters = [
                f'loudnorm=I={target_lufs}:TP=-1.0:LRA=11.0:measured_I=-23:measured_LRA=7:measured_TP=-2:measured_thresh=-34:offset=0',
                'highpass=f=20',      # Remove subsonic rumble
                'lowpass=f=20000',    # Remove ultrasonic content
            ]
            
            # Apply filters
            audio = input_stream.audio
            for filter_str in filters:
                audio = audio.filter('af', filter_str)
            
            # Output with high quality settings
            output = ffmpeg.output(
                audio,
                output_file,
                acodec='flac',  # Always normalize to lossless
                audio_bitrate=None,  # Let FLAC determine optimal compression
                ar=48000,  # Standardize sample rate for processing
                format='flac'
            )
            
            # Run the operation
            ffmpeg.run(output, overwrite_output=True, quiet=True)
            
            # Verify output
            if not os.path.exists(output_file):
                raise RuntimeError("Normalization failed - output file not created")
            
            # Analyze normalized result
            specs, quality = self.analyze_audio_file(output_file)
            
            return ProcessingResult(
                success=True,
                input_file=input_file,
                output_files={'normalized': output_file},
                quality_metrics={
                    'lufs': quality.lufs,
                    'peak_db': quality.peak_db,
                    'dynamic_range': quality.dynamic_range,
                    'quality_score': quality.quality_score
                },
                mastered_for_audora=abs(quality.lufs - target_lufs) <= self.QUALITY_STANDARDS['lufs_tolerance'],
                quality_score=quality.quality_score
            )
            
        except Exception as e:
            logger.error(f"Normalization failed: {e}")
            return ProcessingResult(
                success=False,
                input_file=input_file,
                errors=[str(e)]
            )
    
    def generate_streaming_formats(
        self, 
        master_file: str, 
        output_dir: str,
        formats: List[AudioFormat]
    ) -> ProcessingResult:
        """
        Generate multiple streaming formats from master file
        
        Args:
            master_file: Path to master audio file (normalized)
            output_dir: Directory for output files
            formats: List of formats to generate
            
        Returns:
            ProcessingResult with all generated files
        """
        logger.info(f"Generating formats: {formats}")
        
        output_files = {}
        base_name = Path(master_file).stem
        
        try:
            for format_type in formats:
                output_file = os.path.join(output_dir, f"{base_name}_{format_type.value}")
                
                if format_type == AudioFormat.MP3_320:
                    output_file += '.mp3'
                    self._generate_mp3(master_file, output_file, 320)
                    
                elif format_type == AudioFormat.MP3_256:
                    output_file += '.mp3'
                    self._generate_mp3(master_file, output_file, 256)
                    
                elif format_type == AudioFormat.FLAC_CD:
                    output_file += '.flac'
                    self._generate_flac_cd(master_file, output_file)
                    
                elif format_type == AudioFormat.FLAC_HIRES:
                    output_file += '.flac'
                    self._generate_flac_hires(master_file, output_file)
                
                output_files[format_type.value] = output_file
                logger.info(f"Generated {format_type.value}: {output_file}")
            
            return ProcessingResult(
                success=True,
                input_file=master_file,
                output_files=output_files
            )
            
        except Exception as e:
            logger.error(f"Format generation failed: {e}")
            return ProcessingResult(
                success=False,
                input_file=master_file,
                errors=[str(e)]
            )
    
    def _generate_mp3(self, input_file: str, output_file: str, bitrate: int) -> None:
        """Generate MP3 with specified bitrate"""
        output = ffmpeg.output(
            ffmpeg.input(input_file),
            output_file,
            acodec='libmp3lame',
            audio_bitrate=f'{bitrate}k',
            ar=44100,  # Standard CD sample rate for MP3
            ac=2       # Stereo
        )
        ffmpeg.run(output, overwrite_output=True, quiet=True)
    
    def _generate_flac_cd(self, input_file: str, output_file: str) -> None:
        """Generate CD-quality FLAC (16-bit/44.1kHz)"""
        output = ffmpeg.output(
            ffmpeg.input(input_file),
            output_file,
            acodec='flac',
            ar=44100,
            sample_fmt='s16',  # 16-bit
            ac=2
        )
        ffmpeg.run(output, overwrite_output=True, quiet=True)
    
    def _generate_flac_hires(self, input_file: str, output_file: str) -> None:
        """Generate Hi-Res FLAC (preserve original resolution)"""
        output = ffmpeg.output(
            ffmpeg.input(input_file),
            output_file,
            acodec='flac'
            # Preserve original sample rate and bit depth
        )
        ffmpeg.run(output, overwrite_output=True, quiet=True)
    
    def embed_metadata(
        self, 
        file_path: str, 
        metadata: Dict[str, str],
        mastered_for_audora: bool = False
    ) -> str:
        """
        Embed metadata including Mastered for Audora certification
        
        Args:
            file_path: Path to audio file
            metadata: Dictionary of metadata to embed
            mastered_for_audora: Whether file meets Audora standards
            
        Returns:
            Path to file with embedded metadata
        """
        output_file = f"{file_path}_tagged{Path(file_path).suffix}"
        
        try:
            input_stream = ffmpeg.input(file_path)
            
            # Build metadata arguments
            metadata_args = {}
            
            # Standard metadata
            for key, value in metadata.items():
                metadata_args[key] = value
            
            # Audora-specific metadata
            if mastered_for_audora:
                metadata_args.update({
                    'LABEL': 'Mastered for Audora',
                    'AUDORA_CERTIFIED': 'true',
                    'PROCESSING_DATE': '2025-08-19',  # Use actual date
                    'LUFS_TARGET': '-14.0'
                })
            
            # Apply metadata
            output = ffmpeg.output(
                input_stream,
                output_file,
                codec='copy',  # Don't re-encode
                **{'metadata:g:0': f'{k}={v}' for k, v in metadata_args.items()}
            )
            
            ffmpeg.run(output, overwrite_output=True, quiet=True)
            
            return output_file
            
        except Exception as e:
            logger.error(f"Metadata embedding failed: {e}")
            raise
    
    def validate_input_file(self, file_path: str) -> Tuple[bool, List[str]]:
        """
        Validate input audio file against Audora standards
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            # Check file exists and is readable
            if not os.path.exists(file_path):
                issues.append("File does not exist")
                return False, issues
            
            # Check file size (max 600MB)
            file_size = os.path.getsize(file_path)
            if file_size > 600 * 1024 * 1024:
                issues.append(f"File too large: {file_size / (1024*1024):.1f}MB (max 600MB)")
            
            # Analyze audio specifications
            specs, quality = self.analyze_audio_file(file_path)
            
            # Validate format
            supported_codecs = ['mp3', 'flac', 'wav', 'aiff', 'pcm_s16le', 'pcm_s24le']
            if specs.codec not in supported_codecs:
                issues.append(f"Unsupported codec: {specs.codec}")
            
            # Validate sample rate
            supported_rates = [44100, 48000, 88200, 96000, 176400, 192000]
            if specs.sample_rate not in supported_rates:
                issues.append(f"Unsupported sample rate: {specs.sample_rate}Hz")
            
            # Validate channels
            if specs.channels > 8:
                issues.append(f"Too many channels: {specs.channels} (max 8)")
            
            # Validate duration (minimum 10 seconds)
            if specs.duration < 10:
                issues.append(f"Track too short: {specs.duration:.1f}s (min 10s)")
            
            # Quality warnings
            if quality.quality_score < 50:
                issues.append(f"Low quality score: {quality.quality_score}/100")
            
            if quality.peak_db > 0:
                issues.append("Audio appears to be clipped")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            issues.append(f"Validation error: {str(e)}")
            return False, issues