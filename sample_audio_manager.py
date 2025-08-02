"""
Sample Audio Manager for CLAP Application
Provides access to sample audio files for testing and demonstration.
"""

import os
import tempfile
import requests
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class SampleAudioManager:
    """Manager for sample audio files"""
    
    def __init__(self):
        """Initialize sample audio manager"""
        self.sample_audio_files = {
            # 音楽サンプル
            "音楽サンプル (クラシック)": {
                "url": "https://www.soundjay.com/misc/sounds/bell-ringing-05.wav",
                "description": "クラシックな鐘の音",
                "category": "音楽"
            },
            "音楽サンプル (ロック)": {
                "url": "https://www.soundjay.com/misc/sounds/rock-guitar-01.wav",
                "description": "ロックギターの音",
                "category": "音楽"
            },
            "音楽サンプル (ピアノ)": {
                "url": "https://www.soundjay.com/misc/sounds/piano-01.wav",
                "description": "ピアノの音",
                "category": "音楽"
            },
            
            # 自然音
            "自然音 (雨)": {
                "url": "https://www.soundjay.com/nature/sounds/rain-01.wav",
                "description": "雨の音",
                "category": "自然"
            },
            "自然音 (鳥の鳴き声)": {
                "url": "https://www.soundjay.com/nature/sounds/bird-01.wav",
                "description": "鳥の鳴き声",
                "category": "自然"
            },
            "自然音 (風)": {
                "url": "https://www.soundjay.com/nature/sounds/wind-01.wav",
                "description": "風の音",
                "category": "自然"
            },
            "自然音 (波)": {
                "url": "https://www.soundjay.com/nature/sounds/ocean-wave-01.wav",
                "description": "波の音",
                "category": "自然"
            },
            "自然音 (雷)": {
                "url": "https://www.soundjay.com/nature/sounds/thunder-01.wav",
                "description": "雷の音",
                "category": "自然"
            },
            
            # 都市音
            "都市音 (車の音)": {
                "url": "https://www.soundjay.com/misc/sounds/car-horn-01.wav",
                "description": "車のクラクション",
                "category": "都市"
            },
            "都市音 (電車)": {
                "url": "https://www.soundjay.com/misc/sounds/train-01.wav",
                "description": "電車の音",
                "category": "都市"
            },
            "都市音 (救急車)": {
                "url": "https://www.soundjay.com/misc/sounds/ambulance-01.wav",
                "description": "救急車のサイレン",
                "category": "都市"
            },
            
            # 人の声
            "人の声 (笑い声)": {
                "url": "https://www.soundjay.com/human/sounds/laugh-01.wav",
                "description": "人の笑い声",
                "category": "人の声"
            },
            "人の声 (拍手)": {
                "url": "https://www.soundjay.com/human/sounds/applause-01.wav",
                "description": "拍手の音",
                "category": "人の声"
            },
            "人の声 (咳)": {
                "url": "https://www.soundjay.com/human/sounds/cough-01.wav",
                "description": "咳の音",
                "category": "人の声"
            },
            
            # 機械音
            "機械音 (キーボード)": {
                "url": "https://www.soundjay.com/misc/sounds/keyboard-01.wav",
                "description": "キーボードのタイピング音",
                "category": "機械"
            },
            "機械音 (プリンター)": {
                "url": "https://www.soundjay.com/misc/sounds/printer-01.wav",
                "description": "プリンターの音",
                "category": "機械"
            },
            "機械音 (電話)": {
                "url": "https://www.soundjay.com/misc/sounds/phone-ring-01.wav",
                "description": "電話の着信音",
                "category": "機械"
            },
            "機械音 (時計)": {
                "url": "https://www.soundjay.com/misc/sounds/clock-tick-01.wav",
                "description": "時計の音",
                "category": "機械"
            },
            "機械音 (ドアベル)": {
                "url": "https://www.soundjay.com/misc/sounds/doorbell-01.wav",
                "description": "ドアベルの音",
                "category": "機械"
            },
            "機械音 (アラーム)": {
                "url": "https://www.soundjay.com/misc/sounds/alarm-01.wav",
                "description": "アラーム音",
                "category": "機械"
            },
            
            # 動物音
            "動物音 (犬の鳴き声)": {
                "url": "https://www.soundjay.com/animals/sounds/dog-bark-01.wav",
                "description": "犬の鳴き声",
                "category": "動物"
            },
            "動物音 (猫の鳴き声)": {
                "url": "https://www.soundjay.com/animals/sounds/cat-meow-01.wav",
                "description": "猫の鳴き声",
                "category": "動物"
            },
            "動物音 (馬の鳴き声)": {
                "url": "https://www.soundjay.com/animals/sounds/horse-01.wav",
                "description": "馬の鳴き声",
                "category": "動物"
            }
        }
        
        # カテゴリ別のクエリサジェスト
        self.category_queries = {
            "音楽": [
                "音楽が流れている",
                "楽器の音",
                "ピアノの音",
                "ギターの音",
                "鐘の音",
                "メロディー",
                "リズム",
                "ハーモニー"
            ],
            "自然": [
                "雨の音",
                "風の音",
                "波の音",
                "雷の音",
                "鳥の鳴き声",
                "自然の音",
                "環境音",
                "天候の音"
            ],
            "都市": [
                "車の音",
                "電車の音",
                "サイレンの音",
                "交通の音",
                "都市の騒音",
                "機械の音",
                "エンジンの音"
            ],
            "人の声": [
                "人の話し声",
                "笑い声",
                "拍手",
                "咳の音",
                "人の音",
                "会話の音",
                "感情表現の音"
            ],
            "機械": [
                "キーボードの音",
                "プリンターの音",
                "電話の音",
                "時計の音",
                "ドアベルの音",
                "アラーム音",
                "電子機器の音",
                "機械の動作音"
            ],
            "動物": [
                "犬の鳴き声",
                "猫の鳴き声",
                "馬の鳴き声",
                "動物の音",
                "ペットの音",
                "野生動物の音"
            ]
        }
    
    def get_sample_audio_list(self) -> List[str]:
        """Get list of available sample audio names
        
        Returns:
            List[str]: List of sample audio names
        """
        return list(self.sample_audio_files.keys())
    
    def get_sample_audio_info(self, audio_name: str) -> Optional[Dict]:
        """Get information about a sample audio file
        
        Args:
            audio_name: Name of the sample audio
            
        Returns:
            Optional[Dict]: Audio information or None if not found
        """
        return self.sample_audio_files.get(audio_name)
    
    def download_sample_audio(self, audio_name: str) -> Optional[str]:
        """Download a sample audio file
        
        Args:
            audio_name: Name of the sample audio to download
            
        Returns:
            Optional[str]: Path to downloaded file or None if failed
        """
        if audio_name not in self.sample_audio_files:
            return None
        
        try:
            audio_info = self.sample_audio_files[audio_name]
            response = requests.get(audio_info["url"], timeout=30)
            
            if response.status_code == 200:
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                    tmp_file.write(response.content)
                    return tmp_file.name
            else:
                print(f"Failed to download {audio_name}: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error downloading {audio_name}: {str(e)}")
            return None
    
    def get_category_queries(self, category: str) -> List[str]:
        """Get suggested queries for a specific category
        
        Args:
            category: Audio category
            
        Returns:
            List[str]: List of suggested queries
        """
        return self.category_queries.get(category, [])
    
    def get_all_categories(self) -> List[str]:
        """Get all available categories
        
        Returns:
            List[str]: List of all categories
        """
        return list(self.category_queries.keys())
    
    def get_audio_by_category(self, category: str) -> List[str]:
        """Get audio files by category
        
        Args:
            category: Audio category
            
        Returns:
            List[str]: List of audio names in the category
        """
        return [name for name, info in self.sample_audio_files.items() 
                if info["category"] == category]
    
    def create_synthetic_audio(self, audio_type: str, duration: float = 3.0, 
                              sample_rate: int = 22050) -> Optional[str]:
        """Create synthetic audio for testing
        
        Args:
            audio_type: Type of synthetic audio to create
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            
        Returns:
            Optional[str]: Path to created audio file
        """
        try:
            samples = int(duration * sample_rate)
            
            if audio_type == "sine_wave":
                # 440 Hz sine wave
                t = np.linspace(0, duration, samples)
                audio = np.sin(2 * np.pi * 440 * t)
            elif audio_type == "white_noise":
                # White noise
                audio = np.random.normal(0, 0.1, samples)
            elif audio_type == "pink_noise":
                # Pink noise approximation
                audio = np.random.normal(0, 0.1, samples)
                # Apply simple low-pass filter
                from scipy import signal
                b, a = signal.butter(4, 0.1, 'low')
                audio = signal.filtfilt(b, a, audio)
            elif audio_type == "chirp":
                # Frequency sweep
                t = np.linspace(0, duration, samples)
                audio = signal.chirp(t, 100, duration, 1000)
            else:
                return None
            
            # Normalize
            audio = audio / np.max(np.abs(audio))
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                sf.write(tmp_file.name, audio, sample_rate)
                return tmp_file.name
                
        except Exception as e:
            print(f"Error creating synthetic audio: {str(e)}")
            return None
    
    def get_synthetic_audio_types(self) -> List[str]:
        """Get available synthetic audio types
        
        Returns:
            List[str]: List of synthetic audio types
        """
        return ["sine_wave", "white_noise", "pink_noise", "chirp"] 