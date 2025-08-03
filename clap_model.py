"""
CLAP (Contrastive Language-Audio Pretraining) Model Manager
Handles loading and inference with CLAP models for audio-text understanding.
"""

import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
import os
from typing import List, Dict, Any, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import ClapProcessor, ClapModel
except ImportError:
    print("Warning: transformers library not found. Please install it for CLAP functionality.")


class CLAPModelManager:
    """CLAP Model Manager for audio-text understanding"""
    
    def __init__(self, model_name: str = "laion/clap-htsat-fused"):
        """Initialize CLAP model manager
        
        Args:
            model_name: Hugging Face model name for CLAP
        """
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = 48000  # CLAP default sample rate
        
    def load_model(self) -> bool:
        """Load CLAP model and processor
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            print(f"Loading CLAP model: {self.model_name}")
            self.processor = ClapProcessor.from_pretrained(self.model_name)
            self.model = ClapModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"CLAP model loaded successfully on {self.device}")
            return True
        except Exception as e:
            print(f"Error loading CLAP model: {str(e)}")
            return False
    
    def preprocess_audio(self, audio_path: str) -> Optional[torch.Tensor]:
        """Preprocess audio file for CLAP
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            torch.Tensor: Preprocessed audio tensor
        """
        try:
            # Try multiple audio loading methods
            audio = None
            sr = None
            
            # Method 1: Try librosa with different backends
            try:
                audio, sr = librosa.load(audio_path, sr=self.sample_rate)
                print(f"Successfully loaded audio with librosa: {audio_path}")
            except Exception as e:
                print(f"librosa failed: {str(e)}")
                
                # Method 2: Try soundfile
                try:
                    import soundfile as sf
                    audio, sr = sf.read(audio_path)
                    if len(audio.shape) > 1:
                        audio = np.mean(audio, axis=1)  # Convert to mono
                    # Resample if necessary
                    if sr != self.sample_rate:
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                        sr = self.sample_rate
                    print(f"Successfully loaded audio with soundfile: {audio_path}")
                except Exception as e2:
                    print(f"soundfile failed: {str(e2)}")
                    
                    # Method 3: Try pydub
                    try:
                        from pydub import AudioSegment
                        audio_segment = AudioSegment.from_file(audio_path)
                        audio = np.array(audio_segment.get_array_of_samples())
                        sr = audio_segment.frame_rate
                        # Convert to mono if stereo
                        if audio_segment.channels == 2:
                            audio = audio.reshape((-1, 2))
                            audio = np.mean(audio, axis=1)
                        # Resample if necessary
                        if sr != self.sample_rate:
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                            sr = self.sample_rate
                        print(f"Successfully loaded audio with pydub: {audio_path}")
                    except Exception as e3:
                        print(f"pydub failed: {str(e3)}")
                        raise Exception(f"All audio loading methods failed: {str(e)}, {str(e2)}, {str(e3)}")
            
            if audio is None or len(audio) == 0:
                raise Exception("Audio file is empty or could not be loaded")
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Normalize audio
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            # Convert to tensor
            audio_tensor = torch.tensor(audio, dtype=torch.float32)
            
            return audio_tensor
            
        except Exception as e:
            print(f"Error preprocessing audio: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return None
    
    def encode_audio(self, audio_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """Encode audio using CLAP
        
        Args:
            audio_tensor: Preprocessed audio tensor
            
        Returns:
            torch.Tensor: Audio embeddings
        """
        if self.model is None or self.processor is None:
            print("Model not loaded. Please call load_model() first.")
            return None
        
        try:
            with torch.no_grad():
                # Process audio through CLAP
                inputs = self.processor(
                    audios=audio_tensor.unsqueeze(0), 
                    sampling_rate=self.sample_rate, 
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get audio embeddings
                audio_embeddings = self.model.get_audio_features(**inputs)
                
                return audio_embeddings
                
        except Exception as e:
            print(f"Error encoding audio: {str(e)}")
            return None
    
    def encode_text(self, text: str) -> Optional[torch.Tensor]:
        """Encode text using CLAP
        
        Args:
            text: Input text
            
        Returns:
            torch.Tensor: Text embeddings
        """
        if self.model is None or self.processor is None:
            print("Model not loaded. Please call load_model() first.")
            return None
        
        try:
            with torch.no_grad():
                # Process text through CLAP
                inputs = self.processor(
                    text=text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get text embeddings
                text_embeddings = self.model.get_text_features(**inputs)
                
                return text_embeddings
                
        except Exception as e:
            print(f"Error encoding text: {str(e)}")
            return None
    
    def compute_similarity(self, audio_embeddings: torch.Tensor, text_embeddings: torch.Tensor) -> float:
        """Compute cosine similarity between audio and text embeddings
        
        Args:
            audio_embeddings: Audio embeddings
            text_embeddings: Text embeddings
            
        Returns:
            float: Similarity score
        """
        try:
            # Normalize embeddings
            audio_norm = torch.nn.functional.normalize(audio_embeddings, dim=-1)
            text_norm = torch.nn.functional.normalize(text_embeddings, dim=-1)
            
            # Compute cosine similarity
            similarity = torch.cosine_similarity(audio_norm, text_norm, dim=-1)
            
            return similarity.item()
            
        except Exception as e:
            print(f"Error computing similarity: {str(e)}")
            return 0.0
    
    def audio_text_matching(self, audio_path: str, text_queries: List[str]) -> Dict[str, float]:
        """Match audio with multiple text queries
        
        Args:
            audio_path: Path to audio file
            text_queries: List of text queries to match against
            
        Returns:
            Dict[str, float]: Dictionary mapping text queries to similarity scores
        """
        results = {}
        
        # Preprocess audio
        audio_tensor = self.preprocess_audio(audio_path)
        if audio_tensor is None:
            return results
        
        # Encode audio
        audio_embeddings = self.encode_audio(audio_tensor)
        if audio_embeddings is None:
            return results
        
        # Encode each text query and compute similarity
        for text in text_queries:
            text_embeddings = self.encode_text(text)
            if text_embeddings is not None:
                similarity = self.compute_similarity(audio_embeddings, text_embeddings)
                results[text] = similarity
        
        return results
    
    def audio_text_matching_with_timing(self, audio_path: str, text_queries: List[str]) -> Dict[str, Any]:
        """Match audio with multiple text queries and return timing information
        
        Args:
            audio_path: Path to audio file
            text_queries: List of text queries to match against
            
        Returns:
            Dict[str, Any]: Dictionary with similarity scores and timing information
        """
        import time
        
        results = {
            "similarities": {},
            "timing": {
                "total_time": 0.0,
                "audio_processing_time": 0.0,
                "text_processing_time": 0.0,
                "similarity_computation_time": 0.0,
                "per_query_times": {}
            },
            "audio_info": {}
        }
        
        start_total = time.time()
        
        # Audio processing timing
        start_audio = time.time()
        audio_tensor = self.preprocess_audio(audio_path)
        if audio_tensor is None:
            return results
        
        audio_embeddings = self.encode_audio(audio_tensor)
        if audio_embeddings is None:
            return results
        results["timing"]["audio_processing_time"] = time.time() - start_audio
        
        # Get audio duration
        try:
            audio, sr = librosa.load(audio_path, sr=None)
            results["audio_info"]["duration"] = len(audio) / sr
            results["audio_info"]["sample_rate"] = sr
        except Exception as e:
            print(f"Error getting audio info: {str(e)}")
            results["audio_info"]["duration"] = 0.0
            results["audio_info"]["sample_rate"] = 0
        
        # Text processing and similarity computation timing
        start_text = time.time()
        for text in text_queries:
            query_start = time.time()
            
            text_embeddings = self.encode_text(text)
            if text_embeddings is not None:
                start_similarity = time.time()
                similarity = self.compute_similarity(audio_embeddings, text_embeddings)
                similarity_time = time.time() - start_similarity
                
                results["similarities"][text] = similarity
                results["timing"]["per_query_times"][text] = {
                    "total_time": time.time() - query_start,
                    "similarity_time": similarity_time
                }
        
        results["timing"]["text_processing_time"] = time.time() - start_text
        results["timing"]["total_time"] = time.time() - start_total
        
        return results
    
    def get_audio_features(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """Extract audio features using CLAP
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dict[str, Any]: Audio features and metadata
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Preprocess
            audio_tensor = self.preprocess_audio(audio_path)
            if audio_tensor is None:
                return None
            
            # Encode
            audio_embeddings = self.encode_audio(audio_tensor)
            if audio_embeddings is None:
                return None
            
            # Extract features
            features = {
                "duration": len(audio) / sr,
                "sample_rate": sr,
                "embedding_shape": audio_embeddings.shape,
                "embedding_mean": audio_embeddings.mean().item(),
                "embedding_std": audio_embeddings.std().item(),
                "embedding_norm": torch.norm(audio_embeddings).item(),
            }
            
            return features
            
        except Exception as e:
            print(f"Error extracting audio features: {str(e)}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model
        
        Returns:
            Dict[str, Any]: Model information
        """
        info = {
            "model_name": self.model_name,
            "device": str(self.device),
            "sample_rate": self.sample_rate,
            "model_loaded": self.model is not None,
            "processor_loaded": self.processor is not None,
        }
        
        if self.model is not None:
            info.update({
                "model_parameters": sum(p.numel() for p in self.model.parameters()),
                "model_trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            })
        
        return info


class AudioAnalyzer:
    """Audio analysis utilities for CLAP application"""
    
    def __init__(self):
        """Initialize audio analyzer"""
        self.supported_formats = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    
    def validate_audio_file(self, audio_path: str) -> Dict[str, Any]:
        """Validate audio file before processing
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dict[str, Any]: Validation results with status and details
        """
        validation = {
            "is_valid": False,
            "errors": [],
            "warnings": [],
            "file_info": {}
        }
        
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                validation["errors"].append(f"ファイルが存在しません: {audio_path}")
                return validation
            
            # Get file info
            file_size = self._get_file_size_mb(audio_path)
            file_ext = os.path.splitext(audio_path.lower())[1]
            
            validation["file_info"] = {
                "path": audio_path,
                "size_mb": file_size,
                "extension": file_ext
            }
            
            # Check file size
            if file_size == 0:
                validation["errors"].append("ファイルサイズが0です")
                return validation
            
            if file_size < 0.001:  # Less than 1KB
                validation["warnings"].append(f"ファイルサイズが非常に小さいです ({file_size:.3f}MB)")
            
            if file_size > 100:  # 100MB limit
                validation["errors"].append(f"ファイルサイズが大きすぎます ({file_size:.1f}MB > 100MB)")
                return validation
            
            # Check file extension
            if file_ext not in self.supported_formats:
                validation["warnings"].append(f"サポートされていないファイル形式です: {file_ext}")
            
            # Try to read file header
            try:
                with open(audio_path, 'rb') as f:
                    header = f.read(12)  # Read first 12 bytes
                    
                # Check for common audio file signatures
                if file_ext == '.wav':
                    if not header.startswith(b'RIFF') or not header[8:12] == b'WAVE':
                        validation["errors"].append("WAVファイルのヘッダーが無効です")
                        return validation
                elif file_ext == '.mp3':
                    if not (header.startswith(b'ID3') or header.startswith(b'\xff\xfb') or header.startswith(b'\xff\xf3')):
                        validation["warnings"].append("MP3ファイルのヘッダーが標準的ではありません")
                elif file_ext == '.flac':
                    if not header.startswith(b'fLaC'):
                        validation["errors"].append("FLACファイルのヘッダーが無効です")
                        return validation
                        
            except Exception as e:
                validation["warnings"].append(f"ファイルヘッダーの読み取りに失敗: {str(e)}")
            
            # Try to load audio with multiple methods
            audio_loaded = False
            load_errors = []
            
            # Method 1: librosa
            try:
                audio, sr = librosa.load(audio_path, sr=None, duration=1.0)  # Load only first second for validation
                if len(audio) > 0:
                    audio_loaded = True
                    validation["file_info"]["sample_rate"] = sr
                    validation["file_info"]["duration_seconds"] = len(audio) / sr
            except Exception as e:
                load_errors.append(f"librosa: {str(e)}")
            
            # Method 2: soundfile
            if not audio_loaded:
                try:
                    import soundfile as sf
                    audio, sr = sf.read(audio_path, start=0, frames=48000)  # Read first second
                    if len(audio) > 0:
                        audio_loaded = True
                        validation["file_info"]["sample_rate"] = sr
                        validation["file_info"]["duration_seconds"] = len(audio) / sr
                except Exception as e:
                    load_errors.append(f"soundfile: {str(e)}")
            
            # Method 3: pydub
            if not audio_loaded:
                try:
                    from pydub import AudioSegment
                    audio_segment = AudioSegment.from_file(audio_path)
                    if len(audio_segment) > 0:
                        audio_loaded = True
                        validation["file_info"]["sample_rate"] = audio_segment.frame_rate
                        validation["file_info"]["duration_seconds"] = len(audio_segment) / 1000.0
                except Exception as e:
                    load_errors.append(f"pydub: {str(e)}")
            
            if not audio_loaded:
                validation["errors"].append("すべての音声読み込み方法が失敗しました")
                validation["errors"].extend(load_errors)
                return validation
            
            # Additional checks
            if "duration_seconds" in validation["file_info"]:
                duration = validation["file_info"]["duration_seconds"]
                if duration < 0.1:
                    validation["warnings"].append(f"音声の長さが短すぎます ({duration:.3f}秒)")
                elif duration > 3600:  # 1 hour
                    validation["warnings"].append(f"音声の長さが長すぎます ({duration:.1f}秒)")
            
            # If we get here, file is valid
            validation["is_valid"] = True
            
        except Exception as e:
            validation["errors"].append(f"検証中にエラーが発生しました: {str(e)}")
            import traceback
            validation["errors"].append(f"詳細: {traceback.format_exc()}")
        
        return validation
    
    def analyze_audio_file(self, audio_path: str) -> Dict[str, Any]:
        """Analyze audio file and extract basic features
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dict[str, Any]: Audio analysis results
        """
        # First validate the file
        validation = self.validate_audio_file(audio_path)
        if not validation["is_valid"]:
            print("音声ファイルの検証に失敗しました:")
            for error in validation["errors"]:
                print(f"  - {error}")
            return self._get_default_analysis()
        
        # Show warnings if any
        if validation["warnings"]:
            print("音声ファイルの警告:")
            for warning in validation["warnings"]:
                print(f"  - {warning}")
        
        try:
            # Load audio
            try:
                audio, sr = librosa.load(audio_path, sr=None)
            except Exception as e:
                print(f"librosa failed: {str(e)}")
                
                # Try soundfile
                try:
                    import soundfile as sf
                    audio, sr = sf.read(audio_path)
                    if len(audio.shape) > 1:
                        audio = np.mean(audio, axis=1)  # Convert to mono
                    print(f"Successfully loaded audio with soundfile: {audio_path}")
                except Exception as e2:
                    print(f"soundfile failed: {str(e2)}")
                    
                    # Try pydub
                    try:
                        from pydub import AudioSegment
                        audio_segment = AudioSegment.from_file(audio_path)
                        audio = np.array(audio_segment.get_array_of_samples())
                        sr = audio_segment.frame_rate
                        # Convert to mono if stereo
                        if audio_segment.channels == 2:
                            audio = audio.reshape((-1, 2))
                            audio = np.mean(audio, axis=1)
                        print(f"Successfully loaded audio with pydub: {audio_path}")
                    except Exception as e3:
                        print(f"pydub failed: {str(e3)}")
                        print(f"All audio loading methods failed: {str(e)}, {str(e2)}, {str(e3)}")
                        return self._get_default_analysis()
            
            # Check if audio is valid
            if len(audio) == 0:
                print("Error: Audio file is empty")
                return self._get_default_analysis()
            
            # Basic analysis
            analysis = {
                "duration": len(audio) / sr,
                "sample_rate": sr,
                "channels": 1 if len(audio.shape) == 1 else audio.shape[1],
                "total_samples": len(audio),
                "file_size_mb": validation["file_info"]["size_mb"],
            }
            
            # Spectral features
            if len(audio.shape) > 1:
                audio_mono = np.mean(audio, axis=1)
            else:
                audio_mono = audio
            
            # Ensure audio is not empty
            if len(audio_mono) == 0:
                print("Warning: Audio file is empty after processing")
                return analysis
            
            # Spectral centroid
            try:
                spectral_centroids = librosa.feature.spectral_centroid(y=audio_mono, sr=sr)[0]
                analysis["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
                analysis["spectral_centroid_std"] = float(np.std(spectral_centroids))
            except Exception as e:
                print(f"Error computing spectral centroid: {str(e)}")
                analysis["spectral_centroid_mean"] = 0.0
                analysis["spectral_centroid_std"] = 0.0
            
            # Spectral bandwidth
            try:
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_mono, sr=sr)[0]
                analysis["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))
            except Exception as e:
                print(f"Error computing spectral bandwidth: {str(e)}")
                analysis["spectral_bandwidth_mean"] = 0.0
            
            # Zero crossing rate
            try:
                zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_mono)[0]
                analysis["zero_crossing_rate_mean"] = float(np.mean(zero_crossing_rate))
            except Exception as e:
                print(f"Error computing zero crossing rate: {str(e)}")
                analysis["zero_crossing_rate_mean"] = 0.0
            
            # RMS energy
            try:
                rms = librosa.feature.rms(y=audio_mono)[0]
                analysis["rms_mean"] = float(np.mean(rms))
                analysis["rms_std"] = float(np.std(rms))
            except Exception as e:
                print(f"Error computing RMS: {str(e)}")
                analysis["rms_mean"] = 0.0
                analysis["rms_std"] = 0.0
            
            # MFCC features
            try:
                mfccs = librosa.feature.mfcc(y=audio_mono, sr=sr, n_mfcc=13)
                analysis["mfcc_mean"] = np.mean(mfccs, axis=1).tolist()
                analysis["mfcc_std"] = np.std(mfccs, axis=1).tolist()
            except Exception as e:
                print(f"Error computing MFCC: {str(e)}")
                analysis["mfcc_mean"] = [0.0] * 13
                analysis["mfcc_std"] = [0.0] * 13
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing audio file: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return self._get_default_analysis()
    
    def _get_default_analysis(self) -> Dict[str, Any]:
        """Get default analysis results when analysis fails
        
        Returns:
            Dict[str, Any]: Default analysis results
        """
        return {
            "duration": 0.0,
            "sample_rate": 0,
            "channels": 0,
            "total_samples": 0,
            "file_size_mb": 0.0,
            "spectral_centroid_mean": 0.0,
            "spectral_centroid_std": 0.0,
            "spectral_bandwidth_mean": 0.0,
            "zero_crossing_rate_mean": 0.0,
            "rms_mean": 0.0,
            "rms_std": 0.0,
            "mfcc_mean": [0.0] * 13,
            "mfcc_std": [0.0] * 13
        }
    
    def _get_file_size_mb(self, file_path: str) -> float:
        """Get file size in MB
        
        Args:
            file_path: Path to file
            
        Returns:
            float: File size in MB
        """
        import os
        try:
            size_bytes = os.path.getsize(file_path)
            return size_bytes / (1024 * 1024)
        except:
            return 0.0
    
    def is_supported_format(self, file_path: str) -> bool:
        """Check if audio format is supported
        
        Args:
            file_path: Path to audio file
            
        Returns:
            bool: True if supported, False otherwise
        """
        import os
        _, ext = os.path.splitext(file_path.lower())
        return ext in self.supported_formats 