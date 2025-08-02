"""
CLAP (Contrastive Language-Audio Pretraining) Model Manager
Handles loading and inference with CLAP models for audio-text understanding.
"""

import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
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
            # Load audio with librosa
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Normalize audio
            audio = audio / np.max(np.abs(audio))
            
            # Convert to tensor
            audio_tensor = torch.tensor(audio, dtype=torch.float32)
            
            return audio_tensor
            
        except Exception as e:
            print(f"Error preprocessing audio: {str(e)}")
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
    
    def analyze_audio_file(self, audio_path: str) -> Dict[str, Any]:
        """Analyze audio file and extract basic features
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dict[str, Any]: Audio analysis results
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Basic analysis
            analysis = {
                "duration": len(audio) / sr,
                "sample_rate": sr,
                "channels": 1 if len(audio.shape) == 1 else audio.shape[1],
                "total_samples": len(audio),
                "file_size_mb": self._get_file_size_mb(audio_path),
            }
            
            # Spectral features
            if len(audio.shape) > 1:
                audio_mono = np.mean(audio, axis=1)
            else:
                audio_mono = audio
            
            # Spectral centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_mono, sr=sr)[0]
            analysis["spectral_centroid_mean"] = np.mean(spectral_centroids)
            analysis["spectral_centroid_std"] = np.std(spectral_centroids)
            
            # Spectral bandwidth
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_mono, sr=sr)[0]
            analysis["spectral_bandwidth_mean"] = np.mean(spectral_bandwidth)
            
            # Zero crossing rate
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_mono)[0]
            analysis["zero_crossing_rate_mean"] = np.mean(zero_crossing_rate)
            
            # RMS energy
            rms = librosa.feature.rms(y=audio_mono)[0]
            analysis["rms_mean"] = np.mean(rms)
            analysis["rms_std"] = np.std(rms)
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio_mono, sr=sr, n_mfcc=13)
            analysis["mfcc_mean"] = np.mean(mfccs, axis=1).tolist()
            analysis["mfcc_std"] = np.std(mfccs, axis=1).tolist()
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing audio file: {str(e)}")
            return {}
    
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