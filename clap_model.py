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


class AudioSearchEngine:
    """Audio search engine with success detection and ranking"""
    
    def __init__(self, clap_manager: CLAPModelManager):
        """Initialize audio search engine
        
        Args:
            clap_manager: CLAP model manager instance
        """
        self.clap_manager = clap_manager
        
        # Search success thresholds
        self.success_thresholds = {
            "excellent": 0.7,    # 優秀な検索結果
            "good": 0.5,         # 良好な検索結果
            "fair": 0.3,         # 普通の検索結果
            "poor": 0.1          # 貧弱な検索結果
        }
        
        # Common audio categories and their related queries
        self.audio_categories = {
            "音楽": [
                "音楽が流れている", "楽器の音", "歌", "メロディー", "リズム", "ビート",
                "ピアノ", "ギター", "ドラム", "バイオリン", "オーケストラ", "バンド"
            ],
            "人の声": [
                "人の話し声", "会話", "笑い声", "叫び声", "歌声", "朗読",
                "男性の声", "女性の声", "子供の声", "老人の声"
            ],
            "自然音": [
                "鳥の鳴き声", "風の音", "雨の音", "波の音", "雷の音", "川の流れ",
                "森の音", "虫の音", "動物の鳴き声", "木の葉の音"
            ],
            "機械音": [
                "エンジン音", "車の音", "飛行機の音", "電車の音", "機械の音",
                "アラーム", "ベル", "電話の音", "時計の音", "キーボードの音"
            ],
            "環境音": [
                "街の音", "オフィスの音", "レストランの音", "学校の音", "病院の音",
                "工場の音", "工事の音", "騒音", "静寂"
            ]
        }
    
    def search_audio(self, audio_path: str, search_query: str, 
                    include_related: bool = True, 
                    category_search: bool = False) -> Dict[str, Any]:
        """Search audio content with detailed analysis
        
        Args:
            audio_path: Path to audio file
            search_query: Text query to search for
            include_related: Whether to include related queries
            category_search: Whether to search within categories
            
        Returns:
            Dict[str, Any]: Search results with success analysis
        """
        import time
        
        search_results = {
            "query": search_query,
            "audio_path": audio_path,
            "search_time": time.time(),
            "results": {},
            "success_analysis": {},
            "category_matches": {},
            "recommendations": []
        }
        
        # Prepare queries
        queries = [search_query]
        
        if include_related:
            # Add related queries based on the main query
            related_queries = self._get_related_queries(search_query)
            queries.extend(related_queries)
        
        if category_search:
            # Add category-based queries
            category_queries = self._get_category_queries(search_query)
            queries.extend(category_queries)
        
        # Perform search
        start_time = time.time()
        similarity_scores = self.clap_manager.audio_text_matching(audio_path, queries)
        search_time = time.time() - start_time
        
        search_results["results"] = similarity_scores
        search_results["search_time"] = search_time
        
        # Analyze search success
        success_analysis = self._analyze_search_success(similarity_scores, search_query)
        search_results["success_analysis"] = success_analysis
        
        # Find category matches
        category_matches = self._find_category_matches(similarity_scores)
        search_results["category_matches"] = category_matches
        
        # Generate recommendations
        recommendations = self._generate_recommendations(similarity_scores, success_analysis)
        search_results["recommendations"] = recommendations
        
        return search_results
    
    def _get_related_queries(self, main_query: str) -> List[str]:
        """Get related queries based on the main query
        
        Args:
            main_query: Main search query
            
        Returns:
            List[str]: Related queries
        """
        related_queries = []
        
        # Simple keyword-based expansion
        query_lower = main_query.lower()
        
        if "音楽" in query_lower or "music" in query_lower:
            related_queries.extend(["楽器の音", "メロディー", "リズム"])
        elif "人" in query_lower or "voice" in query_lower or "speech" in query_lower:
            related_queries.extend(["会話", "笑い声", "歌声"])
        elif "鳥" in query_lower or "bird" in query_lower:
            related_queries.extend(["動物の鳴き声", "自然の音"])
        elif "車" in query_lower or "car" in query_lower:
            related_queries.extend(["エンジン音", "機械の音"])
        elif "雨" in query_lower or "rain" in query_lower:
            related_queries.extend(["風の音", "雷の音"])
        
        return list(set(related_queries))  # Remove duplicates
    
    def _get_category_queries(self, search_query: str) -> List[str]:
        """Get category-based queries
        
        Args:
            search_query: Search query
            
        Returns:
            List[str]: Category-based queries
        """
        category_queries = []
        query_lower = search_query.lower()
        
        # Find matching categories
        for category, queries in self.audio_categories.items():
            for query in queries:
                if any(keyword in query_lower for keyword in query.split()):
                    category_queries.extend(queries[:5])  # Limit to 5 queries per category
                    break
        
        return list(set(category_queries))  # Remove duplicates
    
    def _analyze_search_success(self, similarity_scores: Dict[str, float], 
                               main_query: str) -> Dict[str, Any]:
        """Analyze search success based on similarity scores
        
        Args:
            similarity_scores: Dictionary of query to similarity score
            main_query: Main search query
            
        Returns:
            Dict[str, Any]: Success analysis
        """
        if not similarity_scores:
            return {
                "is_successful": False,
                "confidence_level": "none",
                "best_score": 0.0,
                "main_query_score": 0.0,
                "average_score": 0.0,
                "score_distribution": {},
                "success_reason": "検索結果がありません"
            }
        
        # Get main query score
        main_query_score = similarity_scores.get(main_query, 0.0)
        
        # Get best score
        best_score = max(similarity_scores.values())
        best_query = max(similarity_scores.items(), key=lambda x: x[1])[0]
        
        # Calculate average score
        average_score = sum(similarity_scores.values()) / len(similarity_scores)
        
        # Determine confidence level
        confidence_level = "none"
        if best_score >= self.success_thresholds["excellent"]:
            confidence_level = "excellent"
        elif best_score >= self.success_thresholds["good"]:
            confidence_level = "good"
        elif best_score >= self.success_thresholds["fair"]:
            confidence_level = "fair"
        elif best_score >= self.success_thresholds["poor"]:
            confidence_level = "poor"
        
        # Determine if search is successful
        is_successful = best_score >= self.success_thresholds["fair"]
        
        # Generate success reason
        success_reason = self._generate_success_reason(
            is_successful, confidence_level, best_score, main_query_score, best_query
        )
        
        # Score distribution
        score_distribution = {
            "excellent": len([s for s in similarity_scores.values() if s >= self.success_thresholds["excellent"]]),
            "good": len([s for s in similarity_scores.values() if s >= self.success_thresholds["good"]]),
            "fair": len([s for s in similarity_scores.values() if s >= self.success_thresholds["fair"]]),
            "poor": len([s for s in similarity_scores.values() if s >= self.success_thresholds["poor"]])
        }
        
        return {
            "is_successful": is_successful,
            "confidence_level": confidence_level,
            "best_score": best_score,
            "best_query": best_query,
            "main_query_score": main_query_score,
            "average_score": average_score,
            "score_distribution": score_distribution,
            "success_reason": success_reason
        }
    
    def _generate_success_reason(self, is_successful: bool, confidence_level: str,
                                best_score: float, main_query_score: float, 
                                best_query: str) -> str:
        """Generate human-readable success reason
        
        Args:
            is_successful: Whether search was successful
            confidence_level: Confidence level
            best_score: Best similarity score
            main_query_score: Main query score
            best_query: Best matching query
            
        Returns:
            str: Success reason
        """
        if not is_successful:
            return f"検索に失敗しました。最高スコア: {best_score:.3f} (閾値: {self.success_thresholds['fair']})"
        
        reasons = []
        
        if confidence_level == "excellent":
            reasons.append("優秀な検索結果")
        elif confidence_level == "good":
            reasons.append("良好な検索結果")
        elif confidence_level == "fair":
            reasons.append("普通の検索結果")
        
        if best_query != "main_query" and main_query_score < best_score:
            reasons.append(f"関連クエリ「{best_query}」がより良いマッチ")
        
        if best_score > 0.8:
            reasons.append("非常に高い類似度")
        elif best_score > 0.6:
            reasons.append("高い類似度")
        
        return " | ".join(reasons) if reasons else "検索成功"
    
    def _find_category_matches(self, similarity_scores: Dict[str, float]) -> Dict[str, float]:
        """Find category matches based on similarity scores
        
        Args:
            similarity_scores: Dictionary of query to similarity score
            
        Returns:
            Dict[str, float]: Category to average score mapping
        """
        category_scores = {}
        
        for category, queries in self.audio_categories.items():
            category_query_scores = []
            for query in queries:
                if query in similarity_scores:
                    category_query_scores.append(similarity_scores[query])
            
            if category_query_scores:
                category_scores[category] = sum(category_query_scores) / len(category_query_scores)
        
        return category_scores
    
    def _generate_recommendations(self, similarity_scores: Dict[str, float],
                                 success_analysis: Dict[str, Any]) -> List[str]:
        """Generate search recommendations
        
        Args:
            similarity_scores: Dictionary of query to similarity score
            success_analysis: Success analysis results
            
        Returns:
            List[str]: List of recommendations
        """
        recommendations = []
        
        if not success_analysis.get("is_successful", False):
            recommendations.append("より具体的な検索クエリを試してください")
            recommendations.append("音声の種類や特徴を詳しく説明してください")
            recommendations.append("複数のキーワードを組み合わせてみてください")
        
        if success_analysis.get("confidence_level") == "poor":
            recommendations.append("検索結果の信頼度が低いです。別の表現を試してください")
        
        # Find best matching category
        category_matches = self._find_category_matches(similarity_scores)
        if category_matches:
            best_category = max(category_matches.items(), key=lambda x: x[1])[0]
            recommendations.append(f"「{best_category}」カテゴリの音声の可能性が高いです")
        
        return recommendations
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search engine statistics
        
        Returns:
            Dict[str, Any]: Search engine statistics
        """
        return {
            "total_categories": len(self.audio_categories),
            "total_queries": sum(len(queries) for queries in self.audio_categories.values()),
            "success_thresholds": self.success_thresholds,
            "categories": list(self.audio_categories.keys())
        } 