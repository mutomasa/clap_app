"""
Visualization Manager for CLAP Application
Handles creation of various visualizations for audio-text analysis.
"""

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import librosa
import librosa.display
from typing import Dict, List, Any, Optional, Tuple
import warnings
import os
import traceback
warnings.filterwarnings('ignore')


class CLAPVisualizationManager:
    """Visualization manager for CLAP application"""
    
    def __init__(self):
        """Initialize visualization manager"""
        self.colors = px.colors.qualitative.Set3
        self.default_figsize = (12, 8)
    
    def create_audio_waveform(self, audio_path: str, title: str = "Audio Waveform") -> go.Figure:
        """Create audio waveform visualization
        
        Args:
            audio_path: Path to audio file
            title: Plot title
            
        Returns:
            go.Figure: Plotly figure
        """
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                return self._create_error_figure(f"Audio file not found: {audio_path}")
            
            # Check if audio is empty
            if len(audio_path) == 0:
                return self._create_error_figure("Audio file path is empty")
            
            # Load audio with multiple methods
            audio = None
            sr = None
            load_errors = []
            
            # Method 1: librosa
            try:
                audio, sr = librosa.load(audio_path, sr=None)
                if len(audio) > 0:
                    print(f"Successfully loaded audio with librosa: {audio_path}")
            except Exception as e:
                load_errors.append(f"librosa: {str(e)}")
                
                # Method 2: soundfile
                try:
                    import soundfile as sf
                    audio, sr = sf.read(audio_path)
                    if len(audio.shape) > 1:
                        audio = np.mean(audio, axis=1)  # Convert to mono
                    if len(audio) > 0:
                        print(f"Successfully loaded audio with soundfile: {audio_path}")
                except Exception as e2:
                    load_errors.append(f"soundfile: {str(e2)}")
                    
                    # Method 3: pydub
                    try:
                        from pydub import AudioSegment
                        audio_segment = AudioSegment.from_file(audio_path)
                        audio = np.array(audio_segment.get_array_of_samples())
                        sr = audio_segment.frame_rate
                        # Convert to mono if stereo
                        if audio_segment.channels == 2:
                            audio = audio.reshape((-1, 2))
                            audio = np.mean(audio, axis=1)
                        if len(audio) > 0:
                            print(f"Successfully loaded audio with pydub: {audio_path}")
                    except Exception as e3:
                        load_errors.append(f"pydub: {str(e3)}")
                        error_msg = f"All audio loading methods failed: {', '.join(load_errors)}"
                        print(error_msg)
                        return self._create_error_figure(error_msg)
            
            # Check if audio is valid
            if audio is None or len(audio) == 0:
                return self._create_error_figure("Audio file is empty or could not be loaded")
            
            # Create time axis
            time = np.linspace(0, len(audio) / sr, len(audio))
            
            # Create figure
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=time,
                y=audio,
                mode='lines',
                name='Waveform',
                line=dict(color='#1f77b4', width=1),
                fill='tonexty',
                fillcolor='rgba(31, 119, 180, 0.3)'
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Time (seconds)",
                yaxis_title="Amplitude",
                template="plotly_white",
                height=400,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            error_msg = f"Error creating waveform: {str(e)}\nTraceback: {traceback.format_exc()}"
            print(error_msg)
            return self._create_error_figure("Error creating waveform")
    
    def create_spectrogram(self, audio_path: str, title: str = "Spectrogram") -> go.Figure:
        """Create spectrogram visualization
        
        Args:
            audio_path: Path to audio file
            title: Plot title
            
        Returns:
            go.Figure: Plotly figure
        """
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                return self._create_error_figure(f"Audio file not found: {audio_path}")
            
            # Check if audio is empty
            if len(audio_path) == 0:
                return self._create_error_figure("Audio file path is empty")
            
            # Load audio with multiple methods
            audio = None
            sr = None
            load_errors = []
            
            # Method 1: librosa
            try:
                audio, sr = librosa.load(audio_path, sr=None)
                if len(audio) > 0:
                    print(f"Successfully loaded audio with librosa: {audio_path}")
            except Exception as e:
                load_errors.append(f"librosa: {str(e)}")
                
                # Method 2: soundfile
                try:
                    import soundfile as sf
                    audio, sr = sf.read(audio_path)
                    if len(audio.shape) > 1:
                        audio = np.mean(audio, axis=1)  # Convert to mono
                    if len(audio) > 0:
                        print(f"Successfully loaded audio with soundfile: {audio_path}")
                except Exception as e2:
                    load_errors.append(f"soundfile: {str(e2)}")
                    
                    # Method 3: pydub
                    try:
                        from pydub import AudioSegment
                        audio_segment = AudioSegment.from_file(audio_path)
                        audio = np.array(audio_segment.get_array_of_samples())
                        sr = audio_segment.frame_rate
                        # Convert to mono if stereo
                        if audio_segment.channels == 2:
                            audio = audio.reshape((-1, 2))
                            audio = np.mean(audio, axis=1)
                        if len(audio) > 0:
                            print(f"Successfully loaded audio with pydub: {audio_path}")
                    except Exception as e3:
                        load_errors.append(f"pydub: {str(e3)}")
                        error_msg = f"All audio loading methods failed: {', '.join(load_errors)}"
                        print(error_msg)
                        return self._create_error_figure(error_msg)
            
            # Check if audio is valid
            if audio is None or len(audio) == 0:
                return self._create_error_figure("Audio file is empty or could not be loaded")
            
            # Compute spectrogram
            D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
            
            # Create figure
            fig = go.Figure(data=go.Heatmap(
                z=D,
                colorscale='Viridis',
                x=np.linspace(0, len(audio) / sr, D.shape[1]),
                y=np.linspace(0, sr / 2, D.shape[0]),
                colorbar=dict(title="dB")
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Time (seconds)",
                yaxis_title="Frequency (Hz)",
                template="plotly_white",
                height=400
            )
            
            return fig
            
        except Exception as e:
            error_msg = f"Error creating spectrogram: {str(e)}\nTraceback: {traceback.format_exc()}"
            print(error_msg)
            return self._create_error_figure("Error creating spectrogram")
    
    def create_similarity_bar_chart(self, similarity_scores: Dict[str, float], title: str = "Text-Audio Similarity Scores") -> go.Figure:
        """Create similarity scores bar chart
        
        Args:
            similarity_scores: Dictionary of text queries and their similarity scores
            title: Plot title
            
        Returns:
            go.Figure: Plotly figure
        """
        try:
            if not similarity_scores:
                return self._create_error_figure("No similarity scores provided")
            
            # Prepare data
            queries = list(similarity_scores.keys())
            scores = list(similarity_scores.values())
            
            # Create color mapping based on scores
            colors = []
            for score in scores:
                if score >= 0.6:
                    colors.append('#2E8B57')  # Green for high similarity
                elif score >= 0.3:
                    colors.append('#FFD700')  # Yellow for medium similarity
                else:
                    colors.append('#DC143C')  # Red for low similarity
            
            fig = go.Figure(data=go.Bar(
                x=queries,
                y=scores,
                marker_color=colors,
                text=[f'{score:.3f}' for score in scores],
                textposition='auto'
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Text Queries",
                yaxis_title="Similarity Score",
                template="plotly_white",
                height=400,
                yaxis=dict(range=[0, 1])
            )
            
            return fig
            
        except Exception as e:
            error_msg = f"Error creating similarity chart: {str(e)}\nTraceback: {traceback.format_exc()}"
            print(error_msg)
            return self._create_error_figure("Error creating similarity chart")
    
    def create_audio_features_radar(self, audio_features: Dict[str, Any]) -> go.Figure:
        """Create radar chart for audio features
        
        Args:
            audio_features: Dictionary of audio features
            
        Returns:
            go.Figure: Plotly figure
        """
        try:
            if not audio_features:
                return self._create_error_figure("No audio features provided")
            
            # Select features for radar chart
            feature_names = [
                "spectral_centroid_mean",
                "spectral_bandwidth_mean", 
                "zero_crossing_rate_mean",
                "rms_mean"
            ]
            
            feature_values = []
            valid_features = []
            
            for feature in feature_names:
                if feature in audio_features and audio_features[feature] is not None:
                    value = float(audio_features[feature])
                    if not np.isnan(value) and not np.isinf(value):
                        feature_values.append(value)
                        valid_features.append(feature.replace('_', ' ').title())
            
            if not valid_features:
                return self._create_error_figure("No valid audio features found")
            
            # Normalize values to 0-1 range
            if feature_values:
                min_val = min(feature_values)
                max_val = max(feature_values)
                if max_val > min_val:
                    normalized_values = [(v - min_val) / (max_val - min_val) for v in feature_values]
                else:
                    normalized_values = [0.5] * len(feature_values)
            else:
                normalized_values = [0.5] * len(valid_features)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=normalized_values,
                theta=valid_features,
                fill='toself',
                name='Audio Features',
                line_color='#1f77b4'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=False,
                title="Audio Features Radar Chart",
                template="plotly_white",
                height=400
            )
            
            return fig
            
        except Exception as e:
            error_msg = f"Error creating radar chart: {str(e)}\nTraceback: {traceback.format_exc()}"
            print(error_msg)
            return self._create_error_figure("Error creating radar chart")
    
    def create_mfcc_heatmap(self, audio_path: str, title: str = "MFCC Features") -> go.Figure:
        """Create MFCC heatmap visualization
        
        Args:
            audio_path: Path to audio file
            title: Plot title
            
        Returns:
            go.Figure: Plotly figure
        """
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                return self._create_error_figure(f"Audio file not found: {audio_path}")
            
            # Check if audio is empty
            if len(audio_path) == 0:
                return self._create_error_figure("Audio file path is empty")
            
            # Load audio with multiple methods
            audio = None
            sr = None
            load_errors = []
            
            # Method 1: librosa
            try:
                audio, sr = librosa.load(audio_path, sr=None)
                if len(audio) > 0:
                    print(f"Successfully loaded audio with librosa: {audio_path}")
            except Exception as e:
                load_errors.append(f"librosa: {str(e)}")
                
                # Method 2: soundfile
                try:
                    import soundfile as sf
                    audio, sr = sf.read(audio_path)
                    if len(audio.shape) > 1:
                        audio = np.mean(audio, axis=1)  # Convert to mono
                    if len(audio) > 0:
                        print(f"Successfully loaded audio with soundfile: {audio_path}")
                except Exception as e2:
                    load_errors.append(f"soundfile: {str(e2)}")
                    
                    # Method 3: pydub
                    try:
                        from pydub import AudioSegment
                        audio_segment = AudioSegment.from_file(audio_path)
                        audio = np.array(audio_segment.get_array_of_samples())
                        sr = audio_segment.frame_rate
                        # Convert to mono if stereo
                        if audio_segment.channels == 2:
                            audio = audio.reshape((-1, 2))
                            audio = np.mean(audio, axis=1)
                        if len(audio) > 0:
                            print(f"Successfully loaded audio with pydub: {audio_path}")
                    except Exception as e3:
                        load_errors.append(f"pydub: {str(e3)}")
                        error_msg = f"All audio loading methods failed: {', '.join(load_errors)}"
                        print(error_msg)
                        return self._create_error_figure(error_msg)
            
            # Check if audio is valid
            if audio is None or len(audio) == 0:
                return self._create_error_figure("Audio file is empty or could not be loaded")
            
            # Compute MFCC
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            # Check if MFCC computation was successful
            if mfccs.size == 0:
                return self._create_error_figure("MFCC computation failed")
            
            # Create figure
            fig = go.Figure(data=go.Heatmap(
                z=mfccs,
                colorscale='RdBu',
                x=np.linspace(0, len(audio) / sr, mfccs.shape[1]),
                y=[f"MFCC {i+1}" for i in range(mfccs.shape[0])],
                colorbar=dict(title="MFCC Value")
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Time (seconds)",
                yaxis_title="MFCC Coefficients",
                template="plotly_white",
                height=400
            )
            
            return fig
            
        except Exception as e:
            error_msg = f"Error creating MFCC heatmap: {str(e)}\nTraceback: {traceback.format_exc()}"
            print(error_msg)
            return self._create_error_figure("Error creating MFCC heatmap")
    
    def create_audio_analysis_dashboard(self, audio_path: str, audio_features: Dict[str, Any]) -> go.Figure:
        """Create comprehensive audio analysis dashboard
        
        Args:
            audio_path: Path to audio file
            audio_features: Dictionary of audio features
            
        Returns:
            go.Figure: Plotly figure with subplots
        """
        try:
            # Check if file exists
            if not os.path.exists(audio_path):
                return self._create_error_figure(f"Audio file not found: {audio_path}")
            
            # Check if audio is empty
            if len(audio_path) == 0:
                return self._create_error_figure("Audio file path is empty")
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Waveform', 'Spectrogram', 'MFCC Features', 'Audio Features'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"type": "polar"}]]
            )
            
            # Load audio with multiple methods
            audio = None
            sr = None
            load_errors = []
            
            # Method 1: librosa
            try:
                audio, sr = librosa.load(audio_path, sr=None)
                if len(audio) > 0:
                    print(f"Successfully loaded audio with librosa: {audio_path}")
            except Exception as e:
                load_errors.append(f"librosa: {str(e)}")
                
                # Method 2: soundfile
                try:
                    import soundfile as sf
                    audio, sr = sf.read(audio_path)
                    if len(audio.shape) > 1:
                        audio = np.mean(audio, axis=1)  # Convert to mono
                    if len(audio) > 0:
                        print(f"Successfully loaded audio with soundfile: {audio_path}")
                except Exception as e2:
                    load_errors.append(f"soundfile: {str(e2)}")
                    
                    # Method 3: pydub
                    try:
                        from pydub import AudioSegment
                        audio_segment = AudioSegment.from_file(audio_path)
                        audio = np.array(audio_segment.get_array_of_samples())
                        sr = audio_segment.frame_rate
                        # Convert to mono if stereo
                        if audio_segment.channels == 2:
                            audio = audio.reshape((-1, 2))
                            audio = np.mean(audio, axis=1)
                        if len(audio) > 0:
                            print(f"Successfully loaded audio with pydub: {audio_path}")
                    except Exception as e3:
                        load_errors.append(f"pydub: {str(e3)}")
                        error_msg = f"All audio loading methods failed: {', '.join(load_errors)}"
                        print(error_msg)
                        return self._create_error_figure(error_msg)
            
            # Check if audio is valid
            if audio is None or len(audio) == 0:
                return self._create_error_figure("Audio file is empty or could not be loaded")
            
            # 1. Waveform
            time = np.linspace(0, len(audio) / sr, len(audio))
            fig.add_trace(
                go.Scatter(x=time, y=audio, mode='lines', name='Waveform', line=dict(color='#1f77b4')),
                row=1, col=1
            )
            
            # 2. Spectrogram
            D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
            fig.add_trace(
                go.Heatmap(
                    z=D,
                    colorscale='Viridis',
                    x=np.linspace(0, len(audio) / sr, D.shape[1]),
                    y=np.linspace(0, sr / 2, D.shape[0]),
                    showscale=False
                ),
                row=1, col=2
            )
            
            # 3. MFCC
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            fig.add_trace(
                go.Heatmap(
                    z=mfccs,
                    colorscale='RdBu',
                    x=np.linspace(0, len(audio) / sr, mfccs.shape[1]),
                    y=[f"MFCC {i+1}" for i in range(mfccs.shape[0])],
                    showscale=False
                ),
                row=2, col=1
            )
            
            # 4. Audio features radar
            feature_names = ["spectral_centroid_mean", "spectral_bandwidth_mean", "zero_crossing_rate_mean", "rms_mean"]
            feature_values = []
            valid_features = []
            
            for feature in feature_names:
                if feature in audio_features and audio_features[feature] is not None:
                    value = float(audio_features[feature])
                    if not np.isnan(value) and not np.isinf(value):
                        feature_values.append(value)
                        valid_features.append(feature.replace('_', ' ').title())
            
            if valid_features:
                # Normalize values
                min_val = min(feature_values)
                max_val = max(feature_values)
                if max_val > min_val:
                    normalized_values = [(v - min_val) / (max_val - min_val) for v in feature_values]
                else:
                    normalized_values = [0.5] * len(feature_values)
                
                fig.add_trace(
                    go.Scatterpolar(
                        r=normalized_values,
                        theta=valid_features,
                        fill='toself',
                        name='Audio Features',
                        line_color='#1f77b4'
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                title="Audio Analysis Dashboard",
                template="plotly_white",
                height=600,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            error_msg = f"Error creating dashboard: {str(e)}\nTraceback: {traceback.format_exc()}"
            print(error_msg)
            return self._create_error_figure("Error creating dashboard")
    
    def create_model_info_display(self, model_info: Dict[str, Any]) -> go.Figure:
        """Create model information display
        
        Args:
            model_info: Dictionary of model information
            
        Returns:
            go.Figure: Plotly figure
        """
        try:
            if not model_info:
                return self._create_error_figure("No model information provided")
            
            # Prepare table data
            headers = ["Property", "Value"]
            values = []
            
            for key, value in model_info.items():
                if value is not None:
                    values.append([key.replace('_', ' ').title(), str(value)])
            
            if not values:
                return self._create_error_figure("No valid model information found")
            
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=headers,
                    fill_color='#1f77b4',
                    font=dict(color='white', size=14),
                    align='left'
                ),
                cells=dict(
                    values=list(zip(*values)),
                    fill_color='lavender',
                    align='left',
                    font=dict(size=12)
                )
            )])
            
            fig.update_layout(
                title="CLAP Model Information",
                template="plotly_white",
                height=400
            )
            
            return fig
            
        except Exception as e:
            error_msg = f"Error creating model info display: {str(e)}\nTraceback: {traceback.format_exc()}"
            print(error_msg)
            return self._create_error_figure("Error creating model info display")
    
    def create_performance_metrics(self, processing_times: Dict[str, float]) -> go.Figure:
        """Create performance metrics visualization
        
        Args:
            processing_times: Dictionary of processing times
            
        Returns:
            go.Figure: Plotly figure
        """
        try:
            if not processing_times:
                return self._create_error_figure("No processing times provided")
            
            # Prepare data
            operations = list(processing_times.keys())
            times = list(processing_times.values())
            
            fig = go.Figure(data=go.Bar(
                x=operations,
                y=times,
                marker_color='#1f77b4',
                text=[f'{time:.3f}s' for time in times],
                textposition='auto'
            ))
            
            fig.update_layout(
                title="Processing Time Analysis",
                xaxis_title="Operations",
                yaxis_title="Time (seconds)",
                template="plotly_white",
                height=400
            )
            
            return fig
            
        except Exception as e:
            error_msg = f"Error creating performance metrics: {str(e)}\nTraceback: {traceback.format_exc()}"
            print(error_msg)
            return self._create_error_figure("Error creating performance metrics")
    
    def create_detection_timing_analysis(self, timing_data: Dict[str, Any]) -> go.Figure:
        """Create detection timing analysis visualization
        
        Args:
            timing_data: Dictionary containing timing analysis data
            
        Returns:
            go.Figure: Plotly figure
        """
        try:
            if not timing_data:
                return self._create_error_figure("No timing data provided")
            
            similarities = timing_data.get("similarities", {})
            timing = timing_data.get("timing", {})
            
            if not similarities:
                return self._create_error_figure("No similarity scores found in timing data")
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Similarity Scores', 'Processing Times', 'Efficiency Analysis', 'Detection Summary'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"type": "table"}]]
            )
            
            # 1. Similarity scores
            queries = list(similarities.keys())
            scores = list(similarities.values())
            
            colors = []
            for score in scores:
                if score >= 0.6:
                    colors.append('#2E8B57')
                elif score >= 0.3:
                    colors.append('#FFD700')
                else:
                    colors.append('#DC143C')
            
            fig.add_trace(
                go.Bar(
                    x=queries,
                    y=scores,
                    marker_color=colors,
                    name='Similarity Scores'
                ),
                row=1, col=1
            )
            
            # 2. Processing times
            per_query_times = timing.get("per_query_times", {})
            if per_query_times:
                query_times = []
                time_values = []
                
                for query in queries:
                    if query in per_query_times:
                        query_times.append(query)
                        time_values.append(per_query_times[query].get("total_time", 0))
                
                if query_times:
                    fig.add_trace(
                        go.Bar(
                            x=query_times,
                            y=time_values,
                            marker_color='#FF6B6B',
                            name='Processing Times'
                        ),
                        row=1, col=2
                    )
            
            # 3. Efficiency analysis (similarity vs time)
            if per_query_times:
                efficiency_data = []
                for query in queries:
                    if query in per_query_times:
                        score = similarities[query]
                        time_val = per_query_times[query].get("total_time", 0)
                        if time_val > 0:
                            efficiency = score / time_val
                            efficiency_data.append([query, score, time_val, efficiency])
                
                if efficiency_data:
                    efficiency_df = pd.DataFrame(efficiency_data, columns=['Query', 'Score', 'Time', 'Efficiency'])
                    
                    fig.add_trace(
                        go.Scatter(
                            x=efficiency_df['Time'],
                            y=efficiency_df['Score'],
                            mode='markers+text',
                            text=efficiency_df['Query'],
                            textposition="top center",
                            marker=dict(size=10, color=efficiency_df['Efficiency'], colorscale='Viridis'),
                            name='Efficiency'
                        ),
                        row=2, col=1
                    )
            
            # 4. Summary table
            summary_data = []
            for query in queries:
                score = similarities[query]
                time_val = per_query_times.get(query, {}).get("total_time", 0) if per_query_times else 0
                
                if score >= 0.6:
                    status = "🟢 Strong"
                elif score >= 0.3:
                    status = "🟡 Weak"
                else:
                    status = "🔴 None"
                
                summary_data.append([query, f"{score:.3f}", f"{time_val:.3f}s", status])
            
            if summary_data:
                fig.add_trace(
                    go.Table(
                        header=dict(
                            values=["Query", "Score", "Time", "Detection"],
                            fill_color='#1f77b4',
                            font=dict(color='white', size=12),
                            align='left'
                        ),
                        cells=dict(
                            values=list(zip(*summary_data)),
                            fill_color='lavender',
                            align='left',
                            font=dict(size=10)
                        )
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                title="Detection Timing Analysis",
                template="plotly_white",
                height=600,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            error_msg = f"Error creating timing analysis: {str(e)}\nTraceback: {traceback.format_exc()}"
            print(error_msg)
            return self._create_error_figure("Error creating timing analysis")
    
    def create_detection_summary_table(self, timing_data: Dict[str, Any]) -> go.Figure:
        """Create detection summary table
        
        Args:
            timing_data: Dictionary containing timing analysis data
            
        Returns:
            go.Figure: Plotly table figure
        """
        try:
            similarities = timing_data.get("similarities", {})
            timing = timing_data.get("timing", {})
            audio_info = timing_data.get("audio_info", {})
            
            if not similarities:
                return self._create_error_figure("No similarity scores found")
            
            # Prepare table data
            headers = ["Query", "Similarity Score", "Processing Time", "Detection Status"]
            values = []
            
            for query, score in similarities.items():
                query_time = timing.get("per_query_times", {}).get(query, {}).get("total_time", 0)
                
                # Determine detection status
                if score >= 0.6:
                    status = "🟢 Strong Detection"
                elif score >= 0.3:
                    status = "🟡 Weak Detection"
                else:
                    status = "🔴 No Detection"
                
                values.append([
                    query,
                    f"{score:.3f}",
                    f"{query_time:.3f}s",
                    status
                ])
            
            # Add summary row
            total_time = timing.get("total_time", 0)
            audio_duration = audio_info.get("duration", 0)
            avg_similarity = np.mean(list(similarities.values())) if similarities else 0
            
            # 処理効率の計算（ゼロ除算を防ぐ）
            if total_time > 0:
                efficiency = f"{audio_duration / total_time:.2f}x"
            else:
                efficiency = "N/A"
            
            values.append([
                "**SUMMARY**",
                f"{avg_similarity:.3f}",
                f"{total_time:.3f}s",
                f"Audio: {audio_duration:.2f}s | Efficiency: {efficiency}"
            ])
            
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=headers,
                    fill_color='#1f77b4',
                    font=dict(color='white', size=14),
                    align='left'
                ),
                cells=dict(
                    values=list(zip(*values)),
                    fill_color='lavender',
                    align='left',
                    font=dict(size=12)
                )
            )])
            
            fig.update_layout(
                title="Audio Detection Results Summary",
                template="plotly_white",
                height=400
            )
            
            return fig
            
        except Exception as e:
            error_msg = f"Error creating summary table: {str(e)}\nTraceback: {traceback.format_exc()}"
            print(error_msg)
            return self._create_error_figure("Error creating summary table")
    
    def _create_error_figure(self, error_message: str) -> go.Figure:
        """Create error figure when visualization fails
        
        Args:
            error_message: Error message to display
            
        Returns:
            go.Figure: Error figure
        """
        fig = go.Figure()
        
        # Create a more informative error display
        fig.add_annotation(
            text=f"❌ エラーが発生しました<br><br><b>詳細:</b><br>{error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
            bgcolor="rgba(255, 240, 240, 0.9)",
            bordercolor="red",
            borderwidth=2,
            align="center"
        )
        
        # Add helpful information
        fig.add_annotation(
            text="💡 ヒント:<br>• 音声ファイルが正しい形式か確認してください<br>• ファイルサイズが適切か確認してください<br>• 破損していないか確認してください",
            xref="paper", yref="paper",
            x=0.5, y=0.2,
            showarrow=False,
            font=dict(size=12, color="blue"),
            bgcolor="rgba(240, 248, 255, 0.8)",
            bordercolor="blue",
            borderwidth=1,
            align="center"
        )
        
        fig.update_layout(
            template="plotly_white",
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            title="可視化エラー",
            title_font=dict(size=16, color="red")
        )
        return fig 