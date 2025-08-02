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
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            
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
            print(f"Error creating waveform: {str(e)}")
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
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            
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
            print(f"Error creating spectrogram: {str(e)}")
            return self._create_error_figure("Error creating spectrogram")
    
    def create_similarity_bar_chart(self, similarity_scores: Dict[str, float], title: str = "Text-Audio Similarity Scores") -> go.Figure:
        """Create bar chart for similarity scores
        
        Args:
            similarity_scores: Dictionary of text queries and their similarity scores
            title: Plot title
            
        Returns:
            go.Figure: Plotly figure
        """
        try:
            # Prepare data
            texts = list(similarity_scores.keys())
            scores = list(similarity_scores.values())
            
            # Create color scale based on scores
            colors = ['red' if score < 0.3 else 'orange' if score < 0.6 else 'green' for score in scores]
            
            fig = go.Figure(data=go.Bar(
                x=texts,
                y=scores,
                marker_color=colors,
                text=[f"{score:.3f}" for score in scores],
                textposition='auto',
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
            print(f"Error creating similarity chart: {str(e)}")
            return self._create_error_figure("Error creating similarity chart")
    
    def create_audio_features_radar(self, audio_features: Dict[str, Any]) -> go.Figure:
        """Create radar chart for audio features
        
        Args:
            audio_features: Dictionary of audio features
            
        Returns:
            go.Figure: Plotly figure
        """
        try:
            # Select features for radar chart
            feature_names = [
                "spectral_centroid_mean",
                "spectral_bandwidth_mean", 
                "zero_crossing_rate_mean",
                "rms_mean"
            ]
            
            feature_values = []
            for feature in feature_names:
                if feature in audio_features:
                    # Normalize values to 0-1 range
                    value = audio_features[feature]
                    if feature == "spectral_centroid_mean":
                        value = min(value / 2000, 1.0)  # Normalize to typical range
                    elif feature == "spectral_bandwidth_mean":
                        value = min(value / 1000, 1.0)
                    elif feature == "zero_crossing_rate_mean":
                        value = min(value * 10, 1.0)
                    elif feature == "rms_mean":
                        value = min(value * 5, 1.0)
                    feature_values.append(value)
                else:
                    feature_values.append(0)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=feature_values,
                theta=feature_names,
                fill='toself',
                name='Audio Features',
                line_color='#1f77b4',
                fillcolor='rgba(31, 119, 180, 0.3)'
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
            print(f"Error creating radar chart: {str(e)}")
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
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            
            # Compute MFCC
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
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
            print(f"Error creating MFCC heatmap: {str(e)}")
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
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Waveform', 'Spectrogram', 'MFCC Features', 'Audio Features'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"type": "polar"}]]
            )
            
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None)
            
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
            for feature in feature_names:
                if feature in audio_features:
                    value = audio_features[feature]
                    if feature == "spectral_centroid_mean":
                        value = min(value / 2000, 1.0)
                    elif feature == "spectral_bandwidth_mean":
                        value = min(value / 1000, 1.0)
                    elif feature == "zero_crossing_rate_mean":
                        value = min(value * 10, 1.0)
                    elif feature == "rms_mean":
                        value = min(value * 5, 1.0)
                    feature_values.append(value)
                else:
                    feature_values.append(0)
            
            fig.add_trace(
                go.Scatterpolar(
                    r=feature_values,
                    theta=feature_names,
                    fill='toself',
                    name='Audio Features',
                    line_color='#1f77b4',
                    fillcolor='rgba(31, 119, 180, 0.3)'
                ),
                row=2, col=2
            )
            
            # Update layout
            fig.update_layout(
                title="Audio Analysis Dashboard",
                template="plotly_white",
                height=800,
                showlegend=False
            )
            
            # Update axes labels
            fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
            fig.update_yaxes(title_text="Amplitude", row=1, col=1)
            fig.update_xaxes(title_text="Time (seconds)", row=1, col=2)
            fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=2)
            fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
            fig.update_yaxes(title_text="MFCC Coefficients", row=2, col=1)
            
            return fig
            
        except Exception as e:
            print(f"Error creating dashboard: {str(e)}")
            return self._create_error_figure("Error creating dashboard")
    
    def create_model_info_display(self, model_info: Dict[str, Any]) -> go.Figure:
        """Create model information display
        
        Args:
            model_info: Dictionary of model information
            
        Returns:
            go.Figure: Plotly figure
        """
        try:
            # Create table-like display
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=['Property', 'Value'],
                    fill_color='#1f77b4',
                    font=dict(color='white', size=14),
                    align='left'
                ),
                cells=dict(
                    values=[
                        list(model_info.keys()),
                        [str(v) for v in model_info.values()]
                    ],
                    fill_color='lavender',
                    align='left',
                    font=dict(size=12)
                )
            )])
            
            fig.update_layout(
                title="Model Information",
                template="plotly_white",
                height=400
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating model info display: {str(e)}")
            return self._create_error_figure("Error creating model info display")
    
    def create_performance_metrics(self, processing_times: Dict[str, float]) -> go.Figure:
        """Create performance metrics visualization
        
        Args:
            processing_times: Dictionary of processing times
            
        Returns:
            go.Figure: Plotly figure
        """
        try:
            fig = go.Figure(data=go.Bar(
                x=list(processing_times.keys()),
                y=list(processing_times.values()),
                marker_color='#ff7f0e',
                text=[f"{time:.3f}s" for time in processing_times.values()],
                textposition='auto',
            ))
            
            fig.update_layout(
                title="Processing Performance",
                xaxis_title="Operation",
                yaxis_title="Time (seconds)",
                template="plotly_white",
                height=400
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating performance metrics: {str(e)}")
            return self._create_error_figure("Error creating performance metrics")
    
    def _create_error_figure(self, error_message: str) -> go.Figure:
        """Create error figure when visualization fails
        
        Args:
            error_message: Error message to display
            
        Returns:
            go.Figure: Error figure
        """
        fig = go.Figure()
        fig.add_annotation(
            text=error_message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            template="plotly_white",
            height=400,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig 