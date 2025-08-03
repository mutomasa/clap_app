"""
CLAP (Contrastive Language-Audio Pretraining) Application
Streamlit-based web application for audio-text understanding using CLAP models.
"""

import streamlit as st
import time
import os
import tempfile
import pandas as pd
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from clap_model import CLAPModelManager, AudioAnalyzer
from visualization_manager import CLAPVisualizationManager
from sample_audio_manager import SampleAudioManager

# Page configuration
st.set_page_config(
    page_title="CLAP Audio-Text Understanding",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    .feature-card:hover {
        border-color: #667eea;
        transform: translateY(-2px);
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #5a6fd8 0%, #6a4190 100%);
    }
    .audio-player {
        margin: 1rem 0;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
    }
    .similarity-score {
        font-size: 1.2em;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .high-similarity {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .medium-similarity {
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    .low-similarity {
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)


class CLAPApp:
    """CLAP Application for audio-text understanding"""
    
    def __init__(self):
        """Initialize the application"""
        self.clap_manager = CLAPModelManager()
        self.audio_analyzer = AudioAnalyzer()
        self.viz_manager = CLAPVisualizationManager()
        self.sample_audio_manager = SampleAudioManager()
        
        # Initialize session state
        if 'audio_file' not in st.session_state:
            st.session_state.audio_file = None
        if 'audio_path' not in st.session_state:
            st.session_state.audio_path = None
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'similarity_results' not in st.session_state:
            st.session_state.similarity_results = {}
        if 'audio_features' not in st.session_state:
            st.session_state.audio_features = {}
        if 'processing_times' not in st.session_state:
            st.session_state.processing_times = {}
        if 'detection_timing_data' not in st.session_state:
            st.session_state.detection_timing_data = {}
    
    def run(self):
        """Run the application"""
        self._display_header()
        self._display_sidebar()
        self._display_main_content()
    
    def _display_header(self):
        """Display the application header"""
        st.markdown("""
        <div class="main-header">
            <h1>🎵 CLAP Audio-Text Understanding</h1>
            <p>Contrastive Language-Audio Pretraining for Audio-Text Understanding</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _display_sidebar(self):
        """Display the sidebar with model loading and audio upload"""
        st.sidebar.title("🔧 設定")
        
        # Model loading section
        st.sidebar.subheader("🤖 モデル設定")
        
        if not st.session_state.model_loaded:
            if st.sidebar.button("CLAPモデルを読み込み"):
                with st.spinner("CLAPモデルを読み込み中..."):
                    start_time = time.time()
                    success = self.clap_manager.load_model()
                    load_time = time.time() - start_time
                    
                    if success:
                        st.session_state.model_loaded = True
                        st.session_state.processing_times["Model Loading"] = load_time
                        st.sidebar.success(f"✅ モデル読み込み完了 ({load_time:.2f}秒)")
                    else:
                        st.sidebar.error("❌ モデル読み込みに失敗しました")
        else:
            st.sidebar.success("✅ モデル読み込み済み")
            
            # Model info
            model_info = self.clap_manager.get_model_info()
            st.sidebar.subheader("📊 モデル情報")
            st.sidebar.write(f"**モデル:** {model_info['model_name']}")
            st.sidebar.write(f"**デバイス:** {model_info['device']}")
            st.sidebar.write(f"**サンプルレート:** {model_info['sample_rate']} Hz")
            if 'model_parameters' in model_info:
                st.sidebar.write(f"**パラメータ数:** {model_info['model_parameters']:,}")
        
        # Audio upload section
        st.sidebar.subheader("🎵 音声ファイル")
        
        uploaded_file = st.sidebar.file_uploader(
            "音声ファイルをアップロード",
            type=['wav', 'mp3', 'flac', 'm4a', 'ogg'],
            help="サポート形式: WAV, MP3, FLAC, M4A, OGG"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                st.session_state.audio_path = tmp_file.name
                st.session_state.audio_file = uploaded_file
            
            # Validate audio file
            if st.session_state.audio_path:
                with st.spinner("音声ファイルを検証中..."):
                    validation = self.audio_analyzer.validate_audio_file(st.session_state.audio_path)
                
                # Display validation results
                if not validation["is_valid"]:
                    st.sidebar.error("❌ 音声ファイルの検証に失敗しました")
                    with st.sidebar.expander("🔍 検証エラー詳細"):
                        for error in validation["errors"]:
                            st.error(f"• {error}")
                    return
                
                # Show warnings if any
                if validation["warnings"]:
                    st.sidebar.warning("⚠️ 音声ファイルに警告があります")
                    with st.sidebar.expander("⚠️ 警告詳細"):
                        for warning in validation["warnings"]:
                            st.warning(f"• {warning}")
                
                # Display audio player
                st.sidebar.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
                
                # Audio file info
                try:
                    audio_info = self.audio_analyzer.analyze_audio_file(st.session_state.audio_path)
                    if audio_info and audio_info.get('duration', 0) > 0:
                        st.sidebar.subheader("📋 音声情報")
                        st.sidebar.write(f"**長さ:** {audio_info.get('duration', 0):.2f}秒")
                        st.sidebar.write(f"**サンプルレート:** {audio_info.get('sample_rate', 0)} Hz")
                        st.sidebar.write(f"**チャンネル数:** {audio_info.get('channels', 0)}")
                        st.sidebar.write(f"**ファイルサイズ:** {audio_info.get('file_size_mb', 0):.2f} MB")
                        
                        # Store audio features
                        st.session_state.audio_features = audio_info
                        st.sidebar.success("✅ 音声ファイルが正常に読み込まれました")
                    else:
                        st.sidebar.warning("⚠️ 音声ファイルの分析に失敗しました")
                except Exception as e:
                    st.sidebar.error(f"❌ 音声分析エラー: {str(e)}")
                    import traceback
                    with st.sidebar.expander("詳細エラー情報"):
                        st.code(traceback.format_exc())
        
        # Sample audio files
        st.sidebar.subheader("🎵 サンプル音声")
        
        # Category selection
        categories = self.sample_audio_manager.get_all_categories()
        selected_category = st.sidebar.selectbox(
            "カテゴリを選択",
            ["すべて"] + categories,
            help="音声のカテゴリを選択してください"
        )
        
        # Get audio files by category
        if selected_category == "すべて":
            available_samples = self.sample_audio_manager.get_sample_audio_list()
        else:
            available_samples = self.sample_audio_manager.get_audio_by_category(selected_category)
        
        selected_sample = st.sidebar.selectbox(
            "サンプル音声を選択",
            ["なし"] + available_samples,
            help="テスト用の音声サンプルを選択できます"
        )
        
        # Display sample info
        if selected_sample != "なし":
            sample_info = self.sample_audio_manager.get_sample_audio_info(selected_sample)
            if sample_info:
                st.sidebar.info(f"**説明:** {sample_info['description']}")
                st.sidebar.info(f"**カテゴリ:** {sample_info['category']}")
        
        if selected_sample != "なし":
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("サンプル音声を読み込み"):
                    self._load_sample_audio(selected_sample)
            with col2:
                if st.button("合成音声を作成"):
                    self._create_synthetic_audio()
        
        # Synthetic audio section
        st.sidebar.subheader("🔧 合成音声")
        synthetic_types = self.sample_audio_manager.get_synthetic_audio_types()
        selected_synthetic = st.sidebar.selectbox(
            "合成音声タイプ",
            ["なし"] + synthetic_types,
            help="テスト用の合成音声を作成できます"
        )
        
        if selected_synthetic != "なし":
            duration = st.sidebar.slider("長さ (秒)", 1.0, 10.0, 3.0, 0.5)
            if st.sidebar.button("合成音声を作成"):
                self._create_synthetic_audio(selected_synthetic, duration)
        
        # Text queries section
        st.sidebar.subheader("📝 テキストクエリ")
        
        # Custom query input
        custom_query = st.sidebar.text_input(
            "カスタムクエリを入力",
            placeholder="例: 音楽が流れている、人の話し声、鳥の鳴き声...",
            help="独自のテキストクエリを入力できます"
        )
        
        # Category-based query suggestions
        if selected_category != "すべて" and selected_category in categories:
            category_queries = self.sample_audio_manager.get_category_queries(selected_category)
            st.sidebar.subheader(f"🎯 {selected_category}関連クエリ")
            selected_category_queries = st.sidebar.multiselect(
                f"{selected_category}関連のクエリを選択",
                category_queries,
                help=f"{selected_category}カテゴリに適したクエリです"
            )
        else:
            selected_category_queries = []
        
        # General sample queries
        st.sidebar.subheader("📋 一般的なクエリ")
        general_queries = [
            "音楽が流れている",
            "人の話し声",
            "鳥の鳴き声",
            "車のエンジン音",
            "雨の音",
            "笑い声",
            "拍手",
            "ドアが閉まる音",
            "電話の着信音",
            "キーボードのタイピング音",
            "犬の鳴き声",
            "猫の鳴き声",
            "風の音",
            "波の音",
            "雷の音",
            "時計の音",
            "ドアベル",
            "アラーム音",
            "楽器の音",
            "機械の音"
        ]
        
        selected_general_queries = st.sidebar.multiselect(
            "一般的なクエリを選択",
            general_queries,
            default=general_queries[:3],
            help="複数のクエリを選択できます"
        )
        
        # Combine all queries
        all_queries = []
        if custom_query.strip():
            all_queries.append(custom_query.strip())
        all_queries.extend(selected_category_queries)
        all_queries.extend(selected_general_queries)
        
        # Display selected queries
        if all_queries:
            st.sidebar.subheader("📋 分析対象クエリ")
            for i, query in enumerate(all_queries, 1):
                st.sidebar.write(f"{i}. {query}")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("基本分析", help="基本的な音声-テキスト類似度分析"):
                if st.session_state.audio_path and st.session_state.model_loaded:
                    if all_queries:
                        self._analyze_audio_text_similarity(all_queries)
                    else:
                        st.warning("分析するクエリを入力または選択してください")
                else:
                    st.warning("音声ファイルとモデルの両方が必要です")
        
        with col2:
            if st.button("詳細分析", help="タイミング情報付きの詳細分析"):
                if st.session_state.audio_path and st.session_state.model_loaded:
                    if all_queries:
                        self._analyze_audio_text_similarity_with_timing(all_queries)
                    else:
                        st.warning("分析するクエリを入力または選択してください")
                else:
                    st.warning("音声ファイルとモデルの両方が必要です")
    
    def _display_main_content(self):
        """Display the main content area"""
        if not st.session_state.model_loaded:
            st.warning("⚠️ まずサイドバーでCLAPモデルを読み込んでください。")
            return
        
        if not st.session_state.audio_path:
            st.warning("⚠️ 音声ファイルをアップロードしてください。")
            return
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎵 音声分析", 
            "🔍 テキスト-音声マッチング", 
            "📊 可視化", 
            "⏱️ 検出タイミング分析",
            "⚙️ デバッグ情報"
        ])
        
        with tab1:
            self._display_audio_analysis_tab()
        
        with tab2:
            self._display_text_audio_matching_tab()
        
        with tab3:
            self._display_visualization_tab()
        
        with tab4:
            self._display_timing_analysis_tab()
        
        with tab5:
            self._display_debug_tab()
    
    def _display_audio_analysis_tab(self):
        """Display audio analysis tab"""
        st.header("🎵 音声分析")
        
        if st.session_state.audio_features and st.session_state.audio_path:
            try:
                # Audio features display
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📋 基本情報")
                    features = st.session_state.audio_features
                    
                    st.metric("長さ", f"{features.get('duration', 0):.2f}秒")
                    st.metric("サンプルレート", f"{features.get('sample_rate', 0)} Hz")
                    st.metric("チャンネル数", features.get('channels', 0))
                    st.metric("ファイルサイズ", f"{features.get('file_size_mb', 0):.2f} MB")
                
                with col2:
                    st.subheader("🎛️ スペクトル特徴")
                    st.metric("スペクトル重心", f"{features.get('spectral_centroid_mean', 0):.2f}")
                    st.metric("スペクトル帯域幅", f"{features.get('spectral_bandwidth_mean', 0):.2f}")
                    st.metric("ゼロクロス率", f"{features.get('zero_crossing_rate_mean', 0):.4f}")
                    st.metric("RMSエネルギー", f"{features.get('rms_mean', 0):.4f}")
                
                # Audio visualizations
                st.subheader("📈 音声可視化")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Waveform
                    st.write("**波形表示**")
                    try:
                        waveform_fig = self.viz_manager.create_audio_waveform(st.session_state.audio_path)
                        st.plotly_chart(waveform_fig, use_container_width=True, key="audio_analysis_waveform")
                    except Exception as e:
                        st.error(f"波形の表示に失敗しました: {str(e)}")
                        import traceback
                        with st.expander("詳細エラー情報"):
                            st.code(traceback.format_exc())
                
                with col2:
                    # Spectrogram
                    st.write("**スペクトログラム**")
                    try:
                        spectrogram_fig = self.viz_manager.create_spectrogram(st.session_state.audio_path)
                        st.plotly_chart(spectrogram_fig, use_container_width=True, key="audio_analysis_spectrogram")
                    except Exception as e:
                        st.error(f"スペクトログラムの表示に失敗しました: {str(e)}")
                        import traceback
                        with st.expander("詳細エラー情報"):
                            st.code(traceback.format_exc())
                
                # MFCC and Radar chart
                col1, col2 = st.columns(2)
                
                with col1:
                    # MFCC
                    st.write("**MFCC特徴量**")
                    try:
                        mfcc_fig = self.viz_manager.create_mfcc_heatmap(st.session_state.audio_path)
                        st.plotly_chart(mfcc_fig, use_container_width=True, key="audio_analysis_mfcc")
                    except Exception as e:
                        st.error(f"MFCCの表示に失敗しました: {str(e)}")
                        import traceback
                        with st.expander("詳細エラー情報"):
                            st.code(traceback.format_exc())
                
                with col2:
                    # Radar chart
                    st.write("**音声特徴レーダーチャート**")
                    try:
                        radar_fig = self.viz_manager.create_audio_features_radar(st.session_state.audio_features)
                        st.plotly_chart(radar_fig, use_container_width=True, key="audio_analysis_radar")
                    except Exception as e:
                        st.error(f"レーダーチャートの表示に失敗しました: {str(e)}")
                        import traceback
                        with st.expander("詳細エラー情報"):
                            st.code(traceback.format_exc())
                
                # Comprehensive dashboard
                st.subheader("📊 総合分析ダッシュボード")
                try:
                    dashboard_fig = self.viz_manager.create_audio_analysis_dashboard(
                        st.session_state.audio_path, 
                        st.session_state.audio_features
                    )
                    st.plotly_chart(dashboard_fig, use_container_width=True, key="audio_analysis_dashboard")
                except Exception as e:
                    st.error(f"ダッシュボードの表示に失敗しました: {str(e)}")
                    import traceback
                    with st.expander("詳細エラー情報"):
                        st.code(traceback.format_exc())
                        
            except Exception as e:
                st.error(f"音声分析の表示に失敗しました: {str(e)}")
                import traceback
                with st.expander("詳細エラー情報"):
                    st.code(traceback.format_exc())
        else:
            st.warning("⚠️ 音声ファイルが読み込まれていません。サイドバーで音声ファイルをアップロードまたはサンプル音声を選択してください。")
    
    def _display_text_audio_matching_tab(self):
        """Display text-audio matching tab"""
        st.header("🔍 テキスト-音声マッチング")
        
        # Custom text input
        st.subheader("📝 カスタムクエリ")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            custom_text = st.text_input(
                "テキストクエリを入力",
                placeholder="例: 音楽が流れている、人の話し声、鳥の鳴き声..."
            )
        
        with col2:
            if st.button("分析実行"):
                if custom_text:
                    self._analyze_single_query(custom_text)
        
        # Display similarity results
        if st.session_state.similarity_results:
            st.subheader("📊 類似度スコア")
            
            # Sort results by similarity score
            sorted_results = sorted(
                st.session_state.similarity_results.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Display results
            for text, score in sorted_results:
                # Determine similarity level
                if score >= 0.6:
                    css_class = "high-similarity"
                    emoji = "🟢"
                elif score >= 0.3:
                    css_class = "medium-similarity"
                    emoji = "🟡"
                else:
                    css_class = "low-similarity"
                    emoji = "🔴"
                
                st.markdown(f"""
                <div class="similarity-score {css_class}">
                    {emoji} <strong>{text}</strong>: {score:.3f}
                </div>
                """, unsafe_allow_html=True)
            
            # Similarity chart
            similarity_fig = self.viz_manager.create_similarity_bar_chart(st.session_state.similarity_results)
            st.plotly_chart(similarity_fig, use_container_width=True, key="matching_similarity_chart")
    
    def _display_visualization_tab(self):
        """Display visualization tab"""
        st.header("📊 可視化")
        
        if st.session_state.audio_path and st.session_state.audio_features:
            try:
                # Comprehensive dashboard
                st.subheader("🎛️ 音声分析ダッシュボード")
                try:
                    dashboard_fig = self.viz_manager.create_audio_analysis_dashboard(
                        st.session_state.audio_path,
                        st.session_state.audio_features
                    )
                    st.plotly_chart(dashboard_fig, use_container_width=True, key="visualization_dashboard")
                except Exception as e:
                    st.error(f"ダッシュボードの表示に失敗しました: {str(e)}")
                    import traceback
                    with st.expander("詳細エラー情報"):
                        st.code(traceback.format_exc())
                
                # Individual visualizations
                st.subheader("📈 個別可視化")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**波形表示**")
                    try:
                        waveform_fig = self.viz_manager.create_audio_waveform(st.session_state.audio_path)
                        st.plotly_chart(waveform_fig, use_container_width=True, key="visualization_waveform")
                    except Exception as e:
                        st.error(f"波形の表示に失敗しました: {str(e)}")
                        import traceback
                        with st.expander("詳細エラー情報"):
                            st.code(traceback.format_exc())
                
                with col2:
                    st.write("**スペクトログラム**")
                    try:
                        spectrogram_fig = self.viz_manager.create_spectrogram(st.session_state.audio_path)
                        st.plotly_chart(spectrogram_fig, use_container_width=True, key="visualization_spectrogram")
                    except Exception as e:
                        st.error(f"スペクトログラムの表示に失敗しました: {str(e)}")
                        import traceback
                        with st.expander("詳細エラー情報"):
                            st.code(traceback.format_exc())
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**MFCC特徴量**")
                    try:
                        mfcc_fig = self.viz_manager.create_mfcc_heatmap(st.session_state.audio_path)
                        st.plotly_chart(mfcc_fig, use_container_width=True, key="visualization_mfcc")
                    except Exception as e:
                        st.error(f"MFCCの表示に失敗しました: {str(e)}")
                        import traceback
                        with st.expander("詳細エラー情報"):
                            st.code(traceback.format_exc())
                
                with col2:
                    st.write("**音声特徴レーダーチャート**")
                    try:
                        radar_fig = self.viz_manager.create_audio_features_radar(st.session_state.audio_features)
                        st.plotly_chart(radar_fig, use_container_width=True, key="visualization_radar")
                    except Exception as e:
                        st.error(f"レーダーチャートの表示に失敗しました: {str(e)}")
                        import traceback
                        with st.expander("詳細エラー情報"):
                            st.code(traceback.format_exc())
                
            except Exception as e:
                st.error(f"可視化の表示に失敗しました: {str(e)}")
                import traceback
                with st.expander("詳細エラー情報"):
                    st.code(traceback.format_exc())
        else:
            st.warning("⚠️ 音声ファイルが読み込まれていません。サイドバーで音声ファイルをアップロードまたはサンプル音声を選択してください。")
    
    def _display_timing_analysis_tab(self):
        """Display timing analysis tab"""
        st.header("⏱️ 検出タイミング分析")
        
        if st.session_state.detection_timing_data:
            timing_data = st.session_state.detection_timing_data
            
            # Summary table
            st.subheader("📋 検出結果サマリー")
            summary_fig = self.viz_manager.create_detection_summary_table(timing_data)
            st.plotly_chart(summary_fig, use_container_width=True, key="timing_summary")
            
            # Detailed timing analysis
            st.subheader("📊 詳細タイミング分析")
            timing_fig = self.viz_manager.create_detection_timing_analysis(timing_data)
            st.plotly_chart(timing_fig, use_container_width=True, key="timing_analysis")
            
            # Timing details
            st.subheader("🔍 タイミング詳細")
            timing = timing_data.get("timing", {})
            audio_info = timing_data.get("audio_info", {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("総処理時間", f"{timing.get('total_time', 0):.3f}秒")
                st.metric("音声処理時間", f"{timing.get('audio_processing_time', 0):.3f}秒")
            with col2:
                st.metric("テキスト処理時間", f"{timing.get('text_processing_time', 0):.3f}秒")
                st.metric("音声長", f"{audio_info.get('duration', 0):.2f}秒")
            with col3:
                # 処理効率の計算（ゼロ除算を防ぐ）
                total_time = timing.get('total_time', 0)
                audio_duration = audio_info.get('duration', 0)
                if total_time > 0:
                    efficiency = audio_duration / total_time
                    st.metric("処理効率", f"{efficiency:.2f}x")
                else:
                    st.metric("処理効率", "N/A")
                st.metric("クエリ数", len(timing_data.get("similarities", {})))
            
            # Per query timing details
            if timing.get("per_query_times"):
                st.subheader("📈 クエリ別処理時間")
                per_query_data = timing["per_query_times"]
                
                query_df = pd.DataFrame([
                    {
                        "クエリ": query,
                        "総時間": data.get("total_time", 0),
                        "類似度計算時間": data.get("similarity_time", 0),
                        "類似度スコア": timing_data["similarities"].get(query, 0)
                    }
                    for query, data in per_query_data.items()
                ])
                
                st.dataframe(query_df, use_container_width=True)
        else:
            st.warning("⚠️ 詳細分析データがありません。サイドバーで「詳細分析」を実行してください。")
    
    def _display_debug_tab(self):
        """Display debug information tab"""
        st.header("⚙️ デバッグ情報")
        
        # Model information
        st.subheader("🤖 モデル情報")
        model_info = self.clap_manager.get_model_info()
        model_info_fig = self.viz_manager.create_model_info_display(model_info)
        st.plotly_chart(model_info_fig, use_container_width=True, key="debug_model_info")
        
        # Performance metrics
        if st.session_state.processing_times:
            st.subheader("⏱️ 処理時間")
            performance_fig = self.viz_manager.create_performance_metrics(st.session_state.processing_times)
            st.plotly_chart(performance_fig, use_container_width=True, key="debug_performance")
        
        # Audio features details
        if st.session_state.audio_features:
            st.subheader("🎵 音声特徴詳細")
            st.json(st.session_state.audio_features)
    
    def _analyze_audio_text_similarity(self, text_queries: List[str]):
        """Analyze similarity between audio and text queries
        
        Args:
            text_queries: List of text queries to analyze
        """
        if not st.session_state.audio_path or not st.session_state.model_loaded:
            return
        
        with st.spinner("音声-テキスト類似度を分析中..."):
            start_time = time.time()
            
            # Perform audio-text matching
            results = self.clap_manager.audio_text_matching(
                st.session_state.audio_path,
                text_queries
            )
            
            processing_time = time.time() - start_time
            st.session_state.processing_times["Audio-Text Matching"] = processing_time
            
            # Store results
            st.session_state.similarity_results = results
            
            st.success(f"✅ 分析完了 ({processing_time:.2f}秒)")
    
    def _analyze_single_query(self, text_query: str):
        """Analyze single text query
        
        Args:
            text_query: Single text query to analyze
        """
        if not st.session_state.audio_path or not st.session_state.model_loaded:
            return
        
        with st.spinner("単一クエリを分析中..."):
            start_time = time.time()
            
            # Perform audio-text matching
            results = self.clap_manager.audio_text_matching(
                st.session_state.audio_path,
                [text_query]
            )
            
            processing_time = time.time() - start_time
            st.session_state.processing_times["Single Query Analysis"] = processing_time
            
            # Update results
            st.session_state.similarity_results.update(results)
            
            st.success(f"✅ 分析完了 ({processing_time:.2f}秒)")
    
    def _analyze_audio_text_similarity_with_timing(self, text_queries: List[str]):
        """Analyze similarity between audio and text queries with detailed timing
        
        Args:
            text_queries: List of text queries to analyze
        """
        if not st.session_state.audio_path or not st.session_state.model_loaded:
            return
        
        with st.spinner("音声-テキスト類似度を詳細分析中..."):
            # Perform audio-text matching with timing
            results = self.clap_manager.audio_text_matching_with_timing(
                st.session_state.audio_path,
                text_queries
            )
            
            # Store results
            st.session_state.detection_timing_data = results
            st.session_state.similarity_results = results.get("similarities", {})
            
            # Store processing times
            timing = results.get("timing", {})
            st.session_state.processing_times.update({
                "Total Detection Time": timing.get("total_time", 0),
                "Audio Processing": timing.get("audio_processing_time", 0),
                "Text Processing": timing.get("text_processing_time", 0)
            })
            
            st.success(f"✅ 詳細分析完了 (総時間: {timing.get('total_time', 0):.2f}秒)")
    
    def _load_sample_audio(self, audio_name: str):
        """Load sample audio file
        
        Args:
            audio_name: Name of the sample audio to load
        """
        try:
            with st.spinner(f"{audio_name} をダウンロード中..."):
                audio_path = self.sample_audio_manager.download_sample_audio(audio_name)
                
                if audio_path:
                    st.session_state.audio_path = audio_path
                    
                    # Load audio for display
                    import requests
                    sample_info = self.sample_audio_manager.get_sample_audio_info(audio_name)
                    if sample_info:
                        response = requests.get(sample_info["url"])
                        if response.status_code == 200:
                            st.sidebar.audio(response.content, format="audio/wav")
                    
                    # Update audio features
                    audio_info = self.audio_analyzer.analyze_audio_file(audio_path)
                    if audio_info:
                        st.session_state.audio_features = audio_info
                    
                    st.sidebar.success(f"✅ {audio_name} を読み込みました！")
                else:
                    st.sidebar.error(f"❌ {audio_name} の読み込みに失敗しました")
        except Exception as e:
            st.sidebar.error(f"❌ エラー: {str(e)}")
    
    def _create_synthetic_audio(self, audio_type: str = None, duration: float = 3.0):
        """Create synthetic audio for testing
        
        Args:
            audio_type: Type of synthetic audio to create
            duration: Duration in seconds
        """
        if audio_type is None:
            audio_type = "sine_wave"
        
        try:
            with st.spinner(f"{audio_type} を作成中..."):
                audio_path = self.sample_audio_manager.create_synthetic_audio(
                    audio_type, duration
                )
                
                if audio_path:
                    st.session_state.audio_path = audio_path
                    
                    # Load audio for display
                    with open(audio_path, 'rb') as f:
                        audio_data = f.read()
                    st.sidebar.audio(audio_data, format="audio/wav")
                    
                    # Update audio features
                    audio_info = self.audio_analyzer.analyze_audio_file(audio_path)
                    if audio_info:
                        st.session_state.audio_features = audio_info
                    
                    st.sidebar.success(f"✅ {audio_type} を作成しました！")
                else:
                    st.sidebar.error(f"❌ {audio_type} の作成に失敗しました")
        except Exception as e:
            st.sidebar.error(f"❌ エラー: {str(e)}")


def main():
    """Main function"""
    app = CLAPApp()
    app.run()


if __name__ == "__main__":
    main() 