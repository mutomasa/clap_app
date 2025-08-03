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
        
        # Initialize audio search engine
        self.search_engine = None  # Will be initialized after model loading
        
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
        if 'search_results' not in st.session_state:
            st.session_state.search_results = {}
        if 'search_engine_initialized' not in st.session_state:
            st.session_state.search_engine_initialized = False
        if 'matching_success_analysis' not in st.session_state:
            st.session_state.matching_success_analysis = {}
    
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
                        # Initialize search engine after model loading
                        from clap_model import AudioSearchEngine
                        self.search_engine = AudioSearchEngine(self.clap_manager)
                        st.session_state.search_engine_initialized = True
                        st.sidebar.success(f"✅ モデル読み込み完了 ({load_time:.2f}秒)")
                        st.sidebar.success("🔍 音声検索エンジンが初期化されました")
                    else:
                        st.sidebar.error("❌ モデル読み込みに失敗しました")
        else:
            st.sidebar.success("✅ モデル読み込み済み")
            
            # Initialize search engine if not already done
            if self.search_engine is None and not st.session_state.search_engine_initialized:
                try:
                    from clap_model import AudioSearchEngine
                    self.search_engine = AudioSearchEngine(self.clap_manager)
                    st.session_state.search_engine_initialized = True
                    st.sidebar.success("🔍 音声検索エンジンが初期化されました")
                except Exception as e:
                    st.sidebar.error(f"❌ 検索エンジンの初期化に失敗: {str(e)}")
            
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
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🎵 音声分析", 
            "🔍 テキスト-音声マッチング", 
            "🔎 音声検索", 
            "📊 可視化", 
            "⏱️ 検出タイミング分析",
            "⚙️ デバッグ情報"
        ])
        
        with tab1:
            self._display_audio_analysis_tab()
        
        with tab2:
            self._display_text_audio_matching_tab()
        
        with tab3:
            self._display_audio_search_tab()
        
        with tab4:
            self._display_visualization_tab()
        
        with tab5:
            self._display_timing_analysis_tab()
        
        with tab6:
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
        
        # Check if model and audio are loaded
        if not st.session_state.model_loaded:
            st.warning("⚠️ CLAPモデルが読み込まれていません。サイドバーでCLAPモデルを読み込んでください。")
            return
        
        if not st.session_state.audio_path:
            st.warning("⚠️ 音声ファイルが読み込まれていません。サイドバーで音声ファイルをアップロードしてください。")
            return
        
        # Custom text input
        st.subheader("📝 カスタムクエリ")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            custom_text = st.text_input(
                "テキストクエリを入力",
                placeholder="例: 音楽が流れている、人の話し声、鳥の鳴き声...",
                key="matching_custom_text"
            )
        
        with col2:
            if st.button("分析実行", type="primary"):
                if custom_text:
                    self._analyze_single_query_with_success_detection(custom_text)
                else:
                    st.error("テキストクエリを入力してください")
        
        # Display similarity results with success analysis
        if st.session_state.similarity_results:
            self._display_matching_results_with_analysis()
    
    def _analyze_single_query_with_success_detection(self, text_query: str):
        """Analyze single text query with success detection
        
        Args:
            text_query: Single text query to analyze
        """
        if not st.session_state.audio_path or not st.session_state.model_loaded:
            return
        
        with st.spinner("テキスト-音声マッチングを分析中..."):
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
            
            # Perform success analysis
            success_analysis = self._analyze_matching_success(results, text_query)
            st.session_state.matching_success_analysis = success_analysis
            
            st.success(f"✅ 分析完了 ({processing_time:.2f}秒)")
    
    def _analyze_matching_success(self, similarity_scores: Dict[str, float], main_query: str) -> Dict[str, Any]:
        """Analyze matching success based on similarity scores
        
        Args:
            similarity_scores: Dictionary of query to similarity score
            main_query: Main query that was analyzed
            
        Returns:
            Dict[str, Any]: Success analysis results
        """
        if not similarity_scores:
            return {
                "is_successful": False,
                "confidence_level": "none",
                "best_score": 0.0,
                "main_query_score": 0.0,
                "success_reason": "マッチング結果がありません"
            }
        
        # Get main query score
        main_query_score = similarity_scores.get(main_query, 0.0)
        
        # Get best score
        best_score = max(similarity_scores.values())
        best_query = max(similarity_scores.items(), key=lambda x: x[1])[0]
        
        # Define success thresholds
        success_thresholds = {
            "excellent": 0.7,    # 優秀なマッチング
            "good": 0.5,         # 良好なマッチング
            "fair": 0.3,         # 普通のマッチング
            "poor": 0.1          # 貧弱なマッチング
        }
        
        # Determine confidence level
        confidence_level = "none"
        if best_score >= success_thresholds["excellent"]:
            confidence_level = "excellent"
        elif best_score >= success_thresholds["good"]:
            confidence_level = "good"
        elif best_score >= success_thresholds["fair"]:
            confidence_level = "fair"
        elif best_score >= success_thresholds["poor"]:
            confidence_level = "poor"
        
        # Determine if matching is successful
        is_successful = best_score >= success_thresholds["fair"]
        
        # Generate success reason
        success_reason = self._generate_matching_success_reason(
            is_successful, confidence_level, best_score, main_query_score, best_query
        )
        
        return {
            "is_successful": is_successful,
            "confidence_level": confidence_level,
            "best_score": best_score,
            "best_query": best_query,
            "main_query_score": main_query_score,
            "success_thresholds": success_thresholds,
            "success_reason": success_reason
        }
    
    def _generate_matching_success_reason(self, is_successful: bool, confidence_level: str,
                                        best_score: float, main_query_score: float, 
                                        best_query: str) -> str:
        """Generate human-readable matching success reason
        
        Args:
            is_successful: Whether matching was successful
            confidence_level: Confidence level
            best_score: Best similarity score
            main_query_score: Main query score
            best_query: Best matching query
            
        Returns:
            str: Success reason
        """
        if not is_successful:
            return f"マッチングに失敗しました。最高スコア: {best_score:.3f} (閾値: 0.3)"
        
        reasons = []
        
        if confidence_level == "excellent":
            reasons.append("優秀なマッチング")
        elif confidence_level == "good":
            reasons.append("良好なマッチング")
        elif confidence_level == "fair":
            reasons.append("普通のマッチング")
        
        if best_score > 0.8:
            reasons.append("非常に高い類似度")
        elif best_score > 0.6:
            reasons.append("高い類似度")
        
        if main_query_score == best_score:
            reasons.append("メインクエリが最適マッチ")
        
        return " | ".join(reasons) if reasons else "マッチング成功"
    
    def _display_matching_results_with_analysis(self):
        """Display matching results with success analysis"""
        results = st.session_state.similarity_results
        success_analysis = st.session_state.get("matching_success_analysis", {})
        
        # Success analysis display
        if success_analysis:
            st.subheader("📊 マッチング成功分析")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Success status
                if success_analysis.get("is_successful", False):
                    st.success("✅ マッチング成功")
                else:
                    st.error("❌ マッチング失敗")
            
            with col2:
                # Confidence level
                confidence_level = success_analysis.get("confidence_level", "none")
                if confidence_level == "excellent":
                    st.success(f"🎯 優秀 ({confidence_level})")
                elif confidence_level == "good":
                    st.info(f"👍 良好 ({confidence_level})")
                elif confidence_level == "fair":
                    st.warning(f"⚠️ 普通 ({confidence_level})")
                else:
                    st.error(f"❌ 貧弱 ({confidence_level})")
            
            with col3:
                # Best score
                best_score = success_analysis.get("best_score", 0.0)
                st.metric("最高スコア", f"{best_score:.3f}")
            
            with col4:
                # Main query score
                main_query_score = success_analysis.get("main_query_score", 0.0)
                st.metric("メインクエリスコア", f"{main_query_score:.3f}")
            
            # Success reason
            success_reason = success_analysis.get("success_reason", "")
            if success_reason:
                st.info(f"**分析結果:** {success_reason}")
        
        # Detailed results
        st.subheader("📈 詳細結果")
        
        # Sort results by similarity score
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Display results with enhanced styling
        for i, (text, score) in enumerate(sorted_results):
            # Determine similarity level and styling
            if score >= 0.7:
                css_class = "high-similarity"
                emoji = "🟢"
                level_text = "優秀"
            elif score >= 0.5:
                css_class = "medium-similarity"
                emoji = "🟡"
                level_text = "良好"
            elif score >= 0.3:
                css_class = "low-similarity"
                emoji = "🟠"
                level_text = "普通"
            else:
                css_class = "low-similarity"
                emoji = "🔴"
                level_text = "貧弱"
            
            # Highlight main query
            if success_analysis and text == success_analysis.get("best_query", ""):
                text_display = f"**{text}** (最適マッチ)"
            else:
                text_display = text
            
            st.markdown(f"""
            <div class="similarity-score {css_class}">
                {emoji} <strong>{text_display}</strong>: {score:.3f} ({level_text})
            </div>
            """, unsafe_allow_html=True)
        
        # Similarity chart
        st.subheader("📊 類似度チャート")
        similarity_fig = self.viz_manager.create_similarity_bar_chart(results)
        st.plotly_chart(similarity_fig, use_container_width=True, key="matching_similarity_chart")
        
        # Performance metrics
        if st.session_state.processing_times.get("Single Query Analysis"):
            st.subheader("⏱️ パフォーマンス")
            processing_time = st.session_state.processing_times["Single Query Analysis"]
            st.metric("処理時間", f"{processing_time:.3f}秒")
        
        # Recommendations
        if success_analysis:
            recommendations = self._generate_matching_recommendations(success_analysis)
            if recommendations:
                st.subheader("💡 推奨事項")
                for recommendation in recommendations:
                    st.write(f"• {recommendation}")
    
    def _generate_matching_recommendations(self, success_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on matching success analysis
        
        Args:
            success_analysis: Success analysis results
            
        Returns:
            List[str]: List of recommendations
        """
        recommendations = []
        
        if not success_analysis.get("is_successful", False):
            recommendations.append("より具体的なテキストクエリを試してください")
            recommendations.append("音声の特徴を詳しく説明してください")
            recommendations.append("複数のキーワードを組み合わせてみてください")
        
        confidence_level = success_analysis.get("confidence_level", "none")
        if confidence_level == "poor":
            recommendations.append("マッチング結果の信頼度が低いです。別の表現を試してください")
        
        best_score = success_analysis.get("best_score", 0.0)
        if best_score < 0.5:
            recommendations.append("類似度が低いため、より適切なクエリを検討してください")
        
        return recommendations
    
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
    
    def _display_audio_search_tab(self):
        """Display audio search tab"""
        st.header("🔎 音声検索")
        
        # Check if model is loaded
        if not st.session_state.model_loaded:
            st.warning("⚠️ CLAPモデルが読み込まれていません。サイドバーでCLAPモデルを読み込んでください。")
            return
        
        # Initialize search engine if needed
        if self.search_engine is None and not st.session_state.search_engine_initialized:
            try:
                from clap_model import AudioSearchEngine
                self.search_engine = AudioSearchEngine(self.clap_manager)
                st.session_state.search_engine_initialized = True
                st.success("🔍 音声検索エンジンが初期化されました")
            except Exception as e:
                st.error(f"❌ 検索エンジンの初期化に失敗しました: {str(e)}")
                return
        elif self.search_engine is None and st.session_state.search_engine_initialized:
            # Re-initialize if session state says it's initialized but instance is None
            try:
                from clap_model import AudioSearchEngine
                self.search_engine = AudioSearchEngine(self.clap_manager)
                st.info("🔍 音声検索エンジンを再初期化しました")
            except Exception as e:
                st.error(f"❌ 検索エンジンの再初期化に失敗しました: {str(e)}")
                return
        
        if not st.session_state.audio_path:
            st.warning("⚠️ 音声ファイルが読み込まれていません。サイドバーで音声ファイルをアップロードしてください。")
            return
        
        # Search configuration
        st.subheader("⚙️ 検索設定")
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_related = st.checkbox("関連クエリを含める", value=True, 
                                        help="メインクエリに関連するクエリも検索に含めます")
            category_search = st.checkbox("カテゴリ検索", value=False,
                                        help="音声カテゴリに基づく検索を実行します")
        
        with col2:
            # Search statistics
            stats = self.search_engine.get_search_statistics()
            st.info(f"**検索エンジン統計:**\n"
                   f"• カテゴリ数: {stats['total_categories']}\n"
                   f"• 総クエリ数: {stats['total_queries']}\n"
                   f"• 成功閾値: {stats['success_thresholds']['fair']:.1f}")
        
        # Search input
        st.subheader("🔍 検索クエリ")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "検索したい音声内容を入力してください",
                placeholder="例: 音楽が流れている、人の話し声、鳥の鳴き声、車の音..."
            )
        
        with col2:
            if st.button("🔎 検索実行", type="primary"):
                if search_query:
                    self._perform_audio_search(search_query, include_related, category_search)
                else:
                    st.error("検索クエリを入力してください")
        
        # Quick search suggestions
        st.subheader("💡 検索例")
        
        # Category-based suggestions
        stats = self.search_engine.get_search_statistics()
        categories = stats['categories']
        
        selected_category = st.selectbox("カテゴリを選択して検索例を表示", ["すべて"] + categories)
        
        if selected_category != "すべて":
            category_queries = self.search_engine.audio_categories[selected_category]
            st.write(f"**{selected_category}カテゴリの検索例:**")
            
            # Display queries in a grid
            cols = st.columns(3)
            for i, query in enumerate(category_queries[:9]):  # Show first 9 queries
                with cols[i % 3]:
                    if st.button(query, key=f"quick_search_{i}"):
                        self._perform_audio_search(query, include_related, category_search)
        else:
            # Show one example from each category
            st.write("**各カテゴリの検索例:**")
            cols = st.columns(len(categories))
            for i, category in enumerate(categories):
                with cols[i]:
                    example_query = self.search_engine.audio_categories[category][0]
                    if st.button(example_query, key=f"category_example_{i}"):
                        self._perform_audio_search(example_query, include_related, category_search)
        
        # Display search results
        if st.session_state.search_results:
            self._display_search_results()
    
    def _perform_audio_search(self, search_query: str, include_related: bool, category_search: bool):
        """Perform audio search
        
        Args:
            search_query: Search query
            include_related: Whether to include related queries
            category_search: Whether to search within categories
        """
        with st.spinner("音声検索を実行中..."):
            search_results = self.search_engine.search_audio(
                st.session_state.audio_path,
                search_query,
                include_related=include_related,
                category_search=category_search
            )
            
            st.session_state.search_results = search_results
            st.success(f"検索完了！処理時間: {search_results['search_time']:.3f}秒")
    
    def _display_search_results(self):
        """Display search results"""
        results = st.session_state.search_results
        
        st.subheader("📊 検索結果")
        
        # Success analysis
        success_analysis = results["success_analysis"]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Success status
            if success_analysis["is_successful"]:
                st.success("✅ 検索成功")
            else:
                st.error("❌ 検索失敗")
        
        with col2:
            # Confidence level
            confidence_level = success_analysis["confidence_level"]
            if confidence_level == "excellent":
                st.success(f"🎯 優秀 ({confidence_level})")
            elif confidence_level == "good":
                st.info(f"👍 良好 ({confidence_level})")
            elif confidence_level == "fair":
                st.warning(f"⚠️ 普通 ({confidence_level})")
            else:
                st.error(f"❌ 貧弱 ({confidence_level})")
        
        with col3:
            # Best score
            best_score = success_analysis["best_score"]
            st.metric("最高スコア", f"{best_score:.3f}")
        
        with col4:
            # Average score
            avg_score = success_analysis["average_score"]
            st.metric("平均スコア", f"{avg_score:.3f}")
        
        # Success reason
        st.info(f"**検索結果の説明:** {success_analysis['success_reason']}")
        
        # Detailed results
        st.subheader("📈 詳細結果")
        
        # Sort results by score
        sorted_results = sorted(
            results["results"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Display top results
        st.write("**上位検索結果:**")
        for i, (query, score) in enumerate(sorted_results[:10]):  # Show top 10
            # Determine color based on score
            if score >= 0.7:
                color = "🟢"
            elif score >= 0.5:
                color = "🟡"
            elif score >= 0.3:
                color = "🟠"
            else:
                color = "🔴"
            
            # Highlight main query
            if query == results["query"]:
                query_display = f"**{query}** (メインクエリ)"
            else:
                query_display = query
            
            st.write(f"{color} {query_display}: {score:.3f}")
        
        # Category matches
        if results["category_matches"]:
            st.subheader("🏷️ カテゴリマッチ")
            
            # Sort categories by score
            sorted_categories = sorted(
                results["category_matches"].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**カテゴリ別スコア:**")
                for category, score in sorted_categories:
                    st.write(f"• {category}: {score:.3f}")
            
            with col2:
                # Create category chart
                import plotly.graph_objects as go
                
                categories = [cat for cat, _ in sorted_categories]
                scores = [score for _, score in sorted_categories]
                
                fig = go.Figure(data=[
                    go.Bar(x=categories, y=scores, marker_color='lightblue')
                ])
                
                fig.update_layout(
                    title="カテゴリ別マッチングスコア",
                    xaxis_title="カテゴリ",
                    yaxis_title="スコア",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True, key="category_matches_chart")
        
        # Recommendations
        if results["recommendations"]:
            st.subheader("💡 推奨事項")
            for recommendation in results["recommendations"]:
                st.write(f"• {recommendation}")
        
        # Raw results (expandable)
        with st.expander("🔍 生データ"):
            st.json(results)
    
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
        
        # Search engine information
        if self.search_engine:
            st.subheader("🔍 検索エンジン情報")
            search_stats = self.search_engine.get_search_statistics()
            st.json(search_stats)
        
        # Matching success analysis
        if st.session_state.matching_success_analysis:
            st.subheader("🎯 マッチング成功分析")
            st.json(st.session_state.matching_success_analysis)
    
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