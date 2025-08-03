# CLAP Audio-Text Understanding Application

CLAP（Contrastive Language-Audio Pretraining）を使用した音声-テキスト理解アプリケーション。Streamlitを使用した包括的な音声分析ツールです。

## 🚀 プロジェクト概要

このアプリケーションは、CLAPモデルを使用して音声ファイルとテキストクエリの間の類似度を計算し、音声の内容を理解するためのWebアプリケーションです。音声分析、テキスト-音声マッチング、可視化機能を提供します。

## 📋 目次

- [機能](#機能)
- [CLAP技術詳細](#clap技術詳細)
- [ファイル構成](#ファイル構成)
- [インストール](#インストール)
- [使用方法](#使用方法)
- [技術仕様](#技術仕様)
- [ライブラリ](#ライブラリ)
- [将来の拡張案](#-将来の拡張案)
- [トラブルシューティング](#トラブルシューティング)
- [貢献](#貢献)
- [ライセンス](#ライセンス)

## 機能

### 🎯 **主要機能**
- **音声-テキスト類似度計算**: CLAPモデルを使用した音声とテキストの類似度分析
- **音声分析**: 波形、スペクトログラム、MFCC、音声特徴の抽出
- **リアルタイム可視化**: Plotlyを使用したインタラクティブな可視化
- **カスタムクエリ**: ユーザー定義のテキストクエリでの分析
- **サンプル音声**: カテゴリ別のテスト用音声ファイルと合成音声生成
- **カテゴリ別クエリサジェスト**: 音声カテゴリに応じた適切なクエリ提案
- **パフォーマンス監視**: 処理時間とモデル情報の表示

### 🎵 **音声サポート**
- **対応形式**: WAV, MP3, FLAC, M4A, OGG
- **音声特徴**: スペクトル重心、帯域幅、ゼロクロス率、RMSエネルギー
- **可視化**: 波形、スペクトログラム、MFCCヒートマップ、レーダーチャート
- **サンプル音声**: カテゴリ別のテスト用音声ファイル（音楽、自然、都市、人の声、機械、動物）
- **合成音声**: テスト用の合成音声生成（正弦波、ホワイトノイズ、ピンクノイズ、チャープ）

## CLAP技術詳細

### 🧠 **CLAP（Contrastive Language-Audio Pretraining）とは**

CLAPは、音声とテキストの間の対比学習（Contrastive Learning）を使用して、音声とテキストの理解を同時に学習する手法です。

#### **アーキテクチャ**
```
音声入力 → 音声エンコーダー → 音声埋め込み
    ↓
テキスト入力 → テキストエンコーダー → テキスト埋め込み
    ↓
対比学習による類似度計算
```

#### **主要コンポーネント**

1. **音声エンコーダー（Audio Encoder）**
   - HTS-AT（Hierarchical Token Semantic Audio Transformer）
   - 音声信号を階層的に処理
   - 時間-周波数領域の特徴を抽出

2. **テキストエンコーダー（Text Encoder）**
   - Transformerベースの言語モデル
   - テキストの意味的表現を学習

3. **対比学習（Contrastive Learning）**
   - 正のペア（対応する音声-テキスト）と負のペア（非対応）を使用
   - コサイン類似度による類似度計算

#### **学習目標**
```python
# 対比学習の損失関数
def contrastive_loss(audio_embeddings, text_embeddings, temperature=0.07):
    # 正規化
    audio_norm = F.normalize(audio_embeddings, dim=-1)
    text_norm = F.normalize(text_embeddings, dim=-1)
    
    # 類似度行列
    similarity_matrix = torch.matmul(audio_norm, text_norm.T) / temperature
    
    # 対角線が正のペア
    labels = torch.arange(similarity_matrix.size(0))
    
    # クロスエントロピー損失
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss
```

#### **使用モデル**
- **モデル名**: `laion/clap-htsat-fused`
- **アーキテクチャ**: HTS-AT + Fused Transformer
- **パラメータ数**: 約1.2B
- **サンプルレート**: 48kHz
- **埋め込み次元**: 1024

### 📊 **音声特徴抽出**

#### **スペクトル特徴**
```python
# スペクトル重心（Spectral Centroid）
spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)

# スペクトル帯域幅（Spectral Bandwidth）
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)

# ゼロクロス率（Zero Crossing Rate）
zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)

# RMSエネルギー
rms = librosa.feature.rms(y=audio)
```

#### **MFCC特徴**
```python
# MFCC（Mel-frequency cepstral coefficients）
mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
```

## ファイル構成

```
clap_app/
├── app.py                    # メインStreamlitアプリケーション
├── clap_model.py            # CLAPモデル管理クラス
├── visualization_manager.py # 可視化マネージャー
├── sample_audio_manager.py  # サンプル音声管理クラス
├── pyproject.toml          # プロジェクト設定
├── README.md               # プロジェクトドキュメント
└── .venv/                  # 仮想環境
```

### 📄 ファイル詳細説明

#### `app.py` - メインアプリケーション
- **役割**: Streamlit Webアプリケーションのメインエントリーポイント
- **機能**:
  - ユーザーインターフェースの構築
  - タブベースのナビゲーション
  - 音声ファイルアップロードと分析
  - リアルタイム結果表示

#### `clap_model.py` - CLAPモデル管理
- **役割**: CLAPモデルの読み込みと推論
- **主要クラス**:
  - `CLAPModelManager`: CLAPモデルの管理と推論
  - `AudioAnalyzer`: 音声ファイルの分析

#### `visualization_manager.py` - 可視化管理
- **役割**: 音声分析結果の可視化
- **主要機能**:
  - 波形表示
  - スペクトログラム
  - MFCCヒートマップ
  - 類似度スコアチャート
  - レーダーチャート

#### `sample_audio_manager.py` - サンプル音声管理
- **役割**: テスト用音声ファイルと合成音声の管理
- **主要機能**:
  - カテゴリ別サンプル音声の提供
  - 合成音声の生成（正弦波、ノイズ、チャープ）
  - カテゴリ別クエリサジェスト
  - 音声ファイルのダウンロードと管理

## インストール

### Prerequisites
- Python 3.8.1 or higher
- uv package manager (recommended)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd clap_app

# Install dependencies using uv
uv sync

# Activate virtual environment
source .venv/bin/activate

# Run the application
uv run streamlit run app.py
```

### Alternative Installation
```bash
# Using pip
pip install -e .

# Run the application
streamlit run app.py
```

## 使用方法

### 1. モデル読み込み
- サイドバーの「CLAPモデルを読み込み」ボタンをクリック
- 初回実行時はモデルのダウンロードに時間がかかります

### 2. 音声ファイルアップロード
- サイドバーから音声ファイル（WAV, MP3, FLAC, M4A, OGG）をアップロード
- アップロード後、音声プレーヤーで再生可能

### 3. 音声サンプル選択
- カテゴリ別のサンプル音声から選択（音楽、自然、都市、人の声、機械、動物）
- 合成音声の生成（正弦波、ホワイトノイズ、ピンクノイズ、チャープ）
- 音声ファイルのアップロードも可能

### 4. テキストクエリ設定
- カスタムクエリの入力
- カテゴリ別のクエリサジェスト
- 一般的なクエリの選択
- 「選択したクエリで分析」ボタンで分析実行

### 5. 結果確認
- **🎵 音声分析タブ**: 音声特徴と可視化
- **🔍 テキスト-音声マッチングタブ**: 類似度スコア
- **📊 可視化タブ**: 包括的なダッシュボード
- **⚙️ デバッグ情報タブ**: モデル情報とパフォーマンス

### 6. 類似度スコアの解釈
- **🟢 高類似度 (≥0.6)**: 強い関連性
- **🟡 中類似度 (0.3-0.6)**: 中程度の関連性
- **🔴 低類似度 (<0.3)**: 弱い関連性

## 技術仕様

### 🏗️ **アーキテクチャ**
- **フロントエンド**: Streamlit
- **音声処理**: librosa, soundfile
- **機械学習**: PyTorch, transformers
- **可視化**: Plotly, matplotlib
- **数値計算**: NumPy, SciPy
- **ベクトル化**: メモリ上リアルタイム処理（RAG非使用）

### 🔍 **現在の実装方式**
- **リアルタイムエンコーディング**: 音声ファイルの即座なベクトル化
- **メモリ上類似度計算**: コサイン類似度による直接計算
- **セッション管理**: Streamlitセッション状態での結果保存
- **軽量設計**: 外部データベースへの依存なし

### 📊 **現在の実装 vs RAGシステム比較**

| 特徴 | 現在のCLAPアプリ | RAGシステム |
|------|------------------|-------------|
| **ベクトル保存** | メモリ上のみ | ベクトルDB（Pinecone、Weaviate等） |
| **検索方式** | リアルタイム計算 | 事前計算済みベクトルの検索 |
| **スケーラビリティ** | 単一セッション | 大規模データセット対応 |
| **永続化** | なし（セッション終了で消失） | 永続的保存 |
| **処理速度** | 高速（リアルタイム） | 中程度（インデックス検索） |
| **メモリ使用量** | 低（一時的） | 高（永続的） |
| **実装複雑度** | 低 | 高 |
| **用途** | デモンストレーション・実験 | 本格的な音声検索システム |

### 🔧 **処理フロー**
1. **音声読み込み**: librosaで音声ファイルを読み込み
2. **前処理**: サンプルレート統一、正規化
3. **特徴抽出**: スペクトル特徴、MFCCの計算
4. **CLAP推論**: 音声とテキストの埋め込み生成
5. **類似度計算**: コサイン類似度によるスコア算出
6. **可視化**: Plotlyによる結果表示

### 📊 **パフォーマンス**
- **モデル読み込み**: 初回約30-60秒（ネットワーク速度依存）
- **音声分析**: 音声長に依存（通常1-10秒）
- **類似度計算**: クエリ数に依存（1クエリ約0.1-0.5秒）
- **メモリ使用量**: モデル約2-4GB、音声処理約100-500MB

## ライブラリ

### 🧰 **主要ライブラリ**

| ライブラリ名 | バージョン | 主な用途・特徴 |
|-------------|------------|----------------|
| **transformers** | >=4.30.0 | Hugging Face Transformers、CLAPモデル |
| **torch** | >=2.0.0 | PyTorch深層学習フレームワーク |
| **torchaudio** | >=2.0.0 | PyTorch音声処理ライブラリ |
| **librosa** | >=0.10.0 | 音声分析・特徴抽出ライブラリ |
| **soundfile** | >=0.12.0 | 音声ファイル読み書き |
| **plotly** | >=5.15.0 | インタラクティブ可視化 |
| **streamlit** | >=1.28.0 | Webアプリケーションフレームワーク |

### 📦 **依存関係**
```toml
dependencies = [
    "streamlit>=1.28.0",
    "torch>=2.0.0",
    "torchaudio>=2.0.0",
    "transformers>=4.30.0",
    "librosa>=0.10.0",
    "soundfile>=0.12.0",
    "numpy>=1.24.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "plotly>=5.15.0",
    "pandas>=2.0.0",
    "scipy>=1.10.0",
    "psutil>=5.9.0",
    "gradio>=4.0.0",
    "openai-whisper>=20231117",
    "pydub>=0.25.0",
]
```

## 🔮 将来の拡張案

### 🎯 **RAG（Retrieval-Augmented Generation）機能の追加**

現在のアプリケーションはメモリ上でのリアルタイム処理を採用していますが、以下のRAG機能を追加することで、より高度な音声理解システムを構築できます。

#### **📚 音声ライブラリの構築**
```python
class AudioLibrary:
    def __init__(self):
        self.audio_collection = {}  # 音声ファイルのメタデータ
        self.vector_db = None       # ベクトルデータベース
    
    def add_audio_file(self, audio_path: str, metadata: dict):
        # 音声をエンコードしてベクトルDBに保存
        audio_embedding = self.clap_manager.encode_audio(audio_path)
        self.vector_db.store(audio_embedding, metadata)
```

**機能**:
- 大量の音声ファイルの事前エンコード
- メタデータ付き音声コレクションの管理
- ベクトルデータベース（Pinecone、Weaviate等）との統合
- 音声ファイルのバッチ処理とインデックス化

#### **🔍 セマンティック検索**
```python
def semantic_audio_search(self, text_query: str, top_k: int = 10):
    # テキストクエリをベクトル化
    text_embedding = self.clap_manager.encode_text(text_query)
    
    # ベクトルDBで類似音声を検索
    results = self.vector_db.search(text_embedding, top_k=top_k)
    
    return results
```

**機能**:
- 自然言語による音声内容検索
- 音声-音声間の類似度検索
- ハイブリッド検索（テキスト + 音声クエリ）
- 検索結果のランキングとフィルタリング

#### **🎵 音声推奨システム**
```python
def recommend_similar_audio(self, audio_id: str, top_k: int = 5):
    # 基準音声のベクトルを取得
    base_embedding = self.vector_db.get_embedding(audio_id)
    
    # 類似音声を検索
    similar_audio = self.vector_db.search(base_embedding, top_k=top_k)
    
    return similar_audio
```

**機能**:
- 類似音声の自動推薦
- ユーザー嗜好に基づく推奨
- 協調フィルタリングの実装
- 推奨理由の説明機能

#### **🏷️ 音声分類システム**
```python
def auto_classify_audio(self, audio_path: str):
    # 音声をエンコード
    audio_embedding = self.clap_manager.encode_audio(audio_path)
    
    # 事前定義されたカテゴリとの類似度を計算
    categories = ["音楽", "自然", "都市", "人の声", "機械", "動物"]
    category_embeddings = [self.clap_manager.encode_text(cat) for cat in categories]
    
    # 最も類似度の高いカテゴリを選択
    similarities = [self.compute_similarity(audio_embedding, cat_emb) 
                   for cat_emb in category_embeddings]
    
    best_category = categories[np.argmax(similarities)]
    return best_category, max(similarities)
```

**機能**:
- カテゴリ別の自動分類
- マルチラベル分類
- 信頼度スコアの提供
- カスタム分類器の学習

### 🛠️ **技術的実装案**

#### **ベクトルデータベースの選択**
| データベース | 特徴 | 用途 |
|-------------|------|------|
| **Pinecone** | マネージドサービス、高速検索 | 本格的な音声検索システム |
| **Weaviate** | オープンソース、スキーマ定義 | カスタム音声ライブラリ |
| **Qdrant** | 高性能、Rust実装 | 大規模音声データセット |
| **Chroma** | 軽量、Pythonネイティブ | 開発・テスト環境 |

#### **システムアーキテクチャ**
```
音声ファイル → CLAPエンコーダー → ベクトルDB
     ↓
テキストクエリ → CLAPエンコーダー → 検索エンジン
     ↓
検索結果 → ランキング → 推奨システム
```

#### **実装ステップ**
1. **ベクトルデータベースのセットアップ**
2. **音声ライブラリ管理システムの構築**
3. **検索APIの実装**
4. **推奨アルゴリズムの開発**
5. **分類システムの統合**
6. **Web UIの拡張**

### 📊 **期待される効果**

#### **ユーザー体験の向上**
- 大規模音声ライブラリからの高速検索
- パーソナライズされた音声推奨
- 直感的な音声分類機能

#### **システム性能の向上**
- 事前計算による高速レスポンス
- スケーラブルな音声処理
- 永続的な音声データ管理

#### **ビジネス価値**
- 音声コンテンツの効率的な管理
- 音声検索エンジンの構築
- 音声分析プラットフォームの提供

## トラブルシューティング

### Common Issues

#### モデル読み込みエラー
- **ネットワーク接続**: インターネット接続を確認
- **ディスク容量**: モデルファイル用の十分な容量を確保
- **メモリ不足**: システムメモリを確認（推奨8GB以上）

#### 音声ファイルエラー
- **対応形式の確認**: WAV, MP3, FLAC, M4A, OGGのみ対応
- **ファイル破損**: 音声ファイルの整合性を確認
- **権限の確認**: ファイルの読み取り権限を確認

#### 処理速度の問題
- **GPU利用**: CUDA対応GPUがある場合は自動使用
- **音声長**: 長い音声ファイルは処理時間が増加
- **クエリ数**: 多数のクエリは処理時間が増加

### Performance Tips
- GPU利用可能時は自動的に高速化
- 音声ファイルは事前に短縮することを推奨
- 不要なクエリは削除して効率的に実行
- メモリ使用量を監視して適切なファイルサイズを選択

## 貢献

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### 開発ガイドライン
- **コードスタイル**: Black, flake8準拠
- **型ヒント**: 全関数に型ヒントを追加
- **ドキュメント**: 関数とクラスにdocstringを追加
- **テスト**: 新機能にはテストを追加

## ライセンス

This project is licensed under the MIT License - see the LICENSE file for details.

## 参考文献

- [CLAP Paper](https://arxiv.org/abs/2206.04769)
- [Hugging Face CLAP](https://huggingface.co/laion/clap-htsat-fused)
- [LAION Audio Dataset](https://github.com/LAION-AI/CLAP)
- [librosa Documentation](https://librosa.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## Acknowledgments

- [LAION](https://laion.ai/) for CLAP model and dataset
- [Hugging Face](https://huggingface.co/) for transformers library
- [librosa](https://librosa.org/) for audio processing capabilities
- [Streamlit](https://streamlit.io/) for web application framework
- [Plotly](https://plotly.com/) for interactive visualizations 

## 🔍 **現在のアーキテクチャ**

### **メモリ上ベクトル化の仕組み**

1. **リアルタイムエンコーディング**
   ```python
   # 音声ファイルをアップロード時に即座にエンコード
   audio_embeddings = self.clap_manager.encode_audio(audio_tensor)
   text_embeddings = self.clap_manager.encode_text(text_query)
   ```

2. **メモリ上での類似度計算**
   ```python
   # コサイン類似度を直接計算
   similarity = torch.cosine_similarity(audio_norm, text_norm, dim=-1)
   ```

3. **セッション状態での管理**
   ```python
   # Streamlitのセッション状態に結果を保存
   st.session_state.similarity_results = results
   st.session_state.audio_features = audio_info
   ```

### **RAGとの違い**

| 特徴 | 現在のCLAPアプリ | RAGシステム |
|------|------------------|-------------|
| **ベクトル保存** | メモリ上のみ | ベクトルDB（Pinecone、Weaviate等） |
| **検索方式** | リアルタイム計算 | 事前計算済みベクトルの検索 |
| **スケーラビリティ** | 単一セッション | 大規模データセット対応 |
| **永続化** | なし（セッション終了で消失） | 永続的保存 |
| **処理速度** | 高速（リアルタイム） | 中程度（インデックス検索） |

## 🚀 **RAG版への拡張可能性**

もしRAG機能を追加したい場合は、以下のような拡張が可能です：

### **1. ベクトルデータベースの統合**
```python
# 例：Pineconeとの統合
import pinecone

class CLAPRAGManager:
    def __init__(self):
        self.pinecone_index = pinecone.Index("clap-audio-index")
    
    def store_audio_embedding(self, audio_id: str, audio_embedding: np.ndarray):
        self.pinecone_index.upsert([(audio_id, audio_embedding.tolist())])
    
    def search_similar_audio(self, text_query: str, top_k: int = 5):
        text_embedding = self.clap_manager.encode_text(text_query)
        results = self.pinecone_index.query(
            vector=text_embedding.tolist(),
            top_k=top_k
        )
        return results
```

### **2. 音声ライブラリの構築**
```python
class AudioLibrary:
    def __init__(self):
        self.audio_collection = {}  # 音声ファイルのメタデータ
        self.vector_db = None       # ベクトルデータベース
    
    def add_audio_file(self, audio_path: str, metadata: dict):
        # 音声をエンコードしてベクトルDBに保存
        audio_embedding = self.clap_manager.encode_audio(audio_path)
        self.vector_db.store(audio_embedding, metadata)
```

### **3. ハイブリッド検索**
```python
<code_block_to_apply_changes_from>
```

## 💡 **現在のアプリケーションの利点**

1. **シンプルな実装**: 複雑なインフラが不要
2. **高速レスポンス**: リアルタイムでの類似度計算
3. **軽量**: 外部データベースへの依存なし
4. **デモンストレーション**: CLAPモデルの性能を直接確認可能

## 🔮 **将来の拡張案**

もしRAG機能を追加したい場合は、以下のような機能を実装できます：

- **音声ライブラリの構築**: 大量の音声ファイルの事前エンコード
- **セマンティック検索**: 音声内容による検索機能
- **音声推奨システム**: 類似音声の推薦
- **音声分類**: カテゴリ別の自動分類

現在のアプリケーションは、CLAPモデルの基本的な音声-テキスト理解能力をデモンストレーションするのに最適な設計になっています。RAG機能が必要な場合は、上記のような拡張を検討できます。

何か特定の機能を追加したい場合は、お知らせください！ 