# RAG Assistant Prototype

閉じた資料セットに基づいて質問応答を行う、根拠提示型RAGアプリのプロトタイプです。

ユーザーが登録した資料を検索対象として、質問に関連する情報を取得し、その内容に基づいて回答を生成します。回答とあわせて、検索された根拠候補や回答信頼性メモも確認できるようにしています。

展示施設・教育施設・研究室などに蓄積された資料を活用し、利用者や職員の情報探索を支援するシステムの初期試作として開発しました。

## 概要

このアプリでは、資料登録、Vector Store管理、資料に基づく質問応答、根拠候補表示、質問ログ保存、簡易ログ分析を行えます。

現在は、RAG型情報支援アプリの基本機能を素早く検証するため、OpenAI File Search / Vector Storeを用いて実装しています。今後は、OCR、回答可能性判定、ローカルVector DB、ローカルLLMなどを組み合わせた閉域環境向けの構成へ発展させることを想定しています。

## 開発背景

汎用チャットAIは幅広い質問に答えられる一方で、施設固有の資料や内部文書に基づく回答には限界があります。

特に、展示施設・教育施設・研究室などで使う場合、以下のような課題があります。

- 登録資料にない内容を推測で答えてしまう可能性がある
- 回答の根拠が分かりにくい
- PDFや過去資料を横断的に探しにくい
- 実運用では、資料を外部に出さない構成が求められる場合がある
- 利用者の質問傾向を把握し、資料改善やFAQ作成につなげたい場合がある

そこで本プロジェクトでは、まずOpenAI File Search / Vector Storeを用いて、資料に基づく質問応答と根拠提示の基本機能を検証するプロトタイプを作成しました。

## 主な機能

### 資料登録

- txt / md / pdfファイルのアップロード
- 画面上で入力したテキストの直接登録
- OpenAI Vector Storeへの資料登録
- 登録済み資料一覧の表示
- 不要な資料を検索対象から外す機能

### Vector Store管理

- Vector Storeの新規作成
- 使用するVector Storeの切り替え
- 既存Vector Store IDの手入力登録
- OpenAI上のVector Store一覧からの復元
- Vector Store管理情報のローカル保存
- 用途メモ、作成日時、最終使用日時の管理

### 根拠に基づく質問応答

- 登録資料に基づく質問応答
- 回答スタイルの切り替え
  - 簡潔
  - 詳細
  - 箇条書き
  - 説明用
- 検索件数の調整
- 検索結果0件時の簡易ガード
- 資料外質問への応答確認

### 根拠候補の表示

- 検索された根拠候補の表示
- ファイル名の表示
- scoreの表示
- scoreに基づく関連度メモの表示
- 本文プレビューの表示
- OpenAI File Search由来の引用マーカー除去

### 回答信頼性メモ

回答の下に、簡易的な信頼性メモを表示します。

現段階では厳密な回答可能性判定ではありませんが、以下を確認できます。

- 根拠候補が取得されているか
- 回答が通常回答か、確認不可系回答か
- 本文プレビューが取得できているか
- 資料外質問に対して推測回答を抑制できているか

### 質問ログと簡易分析

- セッション中の質問・回答履歴表示
- 質問・回答ログのローカル保存
- 保存済みログの表示
- JSON形式でのログダウンロード
- 保存済みログのクリア
- 保存済みログの簡易分析

ログ分析では、以下を集計します。

- 保存済み質問数
- 通常回答数
- 確認不可系回答数
- 通常回答率
- 確認不可系回答率
- 平均根拠候補数
- 回答スタイルの内訳
- 検索件数設定の内訳
- 根拠候補数の内訳
- 使用Vector Storeの内訳

### 公開用表示モード

スクリーンショット撮影や共有時に、Vector Store IDやFile IDをマスク表示できます。

公開用資料・共有用資料・就活ポートフォリオで画面を見せることを想定し、APIキーや内部IDが画面に映らないようにしています。

## 技術スタック

- Python
- Streamlit
- OpenAI Responses API
- OpenAI File Search
- OpenAI Vector Store
- python-dotenv
- Git / GitHub

## システム構成

```text
User
  ↓
Streamlit UI
  ↓
app.py
  ├── rag_client.py   # OpenAI API / RAG処理
  └── storage.py      # ローカル保存・ログ・Vector Store台帳管理
  ↓
OpenAI File Search / Vector Store
  ↓
Answer + Evidence Candidates
```

RAG処理とローカル保存処理を分けることで、今後の拡張やローカルRAGへの移行をしやすい構成にしています。

## ディレクトリ構成

```text
rag-assistant-prototype/
├── app.py
├── rag_client.py
├── storage.py
├── requirements.txt
├── README.md
├── dev_log.md
├── .env.example
├── .gitignore
├── scripts/
│   ├── ask_rag.py
│   └── setup_vector_store.py
└── data/
    └── raw/
        └── .gitkeep
```

`scripts/` には、初期検証時に使用した補助スクリプトを配置しています。アプリ本体は主に `app.py`、`rag_client.py`、`storage.py` で構成されています。

以下のファイル・ディレクトリはローカル環境専用のため、リポジトリには含めていません。

```text
.env
app_data/
temp_uploads/
demo_docs/
private_docs/
presentation_assets.md
presentation_assets/
screenshots/
```

## セットアップ

### 1. リポジトリを取得

```bash
git clone https://github.com/yutoo77/rag-assistant-prototype.git
cd rag-assistant-prototype
```

### 2. 仮想環境を作成

Windows PowerShellの場合:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

スクリプト実行が制限されている場合:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### 3. ライブラリをインストール

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. `.env` を作成

プロジェクト直下に `.env` を作成します。

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=your_model_name
VECTOR_STORE_ID=your_vector_store_id
```

既存のVector Store IDがない場合は、アプリ起動後に画面上から新規作成できます。

`.env` は `.gitignore` によりGitHubには公開されません。

## 起動方法

```powershell
python -m streamlit run app.py
```

起動後、ブラウザで以下を開きます。

```text
http://localhost:8501
```

## 使い方

1. Streamlitアプリを起動する
2. Vector Storeを作成または選択する
3. 資料をアップロード、またはテキストを直接登録する
4. 登録資料に基づいて質問する
5. 回答と根拠候補を確認する
6. 回答信頼性メモを確認する
7. 質問履歴やログ分析を確認する
8. スクリーンショットを撮る場合は公開用表示モードを使用する

## 現在の制約

このプロジェクトは初期プロトタイプであり、本番運用を前提とした完成版ではありません。

現在の主な制約は以下です。

- OpenAI Vector Storeを利用しているため、資料はOpenAI側に保存される
- スキャンPDFや画像ベースのPDFにはOCR前処理が必要になる場合がある
- 回答可能性判定はまだ簡易的である
- 根拠本文の取得や表示はAPIの返却形式に依存する
- 質問ログに機密情報が含まれる可能性があるため、実運用ではログ管理が必要
- アクセス制御や権限管理は未実装

## 今後の展望

### 短期的な改善

- デモ用資料の整備
- UIの分かりやすさ改善
- エラーメッセージの改善
- 根拠候補表示の安定化
- 共有用スクリーンショットや操作動画の整備

### 中期的な改善

- スキャンPDF向けのOCR前処理
- 回答可能性判定の強化
- 検索scoreと根拠本文に基づく回答検証
- 質問ログ分析の高度化
- 類似質問のクラスタリング

### 長期的な展望

- ローカルVector DBへの移行
- ローカルEmbeddingモデルの利用
- ローカルLLMの利用
- 閉域環境向けRAG構成の検討
- 音声入力・音声出力への対応
- 展示施設・教育施設での情報支援システムへの応用

## この開発で学んだこと

このプロジェクトを通じて、RAGアプリを実用に近づけるには、単にLLMと資料検索を接続するだけでは不十分だと学びました。

特に重要だと感じた点は以下です。

- 資料セットをどのように登録・管理するか
- 回答の根拠をどのように見せるか
- 資料にない内容への回答をどう抑制するか
- 質問ログをどのように蓄積し、改善に活かすか
- 公開リポジトリで機密情報を扱わない設計にすること
- 将来的にローカル環境へ移行できる構成にしておくこと

本プロジェクトは、RAGの実装だけでなく、情報支援サービスとしての設計、ユーザー体験、研究への発展可能性を考えるきっかけになりました。

## 位置づけ

このリポジトリは、就活用ポートフォリオおよび研究プロトタイプとして公開しています。

根拠提示型RAGアプリの初期実装として、資料登録、質問応答、根拠候補表示、ログ保存、Vector Store管理までを実装し、今後はOCR、回答可能性判定、ローカルVector DB、ローカルLLMを組み合わせた閉域環境向けRAGシステムへ発展させる予定です。
