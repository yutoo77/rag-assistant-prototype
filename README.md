# RAG Assistant Prototype

閉じた資料セットに基づいて質問応答を行う、根拠提示型RAGアプリのプロトタイプです。

このアプリでは、ユーザーが登録した資料を検索対象として、質問に関連する情報を取得し、その内容に基づいて回答を生成します。  
回答とあわせて、検索された根拠候補も確認できるようにしています。

## 目的

このプロジェクトは、展示施設・教育施設・研究室などで利用できる、資料ベースの情報支援アプリを試作することを目的としています。

通常の汎用チャットAIではなく、あらかじめ登録された資料セットに基づいて回答する構成を目指しています。

想定している利用例は次の通りです。

- 展示資料に基づく質問応答
- 施設内資料やマニュアルに基づく情報検索
- テキスト化済み資料を用いたRAGデモ
- 研究・発表用のRAGプロトタイプ検証

## 主な機能

現在のプロトタイプでは、以下の機能を実装しています。

- StreamlitによるWeb UI
- テキストファイル、Markdownファイル、PDFファイルのアップロード
- 画面上で直接入力したテキストの資料登録
- OpenAI Vector Storeへの資料登録
- 登録済み資料一覧の表示
- 不要な資料を検索対象から外す機能
- 登録資料に基づく質問応答
- 回答の根拠候補表示
- 質問・回答履歴の表示
- 質問・回答ログのローカル保存
- 質問・回答ログのJSONダウンロード
- 検索件数の調整
- 回答スタイルの切り替え
- Vector Storeの新規作成
- Vector Storeの切り替え
- Vector Store管理情報のローカル保存
- 既存Vector Store IDの手入力登録

## 技術構成

- Python
- Streamlit
- OpenAI Responses API
- OpenAI File Search
- OpenAI Vector Store
- python-dotenv
- Git / GitHub

## ディレクトリ構成

```text
rag-assistant-prototype/
├── app.py              # Streamlitアプリ本体
├── rag_client.py       # OpenAI API / RAG関連処理
├── storage.py          # ローカル保存処理
├── requirements.txt    # Python依存パッケージ
├── README.md
├── .gitignore
├── data/
│   └── raw/
│       └── .gitkeep