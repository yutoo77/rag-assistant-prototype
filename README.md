# RAG Assistant Prototype

閉じた資料に基づいて、根拠つきで質問応答するRAGアプリのプロトタイプです。

## 目的

6/11のゼミ発表に向けて、RAGを用いた情報提供支援システムの初期プロトタイプを作成する。

## 主な機能

- PDFまたはテキスト資料の読み込み
- チャンク分割
- ベクトル検索
- LLMによる回答生成
- 回答根拠の表示

## 技術構成

- Python
- Streamlit
- OpenAI API
- LangChain
- Chroma