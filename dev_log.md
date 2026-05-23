# 開発ログ

## 2026-05-xx

### 実施内容
- OpenAI Vector Storeを作成
- サンプル資料 `rag_sample_museum_guide.txt` を登録
- `ask_rag.py` からFile Searchを使った質問応答を確認

### 確認できたこと
- 資料内にある質問には回答できた
- 資料外の質問では、資料内で確認できない旨を返せた
- file_search_call により検索が実行されていることを確認した

### 気づき
- スキャンPDFはそのままでは検索対象として扱いにくい
- 事前にテキスト化・OCR・Markdown化した資料を使う方が安定する