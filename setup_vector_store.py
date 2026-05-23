import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


RAW_DIR = Path("data/raw")


def get_first_supported_file_path() -> Path:
    supported_extensions = ["*.pdf", "*.txt", "*.md"]

    files = []
    for extension in supported_extensions:
        files.extend(RAW_DIR.glob(extension))

    if not files:
        raise FileNotFoundError(
            "data/raw に .pdf / .txt / .md ファイルがありません。"
        )

    return files[0]

def main():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(".env に OPENAI_API_KEY が設定されていません。")

    client = OpenAI(api_key=api_key)

    target_path = get_first_supported_file_path()
    print(f"アップロード対象ファイル: {target_path}")

    # 1. OpenAI Files API にPDFをアップロード
    with target_path.open("rb") as f:
        uploaded_file = client.files.create(
            file=f,
            purpose="assistants",
        )

    print(f"File ID: {uploaded_file.id}")

    # 2. Vector Storeを作成
    vector_store = client.vector_stores.create(
        name="rag_assistant_knowledge_base",
    )

    print(f"Vector Store ID: {vector_store.id}")

    # 3. アップロードしたPDFをVector Storeに追加
    vector_store_file = client.vector_stores.files.create(
        vector_store_id=vector_store.id,
        file_id=uploaded_file.id,
    )

    print(f"Vector Store File ID: {vector_store_file.id}")
    print("ファイル処理が完了するまで確認します...")

    # 4. 処理完了まで待つ
    while True:
        result = client.vector_stores.files.retrieve(
            vector_store_id=vector_store.id,
            file_id=uploaded_file.id,
        )

        print(f"status: {result.status}")

        if result.status == "completed":
            break

        if result.status in ["failed", "cancelled"]:
            raise RuntimeError(f"ファイル処理に失敗しました: {result.status}")

        time.sleep(3)

    print("\n✅ Vector Storeの準備が完了しました。")
    print("次のIDを .env の VECTOR_STORE_ID に貼ってください。")
    print(f"VECTOR_STORE_ID={vector_store.id}")


if __name__ == "__main__":
    main()