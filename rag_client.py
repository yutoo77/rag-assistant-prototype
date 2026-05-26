import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI


TEMP_UPLOAD_DIR = Path("temp_uploads")


def sanitize_filename(filename: str) -> str:
    """
    入力された資料名を、安全なファイル名に変換する。
    Windowsで使えない文字などを避ける。
    """
    filename = filename.strip()

    if not filename:
        filename = "direct_text_note"

    filename = re.sub(r'[\\/:*?"<>|]', "_", filename)
    filename = re.sub(r"\s+", "_", filename)

    if not filename.endswith(".txt"):
        filename += ".txt"

    return filename


def is_valid_vector_store_id(vector_store_id: str) -> bool:
    """
    Vector Store IDらしい形式か簡易チェックする。
    厳密なAPI確認ではなく、入力ミスを減らすためのチェック。
    """
    return vector_store_id.strip().startswith("vs_")


def get_answer_style_instruction(answer_style: str) -> str:
    """
    画面で選ばれた回答スタイルに応じて、モデルへの追加指示を返す。
    """
    style_instructions = {
        "簡潔": (
            "回答は短めにしてください。"
            "重要な点を中心に、2〜4文程度でまとめてください。"
        ),
        "詳細": (
            "回答はやや詳しくしてください。"
            "背景、理由、注意点が分かるように、必要に応じて段落を分けて説明してください。"
        ),
        "箇条書き": (
            "回答は箇条書きを中心にしてください。"
            "重要な点を整理し、読みやすく示してください。"
        ),
        "発表用": (
            "回答はゼミ発表で説明しやすい形にしてください。"
            "背景、要点、意義が伝わるように、少し丁寧な表現でまとめてください。"
        ),
    }

    return style_instructions.get(answer_style, style_instructions["簡潔"])


def extract_file_search_results(response: Any) -> list[dict[str, str]]:
    """
    Responses API の返答から file_search の検索結果を取り出す。
    回答の根拠候補として画面に表示するために使う。
    """
    results_list = []

    for item in response.output:
        if getattr(item, "type", None) != "file_search_call":
            continue

        results = getattr(item, "results", None)

        if not results:
            continue

        for result in results:
            filename = str(getattr(result, "filename", "unknown"))
            score = str(getattr(result, "score", "unknown"))

            content = getattr(result, "content", None)
            text_parts = []

            if content:
                for c in content:
                    text = getattr(c, "text", "")
                    if text:
                        text_parts.append(text)

            preview = "\n".join(text_parts)

            results_list.append(
                {
                    "filename": filename,
                    "score": score,
                    "preview": preview,
                }
            )

    return results_list


def create_vector_store(
    client: OpenAI,
    name: str,
) -> dict[str, str]:
    """
    新しいVector Storeを作成する。
    """
    vector_store = client.vector_stores.create(name=name)

    return {
        "name": str(getattr(vector_store, "name", name)),
        "vector_store_id": str(vector_store.id),
    }


def upload_path_to_vector_store(
    client: OpenAI,
    file_path: Path,
    vector_store_id: str,
) -> dict[str, str]:
    """
    ローカルファイルをOpenAI Files APIにアップロードし、
    指定したVector Storeに追加する共通処理。
    """
    with file_path.open("rb") as f:
        openai_file = client.files.create(
            file=f,
            purpose="assistants",
        )

    vector_store_file = client.vector_stores.files.create(
        vector_store_id=vector_store_id,
        file_id=openai_file.id,
    )

    status = vector_store_file.status

    while status not in ["completed", "failed", "cancelled"]:
        time.sleep(2)

        current = client.vector_stores.files.retrieve(
            vector_store_id=vector_store_id,
            file_id=openai_file.id,
        )

        status = current.status

    if status != "completed":
        raise RuntimeError(f"ファイル処理に失敗しました。status={status}")

    return {
        "filename": file_path.name,
        "file_id": openai_file.id,
        "vector_store_id": vector_store_id,
        "status": status,
    }


def upload_file_to_vector_store(
    client: OpenAI,
    uploaded_file: Any,
    vector_store_id: str,
) -> dict[str, str]:
    """
    Streamlitでアップロードされたファイルを一時保存し、
    既存のVector Storeに追加する。
    """
    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    temp_path = TEMP_UPLOAD_DIR / uploaded_file.name
    temp_path.write_bytes(uploaded_file.getvalue())

    return upload_path_to_vector_store(
        client=client,
        file_path=temp_path,
        vector_store_id=vector_store_id,
    )


def upload_text_to_vector_store(
    client: OpenAI,
    title: str,
    text: str,
    vector_store_id: str,
) -> dict[str, str]:
    """
    画面に直接入力されたテキストを .txt ファイルとして一時保存し、
    Vector Storeに追加する。
    """
    if not text.strip():
        raise ValueError("登録するテキストが空です。")

    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    safe_filename = sanitize_filename(title)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_path = TEMP_UPLOAD_DIR / f"{timestamp}_{safe_filename}"

    temp_path.write_text(text, encoding="utf-8")

    return upload_path_to_vector_store(
        client=client,
        file_path=temp_path,
        vector_store_id=vector_store_id,
    )


def list_vector_store_files(
    client: OpenAI,
    vector_store_id: str,
) -> list[dict[str, str]]:
    """
    Vector Storeに登録されているファイル一覧を取得する。
    """
    vector_store_files = client.vector_stores.files.list(
        vector_store_id=vector_store_id
    )

    file_list = []

    for vs_file in vector_store_files.data:
        file_id = str(
            getattr(vs_file, "file_id", None)
            or getattr(vs_file, "id", "unknown")
        )
        status = str(getattr(vs_file, "status", "unknown"))

        filename = "unknown"

        try:
            file_info = client.files.retrieve(file_id)
            filename = str(getattr(file_info, "filename", "unknown"))
        except Exception:
            filename = f"filename取得失敗: {file_id}"

        file_list.append(
            {
                "filename": filename,
                "file_id": file_id,
                "status": status,
            }
        )

    return file_list


def remove_file_from_vector_store(
    client: OpenAI,
    vector_store_id: str,
    file_id: str,
) -> dict[str, str]:
    """
    指定したファイルをVector Storeの検索対象から外す。
    注意: OpenAI上のファイル本体を削除するのではなく、
    このVector Storeとの紐づきを外すだけ。
    """
    deleted = client.vector_stores.files.delete(
        vector_store_id=vector_store_id,
        file_id=file_id,
    )

    return {
        "file_id": str(getattr(deleted, "id", file_id)),
        "deleted": str(getattr(deleted, "deleted", "unknown")),
    }


def ask_rag(
    client: OpenAI,
    question: str,
    model: str,
    vector_store_id: str,
    max_num_results: int,
    answer_style: str,
) -> tuple[str, list[dict[str, str]]]:
    """
    File Searchを使ってVector Store内の資料を検索し、
    その結果に基づいて回答を生成する。

    簡易ガード:
    - 検索結果が0件の場合は、モデルの回答をそのまま使わず、
      「資料内では確認できません」という回答に差し替える。
    """
    style_instruction = get_answer_style_instruction(answer_style)

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "あなたは、与えられた資料だけを根拠に回答するRAGアシスタントです。"
                    "資料内に根拠が見つからない場合は、推測せず"
                    "「資料内では確認できません」と答えてください。"
                    "回答は日本語で、分かりやすく書いてください。"
                    "可能であれば、根拠となる内容に沿って説明してください。"
                    f"{style_instruction}"
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        tools=[
            {
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
                "max_num_results": max_num_results,
            }
        ],
        include=["file_search_call.results"],
    )

    answer = response.output_text
    search_results = extract_file_search_results(response)

    if not search_results:
        guarded_answer = (
            "資料内では確認できません。"
            "関連する根拠候補が取得できなかったため、"
            "登録資料に基づく回答は控えます。"
        )

        return guarded_answer, search_results

    return answer, search_results