import json
from datetime import datetime
from pathlib import Path
from typing import Any


APP_DATA_DIR = Path("app_data")
VECTOR_STORE_REGISTRY_PATH = APP_DATA_DIR / "vector_stores.json"
QA_LOGS_PATH = APP_DATA_DIR / "qa_logs.jsonl"


def now_iso() -> str:
    """
    現在時刻をISO形式の文字列で返す。
    """
    return datetime.now().isoformat(timespec="seconds")


def load_vector_store_registry() -> list[dict[str, str]]:
    """
    ローカルに保存されたVector Store管理情報を読み込む。
    """
    if not VECTOR_STORE_REGISTRY_PATH.exists():
        return []

    try:
        text = VECTOR_STORE_REGISTRY_PATH.read_text(encoding="utf-8")
        data = json.loads(text)

        if isinstance(data, list):
            return data

        return []
    except Exception:
        return []


def save_vector_store_registry(registry: list[dict[str, str]]) -> None:
    """
    Vector Store管理情報をローカルJSONに保存する。
    """
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    VECTOR_STORE_REGISTRY_PATH.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def upsert_vector_store_registry(
    vector_store_id: str,
    name: str,
    memo: str = "",
    source: str = "app",
) -> None:
    """
    Vector Store管理情報を追加・更新する。
    すでに同じIDがある場合は、名前やメモを更新する。
    """
    registry = load_vector_store_registry()
    current_time = now_iso()

    for item in registry:
        if item.get("vector_store_id") == vector_store_id:
            item["name"] = name
            item["memo"] = memo
            item["source"] = source
            item["updated_at"] = current_time
            save_vector_store_registry(registry)
            return

    registry.append(
        {
            "name": name,
            "vector_store_id": vector_store_id,
            "memo": memo,
            "source": source,
            "created_at": current_time,
            "updated_at": current_time,
            "last_used_at": "",
        }
    )

    save_vector_store_registry(registry)


def delete_vector_store_from_registry(vector_store_id: str) -> None:
    """
    ローカル管理台帳からVector Store情報を削除する。
    注意: OpenAI上のVector Store本体は削除しない。
    """
    registry = load_vector_store_registry()

    updated_registry = [
        item for item in registry if item.get("vector_store_id") != vector_store_id
    ]

    save_vector_store_registry(updated_registry)


def mark_vector_store_used(vector_store_id: str) -> None:
    """
    使用したVector Storeの最終使用日時を更新する。
    """
    registry = load_vector_store_registry()
    current_time = now_iso()

    changed = False

    for item in registry:
        if item.get("vector_store_id") == vector_store_id:
            item["last_used_at"] = current_time
            item["updated_at"] = current_time
            changed = True
            break

    if changed:
        save_vector_store_registry(registry)


def append_qa_log(log_item: dict[str, Any]) -> None:
    """
    質問・回答ログをJSONL形式でローカル保存する。
    1行に1件のJSONを保存する。
    """
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with QA_LOGS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_item, ensure_ascii=False) + "\n")


def load_qa_logs(limit: int = 50) -> list[dict[str, Any]]:
    """
    ローカル保存された質問・回答ログを読み込む。
    新しいログほど上に表示するため、逆順で返す。
    """
    if not QA_LOGS_PATH.exists():
        return []

    logs: list[dict[str, Any]] = []

    try:
        with QA_LOGS_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        logs.append(item)
                except json.JSONDecodeError:
                    continue

    except Exception:
        return []

    logs.reverse()

    if limit <= 0:
        return logs

    return logs[:limit]


def create_saved_logs_json(limit: int = 0) -> str:
    """
    保存済み質問ログをJSON文字列として出力する。
    limit=0 の場合は全件出力する。
    """
    logs = load_qa_logs(limit=limit)

    export_data = {
        "app_name": "RAG Assistant Prototype",
        "exported_at": now_iso(),
        "logs": logs,
    }

    return json.dumps(export_data, ensure_ascii=False, indent=2)


def clear_qa_logs() -> None:
    """
    ローカル保存された質問ログを削除する。
    """
    if QA_LOGS_PATH.exists():
        QA_LOGS_PATH.unlink()