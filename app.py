import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


TEMP_UPLOAD_DIR = Path("temp_uploads")


def get_env_value(name: str, default: str | None = None) -> str:
    """
    .env から環境変数を読み込むための補助関数。
    必須項目が空の場合はエラーにする。
    """
    value = os.getenv(name, default)

    if value is None or value == "":
        raise ValueError(f"{name} が .env に設定されていません。")

    return value


@st.cache_resource
def get_openai_client(api_key: str) -> OpenAI:
    """
    OpenAIクライアントを作成する。
    Streamlitの再実行ごとに毎回作り直さないように cache_resource を使う。
    """
    return OpenAI(api_key=api_key)


def init_session_state() -> None:
    """
    Streamlitのセッション状態を初期化する。
    質問・回答履歴など、画面操作中に保持したい情報を入れる。
    """
    if "qa_history" not in st.session_state:
        st.session_state["qa_history"] = []


def add_qa_history(
    question: str,
    answer: str,
    search_results: list[dict[str, str]],
) -> None:
    """
    質問・回答・根拠候補を履歴に追加する。
    新しい履歴ほど上に表示したいので、先頭に追加する。
    """
    history_item = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "answer": answer,
        "search_results": search_results,
    }

    st.session_state["qa_history"].insert(0, history_item)


def create_history_json() -> str:
    """
    質問・回答履歴をJSON文字列として出力する。
    発表準備や開発ログに使えるようにする。
    """
    export_data = {
        "app_name": "RAG Assistant Prototype",
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "history": st.session_state.get("qa_history", []),
    }

    return json.dumps(export_data, ensure_ascii=False, indent=2)


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


def upload_file_to_vector_store(
    client: OpenAI,
    uploaded_file: Any,
    vector_store_id: str,
) -> dict[str, str]:
    """
    StreamlitでアップロードされたファイルをOpenAIにアップロードし、
    既存のVector Storeに追加する。
    """
    TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    temp_path = TEMP_UPLOAD_DIR / uploaded_file.name

    # Streamlitのアップロードファイルを一時保存する
    temp_path.write_bytes(uploaded_file.getvalue())

    # 1. OpenAI Files API にアップロード
    with temp_path.open("rb") as f:
        openai_file = client.files.create(
            file=f,
            purpose="assistants",
        )

    # 2. 既存のVector Storeに追加
    vector_store_file = client.vector_stores.files.create(
        vector_store_id=vector_store_id,
        file_id=openai_file.id,
    )

    # 3. 処理完了まで待つ
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
        "filename": uploaded_file.name,
        "file_id": openai_file.id,
        "vector_store_id": vector_store_id,
        "status": status,
    }


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


def ask_rag(
    client: OpenAI,
    question: str,
    model: str,
    vector_store_id: str,
) -> tuple[str, list[dict[str, str]]]:
    """
    File Searchを使ってVector Store内の資料を検索し、
    その結果に基づいて回答を生成する。
    """
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "あなたは、与えられた資料だけを根拠に回答するRAGアシスタントです。"
                    "資料内に根拠が見つからない場合は、推測せず"
                    "「資料内では確認できません」と答えてください。"
                    "回答は日本語で、簡潔かつ分かりやすく書いてください。"
                    "可能であれば、根拠となる内容に沿って説明してください。"
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
                "max_num_results": 5,
            }
        ],
        include=["file_search_call.results"],
    )

    answer = response.output_text
    search_results = extract_file_search_results(response)

    return answer, search_results


def render_search_results(search_results: list[dict[str, str]]) -> None:
    """
    検索された根拠候補を画面に表示する。
    回答直後と履歴表示の両方で使えるように関数化している。
    """
    if not search_results:
        st.info("検索結果の詳細は取得できませんでした。")
        return

    for i, result in enumerate(search_results, start=1):
        with st.expander(f"根拠候補 {i}: {result['filename']}"):
            st.write("score")
            st.code(result["score"])

            st.write("本文プレビュー")
            preview = result.get("preview", "")

            if preview:
                st.write(preview[:1500])
            else:
                st.info("本文プレビューは取得できませんでした。")


def render_qa_history() -> None:
    """
    質問・回答履歴を画面下部に表示する。
    """
    st.markdown("---")
    st.subheader("質問・回答履歴")

    qa_history = st.session_state.get("qa_history", [])

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("履歴をクリア"):
            st.session_state["qa_history"] = []
            st.rerun()

    with col2:
        if qa_history:
            history_json = create_history_json()

            st.download_button(
                label="履歴をJSONでダウンロード",
                data=history_json,
                file_name="qa_history.json",
                mime="application/json",
            )

    if not qa_history:
        st.info("まだ質問履歴はありません。")
        return

    st.write(f"{len(qa_history)} 件の質問履歴があります。")

    for i, item in enumerate(qa_history, start=1):
        timestamp = item.get("timestamp", "no timestamp")
        question_text = item["question"]
        answer_text = item["answer"]
        search_results = item["search_results"]

        with st.expander(f"履歴 {i}: {question_text[:40]}"):
            st.caption(f"日時: {timestamp}")

            st.markdown("**質問**")
            st.write(question_text)

            st.markdown("**回答**")
            st.write(answer_text)

            st.markdown("**根拠候補**")
            st.write(f"{len(search_results)} 件")

            render_search_results(search_results)


def main():
    load_dotenv()

    st.set_page_config(
        page_title="RAG Assistant Prototype",
        page_icon="📚",
        layout="wide",
    )

    init_session_state()

    st.title("📚 RAG Assistant Prototype")
    st.caption("閉じた資料に基づいて、根拠つきで質問応答するRAGアプリ")

    try:
        api_key = get_env_value("OPENAI_API_KEY")
        model = get_env_value("OPENAI_MODEL", "gpt-5.5")
        vector_store_id = get_env_value("VECTOR_STORE_ID")
    except Exception as e:
        st.error("環境変数の読み込みに失敗しました。`.env` を確認してください。")
        st.exception(e)
        return

    client = get_openai_client(api_key)

    with st.sidebar:
        st.header("設定")

        st.write("使用モデル")
        st.code(model)

        st.write("Vector Store ID")
        st.code(vector_store_id)

        st.markdown("---")
        st.header("資料アップロード")

        uploaded_file = st.file_uploader(
            "検索対象に追加する資料を選んでください",
            type=["txt", "md", "pdf"],
        )

        if uploaded_file is not None:
            st.write("選択中のファイル")
            st.code(uploaded_file.name)

            if st.button("この資料をVector Storeに追加"):
                with st.spinner("資料をアップロードして検索可能にしています..."):
                    try:
                        result = upload_file_to_vector_store(
                            client=client,
                            uploaded_file=uploaded_file,
                            vector_store_id=vector_store_id,
                        )

                        st.success("資料の追加が完了しました。")
                        st.write("追加結果")
                        st.json(result)

                        # ファイル一覧のキャッシュを消す。
                        # 次に一覧更新ボタンを押したとき、新しい状態を取得できる。
                        st.session_state.pop("registered_files", None)

                    except Exception as e:
                        st.error("資料の追加中にエラーが発生しました。")
                        st.exception(e)

        st.markdown("---")
        st.header("登録済み資料")

        if st.button("登録済み資料一覧を更新"):
            with st.spinner("登録済み資料を取得しています..."):
                try:
                    registered_files = list_vector_store_files(
                        client=client,
                        vector_store_id=vector_store_id,
                    )

                    st.session_state["registered_files"] = registered_files

                except Exception as e:
                    st.error("登録済み資料の取得中にエラーが発生しました。")
                    st.exception(e)

        registered_files = st.session_state.get("registered_files", [])

        if registered_files:
            st.write(f"{len(registered_files)} 件の資料が登録されています。")

            for file in registered_files:
                status = file["status"]

                if status == "completed":
                    status_label = "✅ completed"
                else:
                    status_label = f"⚠️ {status}"

                with st.expander(file["filename"]):
                    st.write("status")
                    st.code(status_label)

                    st.write("file_id")
                    st.code(file["file_id"])
        else:
            st.info("まだ資料一覧を取得していません。")

        st.markdown("---")
        st.caption(
            "現在は、登録済みVector Storeに資料を追加し、その資料群に対して質問する構成です。"
        )

    st.subheader("質問")

    example_questions = [
        "この展示の目的は何ですか？",
        "RAGを使う理由は何ですか？",
        "スキャンPDFを扱うときの課題は何ですか？",
        "この展示の入場料はいくらですか？",
    ]

    selected_example = st.selectbox(
        "サンプル質問を選ぶ",
        [""] + example_questions,
    )

    default_question = selected_example if selected_example else ""

    with st.form("question_form"):
        question = st.text_area(
            "資料に対する質問を入力してください",
            value=default_question,
            height=120,
        )

        submitted = st.form_submit_button("質問する")

    if submitted:
        if not question.strip():
            st.warning("質問を入力してください。")
            render_qa_history()
            return

        with st.spinner("資料を検索して回答を生成しています..."):
            try:
                answer, search_results = ask_rag(
                    client=client,
                    question=question,
                    model=model,
                    vector_store_id=vector_store_id,
                )
            except Exception as e:
                st.error("質問応答中にエラーが発生しました。")
                st.exception(e)
                render_qa_history()
                return

        add_qa_history(
            question=question,
            answer=answer,
            search_results=search_results,
        )

        st.subheader("回答")
        st.write(answer)

        st.subheader("検索された根拠候補")
        render_search_results(search_results)

    render_qa_history()


if __name__ == "__main__":
    main()