import os
import time
from pathlib import Path
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


TEMP_UPLOAD_DIR = Path("temp_uploads")


def get_env_value(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)

    if value is None or value == "":
        raise ValueError(f"{name} が .env に設定されていません。")

    return value


@st.cache_resource
def get_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def extract_file_search_results(response: Any) -> list[dict[str, str]]:
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


def ask_rag(
    client: OpenAI,
    question: str,
    model: str,
    vector_store_id: str,
) -> tuple[str, list[dict[str, str]]]:
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


def main():
    load_dotenv()

    st.set_page_config(
        page_title="RAG Assistant Prototype",
        page_icon="📚",
        layout="wide",
    )

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

                    except Exception as e:
                        st.error("資料の追加中にエラーが発生しました。")
                        st.exception(e)

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
                return

        st.subheader("回答")
        st.write(answer)

        st.subheader("検索された根拠候補")

        if not search_results:
            st.info("検索結果の詳細は取得できませんでした。")
        else:
            for i, result in enumerate(search_results, start=1):
                with st.expander(f"根拠候補 {i}: {result['filename']}"):
                    st.write("score")
                    st.code(result["score"])

                    st.write("本文プレビュー")
                    st.write(result["preview"][:1500])


if __name__ == "__main__":
    main()