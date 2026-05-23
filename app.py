import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


def get_env_value(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)

    if value is None or value == "":
        raise ValueError(f"{name} が .env に設定されていません。")

    return value


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


def ask_rag(question: str) -> tuple[str, list[dict[str, str]]]:
    load_dotenv()

    api_key = get_env_value("OPENAI_API_KEY")
    model = get_env_value("OPENAI_MODEL", "gpt-5.5")
    vector_store_id = get_env_value("VECTOR_STORE_ID")

    client = OpenAI(api_key=api_key)

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
    st.set_page_config(
        page_title="RAG Assistant Prototype",
        page_icon="📚",
        layout="wide",
    )

    st.title("📚 RAG Assistant Prototype")
    st.caption("閉じた資料に基づいて、根拠つきで質問応答するRAGアプリ")

    with st.sidebar:
        st.header("設定")

        load_dotenv()
        model = os.getenv("OPENAI_MODEL", "未設定")
        vector_store_id = os.getenv("VECTOR_STORE_ID", "未設定")

        st.write("使用モデル")
        st.code(model)

        st.write("Vector Store ID")
        st.code(vector_store_id)

        st.markdown("---")
        st.write("現在は、すでに登録済みのVector Storeを検索対象にしています。")
        st.write("ファイルアップロード機能は後続ステップで追加予定です。")

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
                answer, search_results = ask_rag(question)
            except Exception as e:
                st.error("エラーが発生しました。")
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