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

    if "registered_files" not in st.session_state:
        st.session_state["registered_files"] = []

    if "active_vector_store_id" not in st.session_state:
        st.session_state["active_vector_store_id"] = ""


def ensure_active_vector_store(env_vector_store_id: str) -> None:
    """
    使用中のVector Store IDが未設定なら、.env の VECTOR_STORE_ID を使う。
    """
    if not st.session_state.get("active_vector_store_id"):
        st.session_state["active_vector_store_id"] = env_vector_store_id


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


def add_qa_history(
    question: str,
    answer: str,
    search_results: list[dict[str, str]],
    max_num_results: int,
    answer_style: str,
    vector_store_id: str,
) -> None:
    """
    質問・回答・根拠候補・回答設定を履歴に追加する。
    新しい履歴ほど上に表示したいので、先頭に追加する。
    """
    history_item = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "answer": answer,
        "settings": {
            "max_num_results": max_num_results,
            "answer_style": answer_style,
            "vector_store_id": vector_store_id,
        },
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


def upload_file_to_vector_store(
    client: OpenAI,
    uploaded_file: Any,
    vector_store_id: str,
) -> dict[str, str]:
    """
    StreamlitでアップロードされたファイルをOpenAIにアップロードし、
    指定したVector Storeに追加する。
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

    # 2. Vector Storeに追加
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
        settings = item.get("settings", {})

        max_num_results = settings.get("max_num_results", "unknown")
        answer_style = settings.get("answer_style", "unknown")
        vector_store_id = settings.get("vector_store_id", "unknown")

        with st.expander(f"履歴 {i}: {question_text[:40]}"):
            st.caption(f"日時: {timestamp}")

            st.markdown("**回答設定**")
            st.write(f"- 検索件数: {max_num_results}")
            st.write(f"- 回答スタイル: {answer_style}")
            st.write(f"- 使用Vector Store: `{vector_store_id}`")

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
        env_vector_store_id = get_env_value("VECTOR_STORE_ID")
    except Exception as e:
        st.error("環境変数の読み込みに失敗しました。`.env` を確認してください。")
        st.exception(e)
        return

    ensure_active_vector_store(env_vector_store_id)

    client = get_openai_client(api_key)
    active_vector_store_id = st.session_state["active_vector_store_id"]

    with st.sidebar:
        st.header("設定")

        st.write("使用モデル")
        st.code(model)

        st.write(".env の Vector Store ID")
        st.code(env_vector_store_id)

        st.write("現在使用中の Vector Store ID")
        st.code(active_vector_store_id)

        if active_vector_store_id != env_vector_store_id:
            st.warning(
                "現在は .env とは別のVector Storeを一時的に使用しています。"
                "この設定を次回以降も使う場合は、.env の VECTOR_STORE_ID を更新してください。"
            )

        if st.button(".env のVector Storeに戻す"):
            st.session_state["active_vector_store_id"] = env_vector_store_id
            st.session_state["registered_files"] = []
            st.rerun()

        st.markdown("---")
        st.header("Vector Store作成")

        new_vector_store_name = st.text_input(
            "新しいVector Store名",
            value="rag_assistant_demo_store",
        )

        if st.button("新しいVector Storeを作成して使用"):
            if not new_vector_store_name.strip():
                st.warning("Vector Store名を入力してください。")
            else:
                with st.spinner("新しいVector Storeを作成しています..."):
                    try:
                        result = create_vector_store(
                            client=client,
                            name=new_vector_store_name.strip(),
                        )

                        st.session_state["active_vector_store_id"] = result[
                            "vector_store_id"
                        ]
                        st.session_state["registered_files"] = []

                        st.success("新しいVector Storeを作成し、使用対象に切り替えました。")
                        st.json(result)
                        st.info(
                            "このVector Storeを次回以降も使う場合は、"
                            ".env の VECTOR_STORE_ID にこのIDを貼り替えてください。"
                        )

                        st.rerun()

                    except Exception as e:
                        st.error("Vector Store作成中にエラーが発生しました。")
                        st.exception(e)

        st.markdown("---")
        st.header("回答設定")

        max_num_results = st.selectbox(
            "検索件数",
            options=[3, 5, 8, 10],
            index=1,
            help="質問に対して、Vector Storeから取得する根拠候補の最大件数です。",
        )

        answer_style = st.selectbox(
            "回答スタイル",
            options=["簡潔", "詳細", "箇条書き", "発表用"],
            index=0,
            help="同じ検索結果でも、回答の書き方を切り替えられます。",
        )

        st.caption(
            "検索件数を増やすと根拠候補は増えますが、関係の薄い情報も混ざる可能性があります。"
        )

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
                            vector_store_id=active_vector_store_id,
                        )

                        st.success("資料の追加が完了しました。")
                        st.write("追加結果")
                        st.json(result)

                        # ファイル一覧のキャッシュを消す。
                        # 次に一覧更新ボタンを押したとき、新しい状態を取得できる。
                        st.session_state["registered_files"] = []

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
                        vector_store_id=active_vector_store_id,
                    )

                    st.session_state["registered_files"] = registered_files

                except Exception as e:
                    st.error("登録済み資料の取得中にエラーが発生しました。")
                    st.exception(e)

        registered_files = st.session_state.get("registered_files", [])

        if registered_files:
            st.write(f"{len(registered_files)} 件の資料が登録されています。")

            for i, file in enumerate(registered_files, start=1):
                status = file["status"]
                filename = file["filename"]
                file_id = file["file_id"]

                if status == "completed":
                    status_label = "✅ completed"
                else:
                    status_label = f"⚠️ {status}"

                with st.expander(filename):
                    st.write("status")
                    st.code(status_label)

                    st.write("file_id")
                    st.code(file_id)

                    st.warning(
                        "この操作を行うと、この資料は現在のVector Storeの検索対象から外れます。"
                    )

                    delete_button_key = f"delete_{i}_{file_id}"

                    if st.button(
                        "この資料を検索対象から外す",
                        key=delete_button_key,
                    ):
                        with st.spinner("資料をVector Storeから外しています..."):
                            try:
                                delete_result = remove_file_from_vector_store(
                                    client=client,
                                    vector_store_id=active_vector_store_id,
                                    file_id=file_id,
                                )

                                st.success("資料を検索対象から外しました。")
                                st.json(delete_result)

                                st.session_state["registered_files"] = []
                                st.rerun()

                            except Exception as e:
                                st.error("資料の削除中にエラーが発生しました。")
                                st.exception(e)
        else:
            st.info("まだ資料一覧を取得していません。")

        st.markdown("---")
        st.caption(
            "現在使用中のVector Storeに資料を追加し、その資料群に対して質問する構成です。"
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
                    vector_store_id=active_vector_store_id,
                    max_num_results=max_num_results,
                    answer_style=answer_style,
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
            max_num_results=max_num_results,
            answer_style=answer_style,
            vector_store_id=active_vector_store_id,
        )

        st.subheader("回答")
        st.write(answer)

        st.subheader("検索された根拠候補")
        render_search_results(search_results)

    render_qa_history()


if __name__ == "__main__":
    main()