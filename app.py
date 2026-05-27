import json
import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from rag_client import (
    ask_rag,
    create_vector_store,
    is_valid_vector_store_id,
    list_vector_store_files,
    remove_file_from_vector_store,
    upload_file_to_vector_store,
    upload_text_to_vector_store,
)

from storage import (
    append_qa_log,
    clear_qa_logs,
    create_saved_logs_json,
    delete_vector_store_from_registry,
    load_qa_logs,
    load_vector_store_registry,
    mark_vector_store_used,
    now_iso,
    upsert_vector_store_registry,
)


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
    また、.env のVector Storeも管理台帳に登録する。
    """
    if not st.session_state.get("active_vector_store_id"):
        st.session_state["active_vector_store_id"] = env_vector_store_id

    upsert_vector_store_registry(
        vector_store_id=env_vector_store_id,
        name="env_default_store",
        memo=".env に設定されているデフォルトのVector Store",
        source="env",
    )


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
    画面上の履歴に加えて、ローカルログにも保存する。
    """
    history_item = {
        "timestamp": now_iso(),
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
    append_qa_log(history_item)


def create_history_json() -> str:
    """
    現在のセッション中の質問・回答履歴をJSON文字列として出力する。
    """
    export_data = {
        "app_name": "RAG Assistant Prototype",
        "exported_at": now_iso(),
        "history": st.session_state.get("qa_history", []),
    }

    return json.dumps(export_data, ensure_ascii=False, indent=2)


def render_search_results(search_results: list[dict[str, str]]) -> None:
    """
    検索された根拠候補を画面に表示する。
    scoreは厳密な正解確率ではなく、関連度の目安として表示する。
    """
    if not search_results:
        st.info("検索結果の詳細は取得できませんでした。")
        return

    for i, result in enumerate(search_results, start=1):
        filename = result.get("filename", "unknown")
        score_text = str(result.get("score", "unknown"))

        with st.expander(f"根拠候補 {i}: {filename}"):
            st.write("score")
            st.code(score_text)

            try:
                score_value = float(score_text)

                if score_value >= 0.75:
                    st.success("関連度メモ: 高めの可能性があります。")
                elif score_value >= 0.5:
                    st.info("関連度メモ: 中程度の可能性があります。")
                else:
                    st.warning("関連度メモ: 低めの可能性があります。")
            except ValueError:
                st.caption(
                    "関連度メモ: scoreの形式が不明なため、関連度ラベルは表示していません。"
                )

            st.caption(
                "注: scoreは検索結果の目安であり、回答の正しさを保証する値ではありません。"
            )

            st.write("本文プレビュー")
            preview = result.get("preview", "")

            if preview:
                st.write(preview[:1500])
            else:
                st.info("本文プレビューは取得できませんでした。")


def render_reliability_note(
    answer: str,
    search_results: list[dict[str, str]],
) -> None:
    """
    回答の信頼性に関する簡易メモを表示する。
    これは厳密な回答可能性判定ではなく、発表・検証用の補助表示。
    """
    st.subheader("回答信頼性メモ")

    no_answer_phrases = [
        "資料内では確認できません",
        "資料内で確認できません",
        "確認できません",
        "記載されていません",
        "情報はありません",
    ]

    seems_no_answer = any(phrase in answer for phrase in no_answer_phrases)

    if not search_results:
        st.warning(
            "根拠候補が取得できませんでした。"
            "この回答は登録資料に基づいているか確認が難しいため、注意が必要です。"
        )
        return

    st.info(
        f"{len(search_results)} 件の根拠候補が検索されました。"
        "回答は、これらの候補をもとに生成されています。"
    )

    if seems_no_answer:
        st.success(
            "回答文に「資料内では確認できません」系の表現が含まれています。"
            "資料外質問に対して、推測を抑制できている可能性があります。"
        )
    else:
        st.info(
            "回答文は通常回答として生成されています。"
            "現段階では、回答が本当に資料内根拠だけで十分かは、"
            "下の根拠候補を見て確認する必要があります。"
        )

    missing_preview_count = sum(
        1 for result in search_results if not result.get("preview")
    )

    if missing_preview_count > 0:
        st.caption(
            f"補足: {missing_preview_count} 件の根拠候補では本文プレビューを取得できませんでした。"
            "APIの返却形式によって、本文断片が表示できない場合があります。"
        )


def render_history_item(index: int, item: dict[str, Any]) -> None:
    """
    1件分の質問・回答履歴を表示する。
    """
    timestamp = item.get("timestamp", "no timestamp")
    question_text = item.get("question", "")
    answer_text = item.get("answer", "")
    search_results = item.get("search_results", [])
    settings = item.get("settings", {})

    max_num_results = settings.get("max_num_results", "unknown")
    answer_style = settings.get("answer_style", "unknown")
    vector_store_id = settings.get("vector_store_id", "unknown")

    with st.expander(f"履歴 {index}: {question_text[:40]}"):
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


def render_session_history() -> None:
    """
    現在のアプリ起動中の質問・回答履歴を表示する。
    """
    st.subheader("このセッションの質問・回答履歴")

    qa_history = st.session_state.get("qa_history", [])

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("セッション履歴をクリア"):
            st.session_state["qa_history"] = []
            st.rerun()

    with col2:
        if qa_history:
            history_json = create_history_json()

            st.download_button(
                label="セッション履歴をJSONでダウンロード",
                data=history_json,
                file_name="qa_history_session.json",
                mime="application/json",
            )

    if not qa_history:
        st.info("このセッションの質問履歴はまだありません。")
        return

    st.write(f"{len(qa_history)} 件の質問履歴があります。")

    for i, item in enumerate(qa_history, start=1):
        render_history_item(i, item)


def render_saved_logs() -> None:
    """
    ローカル保存済みの質問ログを表示する。
    """
    st.subheader("ローカル保存済み質問ログ")

    col1, col2 = st.columns([1, 2])

    saved_logs = load_qa_logs(limit=50)

    with col1:
        if st.button("保存済みログをクリア"):
            clear_qa_logs()
            st.success("保存済みログを削除しました。")
            st.rerun()

    with col2:
        if saved_logs:
            logs_json = create_saved_logs_json(limit=0)

            st.download_button(
                label="保存済みログをJSONでダウンロード",
                data=logs_json,
                file_name="qa_logs_saved.json",
                mime="application/json",
            )

    if not saved_logs:
        st.info("ローカル保存済みログはまだありません。")
        return

    st.write(f"最新50件のうち {len(saved_logs)} 件を表示しています。")

    for i, item in enumerate(saved_logs, start=1):
        render_history_item(i, item)


def render_history_page() -> None:
    """
    履歴タブ全体を表示する。
    """
    st.header("履歴")

    tab_session, tab_saved = st.tabs(
        ["このセッションの履歴", "保存済みログ"]
    )

    with tab_session:
        render_session_history()

    with tab_saved:
        render_saved_logs()


def render_app_overview(
    model: str,
    env_vector_store_id: str,
    active_vector_store_id: str,
) -> None:
    """
    アプリ概要タブを表示する。
    背景・問題意識・提案方針・現状・課題・展望を発表向けに整理する。
    """
    st.header("アプリ概要")

    st.markdown(
        """
        このアプリは、登録した資料に基づいて質問応答を行う
        **根拠提示型RAGアプリのプロトタイプ**です。

        展示施設・教育施設・研究室などで、内部資料や解説文書をもとに
        質問応答を行う情報支援システムを想定しています。
        """
    )

    st.markdown("---")

    st.subheader("1. 背景")

    st.markdown(
        """
        展示施設や教育施設には、展示解説、過去資料、マニュアル、研究メモなど、
        多様な知識が蓄積されています。

        しかし、それらの資料はPDF、Word、テキスト、紙資料、スキャン画像など
        さまざまな形式で存在しており、必要な情報をすぐに探すことは簡単ではありません。

        また、利用者が展示や資料を見ながら疑問を持った場合に、
        いつでも職員や専門家が個別に対応できるとは限りません。
        """
    )

    st.subheader("2. 問題意識")

    st.markdown(
        """
        通常の大規模言語モデルは幅広い知識を持っていますが、
        施設固有の資料や内部文書に基づく回答には限界があります。

        特に、以下のような課題があります。

        - 登録資料にない内容を推測で答えてしまう可能性がある
        - 回答の根拠が分かりにくい
        - 施設が管理する資料や知識を外部に出したくない場合がある
        - スキャンPDFや画像資料はそのままでは検索対象にしづらい
        """
    )

    st.subheader("3. 提案方針")

    st.markdown(
        """
        本プロトタイプでは、**閉じた資料セットに基づくRAG型質問応答**を試作しています。

        RAGでは、ユーザーの質問に関連する資料を検索し、
        その検索結果をもとに回答を生成します。

        これにより、一般的な雑談AIではなく、
        あらかじめ登録した資料に基づいて回答する情報支援アプリを目指します。
        """
    )

    st.info(
        "現段階ではOpenAI File Search / Vector Storeを用いた初期プロトタイプです。"
        "将来的にはローカルVector DBやローカルLLMによる閉域構成への発展を想定しています。"
    )

    st.subheader("4. 現在の試作内容")

    st.markdown(
        """
        現在のプロトタイプでは、以下の流れを実装しています。

        1. 資料をアップロード、または画面上で直接テキスト登録する
        2. 資料をVector Storeに登録する
        3. 登録資料に対して質問する
        4. 関連する根拠候補を検索する
        5. 検索結果に基づいて回答を生成する
        6. 回答・根拠候補・信頼性メモ・質問ログを確認する
        """
    )

    st.subheader("5. 実装済み機能")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(
            """
            **資料管理**

            - txt / md / pdf のアップロード
            - 直接入力テキストの資料登録
            - 登録済み資料一覧の表示
            - 資料を検索対象から外す機能
            - 資料追加後の一覧自動更新
            """
        )

        st.markdown(
            """
            **Vector Store管理**

            - Vector Storeの新規作成
            - 使用するVector Storeの切り替え
            - 既存Vector Store IDの手入力登録
            - ローカル台帳による管理
            """
        )

    with col_right:
        st.markdown(
            """
            **質問応答**

            - 登録資料に基づく質問応答
            - 根拠候補の表示
            - scoreの関連度メモ表示
            - 回答信頼性メモ
            - 検索結果0件時の簡易ガード
            """
        )

        st.markdown(
            """
            **ログ・分析**

            - セッション履歴の表示
            - 質問・回答ログのローカル保存
            - 保存済みログのJSONダウンロード
            - 質問ログの簡易分析
            - デモ手順タブ
            """
        )

    st.subheader("6. 現在の設定")

    col1, col2 = st.columns(2)

    with col1:
        st.write("使用モデル")
        st.code(model)

        st.write(".env の Vector Store ID")
        st.code(env_vector_store_id)

    with col2:
        st.write("現在使用中の Vector Store ID")
        st.code(active_vector_store_id)

        if active_vector_store_id != env_vector_store_id:
            st.warning(
                "現在は .env とは別のVector Storeを一時的に使用しています。"
            )
        else:
            st.success(".env に設定されたVector Storeを使用しています。")

    st.subheader("7. 現時点の限界")

    st.markdown(
        """
        現在の実装は初期プロトタイプであり、実運用にはまだ課題があります。

        - 現在のVector StoreはOpenAI側に作成される
        - スキャンPDFや画像PDFはOCRなしでは扱いにくい
        - 回答可能性判定はまだ簡易的である
        - 根拠箇所の本文プレビューが安定して取得できない場合がある
        - 質問ログには個人情報や利用状況が含まれる可能性がある
        - 実運用ではアクセス制御やログ管理方針が必要になる
        """
    )

    st.warning(
        "現在の資料外回答抑制は、プロンプト制御と検索結果0件時の簡易ガードに基づくものです。"
        "今後は検索スコアや根拠文との一致度を使った回答可能性判定が必要です。"
    )

    st.subheader("8. 今後の展望")

    st.markdown(
        """
        今後は、以下の方向に発展させることを想定しています。

        - OCRによるスキャンPDF・画像資料のテキスト化
        - 検索スコアや根拠文を用いた回答可能性判定
        - 根拠箇所の引用表示の安定化
        - 質問ログのEmbeddingクラスタリング
        - 類似質問のグルーピングによる利用者関心の分析
        - ローカルVector DBへの移行
        - ローカルLLMによる閉域構成
        - 音声入力・音声読み上げによる対話体験の拡張
        """
    )

    st.success(
        "本プロトタイプでは、まずOpenAI版でRAGアプリの体験・機能要件・課題を確認し、"
        "その後、より実用的な閉域RAG構成へ発展させることを目指しています。"
    )

def render_vector_store_registry_panel() -> None:
    """
    ローカル保存されたVector Store管理情報を表示し、切り替えできるようにする。
    """
    st.header("Vector Store管理")

    st.markdown(
        """
        ここでは、OpenAI上に作成済みのVector Store IDをローカル台帳として管理します。
        Vector Store本体をローカル保存するのではなく、ID・名前・用途メモを保存して切り替えやすくします。
        """
    )

    st.subheader("既存Vector Store IDを登録")

    manual_name = st.text_input(
        "表示名",
        value="existing_vector_store",
        key="manual_vector_store_name",
    )

    manual_vector_store_id = st.text_input(
        "Vector Store ID",
        placeholder="vs_...",
        key="manual_vector_store_id",
    )

    manual_memo = st.text_area(
        "用途メモ",
        value="過去に作成したVector Store",
        height=80,
        key="manual_vector_store_memo",
    )

    col_add, col_switch = st.columns(2)

    with col_add:
        if st.button("管理台帳に追加"):
            target_id = manual_vector_store_id.strip()

            if not is_valid_vector_store_id(target_id):
                st.warning("Vector Store IDは `vs_` から始まる形式で入力してください。")
            elif not manual_name.strip():
                st.warning("表示名を入力してください。")
            else:
                upsert_vector_store_registry(
                    vector_store_id=target_id,
                    name=manual_name.strip(),
                    memo=manual_memo.strip(),
                    source="manual",
                )

                st.success("Vector Storeを管理台帳に追加しました。")
                st.rerun()

    with col_switch:
        if st.button("追加してすぐ使用"):
            target_id = manual_vector_store_id.strip()

            if not is_valid_vector_store_id(target_id):
                st.warning("Vector Store IDは `vs_` から始まる形式で入力してください。")
            elif not manual_name.strip():
                st.warning("表示名を入力してください。")
            else:
                upsert_vector_store_registry(
                    vector_store_id=target_id,
                    name=manual_name.strip(),
                    memo=manual_memo.strip(),
                    source="manual",
                )

                st.session_state["active_vector_store_id"] = target_id
                st.session_state["registered_files"] = []
                mark_vector_store_used(target_id)

                st.success("Vector Storeを管理台帳に追加し、使用対象に切り替えました。")
                st.rerun()

    st.markdown("---")

    registry = load_vector_store_registry()

    if not registry:
        st.info("まだVector Store管理情報は保存されていません。")
        return

    st.subheader("登録済みVector Store")

    st.write(f"{len(registry)} 件のVector Storeがローカルに記録されています。")

    for i, item in enumerate(registry, start=1):
        name = item.get("name", "unknown")
        vector_store_id = item.get("vector_store_id", "unknown")
        memo = item.get("memo", "")
        source = item.get("source", "unknown")
        created_at = item.get("created_at", "")
        updated_at = item.get("updated_at", "")
        last_used_at = item.get("last_used_at", "")

        with st.expander(f"{name} / {vector_store_id}"):
            st.write("用途メモ")
            st.write(memo if memo else "メモなし")

            st.write("source")
            st.code(source)

            st.write("created_at")
            st.code(created_at if created_at else "unknown")

            st.write("updated_at")
            st.code(updated_at if updated_at else "unknown")

            st.write("last_used_at")
            st.code(last_used_at if last_used_at else "まだ使用記録なし")

            col_use, col_remove = st.columns(2)

            with col_use:
                if st.button(
                    "このVector Storeに切り替える",
                    key=f"switch_vector_store_{i}_{vector_store_id}",
                ):
                    st.session_state["active_vector_store_id"] = vector_store_id
                    st.session_state["registered_files"] = []
                    mark_vector_store_used(vector_store_id)
                    st.success("使用するVector Storeを切り替えました。")
                    st.rerun()

            with col_remove:
                if st.button(
                    "ローカル台帳から削除",
                    key=f"remove_registry_{i}_{vector_store_id}",
                ):
                    if vector_store_id == st.session_state.get("active_vector_store_id"):
                        st.warning(
                            "現在使用中のVector Storeは台帳から削除できません。"
                            "別のVector Storeに切り替えてから削除してください。"
                        )
                    else:
                        delete_vector_store_from_registry(vector_store_id)
                        st.success("ローカル台帳から削除しました。")
                        st.rerun()


def render_sidebar(
    client: OpenAI,
    model: str,
    env_vector_store_id: str,
    active_vector_store_id: str,
) -> tuple[int, str]:
    """
    サイドバーを描画し、回答設定を返す。
    """
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
            mark_vector_store_used(env_vector_store_id)
            st.rerun()

        st.markdown("---")
        st.header("Vector Store作成")

        new_vector_store_name = st.text_input(
            "新しいVector Store名",
            value="rag_assistant_demo_store",
        )

        new_vector_store_memo = st.text_area(
            "用途メモ",
            value="発表デモ用の資料セット",
            height=80,
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

                        new_vector_store_id = result["vector_store_id"]

                        upsert_vector_store_registry(
                            vector_store_id=new_vector_store_id,
                            name=new_vector_store_name.strip(),
                            memo=new_vector_store_memo.strip(),
                            source="app",
                        )

                        mark_vector_store_used(new_vector_store_id)

                        st.session_state["active_vector_store_id"] = new_vector_store_id
                        st.session_state["registered_files"] = []

                        st.success("新しいVector Storeを作成し、使用対象に切り替えました。")
                        st.json(result)

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
        st.header("資料登録")

        st.subheader("ファイルから登録")

        uploaded_file = st.file_uploader(
            "検索対象に追加する資料を選んでください",
            type=["txt", "md", "pdf"],
        )

        if uploaded_file is not None:
            st.write("選択中のファイル")
            st.code(uploaded_file.name)

            if st.button("このファイルをVector Storeに追加"):
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

                        st.session_state["registered_files"] = []

                    except Exception as e:
                        st.error("資料の追加中にエラーが発生しました。")
                        st.exception(e)

        st.markdown("---")
        st.subheader("テキストを直接登録")

        direct_text_title = st.text_input(
            "資料名",
            value="direct_note",
            help="Vector Storeに登録される一時ファイル名の一部になります。",
        )

        direct_text = st.text_area(
            "登録するテキスト",
            height=180,
            placeholder=(
                "ここに資料として登録したい文章を入力してください。\n"
                "例: この展示では、星座の見つけ方と季節ごとの星空の違いを説明する。"
            ),
        )

        if st.button("このテキストをVector Storeに追加"):
            if not direct_text.strip():
                st.warning("登録するテキストを入力してください。")
            else:
                with st.spinner("入力テキストを資料として登録しています..."):
                    try:
                        result = upload_text_to_vector_store(
                            client=client,
                            title=direct_text_title,
                            text=direct_text,
                            vector_store_id=active_vector_store_id,
                        )

                        st.success("テキスト資料の追加が完了しました。")
                        st.write("追加結果")
                        st.json(result)

                        st.session_state["registered_files"] = []

                    except Exception as e:
                        st.error("テキスト資料の追加中にエラーが発生しました。")
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

    return max_num_results, answer_style


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

    max_num_results, answer_style = render_sidebar(
        client=client,
        model=model,
        env_vector_store_id=env_vector_store_id,
        active_vector_store_id=active_vector_store_id,
    )

    tab_question, tab_history, tab_vector_stores, tab_overview = st.tabs(
        ["質問", "履歴", "Vector Store管理", "アプリ概要"]
    )

    with tab_question:
        st.subheader("質問")

        demo_questions = {
            "1. システムの目的": "このシステムの目的は何ですか？",
            "2. RAGを使う理由": "なぜRAGを使うのですか？",
            "3. 季節ごとの星座": "春と冬の代表的な星座を教えてください。",
            "4. 星の色の違い": "星の色が違うのはなぜですか？",
            "5. スキャンPDFの課題": "スキャンPDFを扱うときの課題は何ですか？",
            "6. ローカルLLMの必要性": "科学館実用版でローカルLLMが重要になる理由は何ですか？",
            "7. 今後の開発方針": "今後の開発方針を短期・中期・長期に分けて教えてください。",
            "8. 資料外質問：入場料": "この展示の入場料はいくらですか？",
            "9. 資料外質問：開館時間": "科学館の開館時間を教えてください。",
        }

        selected_label = st.selectbox(
            "デモ質問を選ぶ",
            ["自由入力"] + list(demo_questions.keys()),
        )

        default_question = (
            ""
            if selected_label == "自由入力"
            else demo_questions[selected_label]
        )

        if selected_label.startswith("8.") or selected_label.startswith("9."):
            st.info(
                "これは資料外質問の確認用です。資料に情報がないため、"
                "「資料内では確認できません」と返るのが期待される挙動です。"
            )
        else:
            st.caption("資料内に根拠がある想定のデモ質問です。")

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
            else:
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
                        return

                mark_vector_store_used(active_vector_store_id)

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

                render_reliability_note(
                    answer=answer,
                    search_results=search_results,
                )

                st.subheader("検索された根拠候補")
                render_search_results(search_results)

    with tab_history:
        render_history_page()

    with tab_vector_stores:
        render_vector_store_registry_panel()

    with tab_overview:
        render_app_overview(
            model=model,
            env_vector_store_id=env_vector_store_id,
            active_vector_store_id=active_vector_store_id,
        )


if __name__ == "__main__":
    main()
