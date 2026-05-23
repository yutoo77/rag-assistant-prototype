import os

from dotenv import load_dotenv
from openai import OpenAI


def print_file_search_results(response):
    print("\n===== 検索・参照情報 =====")

    found = False

    for item in response.output:
        if getattr(item, "type", None) == "file_search_call":
            found = True
            print(f"file_search_call status: {getattr(item, 'status', 'unknown')}")

            results = getattr(item, "results", None)

            if not results:
                print("検索結果の詳細は取得できませんでした。")
                continue

            for i, result in enumerate(results, start=1):
                print(f"\n--- result {i} ---")
                print(f"filename: {getattr(result, 'filename', 'unknown')}")
                print(f"score: {getattr(result, 'score', 'unknown')}")

                content = getattr(result, "content", None)

                if content:
                    text_parts = []
                    for c in content:
                        text = getattr(c, "text", "")
                        if text:
                            text_parts.append(text)

                    preview = "\n".join(text_parts)
                    print(preview[:500])
                else:
                    print("content: なし")

    if not found:
        print("file_search_call が見つかりませんでした。")


def main():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-5.5")
    vector_store_id = os.getenv("VECTOR_STORE_ID")

    if not api_key:
        raise ValueError(".env に OPENAI_API_KEY が設定されていません。")

    if not vector_store_id:
        raise ValueError(".env に VECTOR_STORE_ID が設定されていません。")

    client = OpenAI(api_key=api_key)

    print("RAG質問応答を開始します。")
    print("終了したいときは Ctrl + C を押してください。")

    while True:
        question = input("\n質問を入力してください: ")

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

        print("\n===== 回答 =====")
        print(response.output_text)

        print_file_search_results(response)


if __name__ == "__main__":
    main()