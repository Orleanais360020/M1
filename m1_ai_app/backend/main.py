import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from google import genai


# ============================================================
# Gemini API キー
# ============================================================
# 環境変数 GEMINI_API_KEY に設定しておくのが推奨です。
# ローカルで一時的に試すだけなら、下の "YOUR_GEMINI_API_KEY_HERE" を
# 自分のキーに書き換えても動きます（公開リポジトリには絶対に上げないでください）。
# ============================================================

API_KEY = os.getenv("GEMINI_API_KEY") or "YOUR_GEMINI_API_KEY_HERE"

if not API_KEY or API_KEY == "YOUR_GEMINI_API_KEY_HERE":
    raise RuntimeError(
        "Gemini APIキーが設定されていません。\n"
        "環境変数 GEMINI_API_KEY を設定するか、main.py の API_KEY を自分のキーに書き換えてください。"
    )

# Gemini Developer API 用クライアント
client = genai.Client(api_key=API_KEY)

# 利用するモデル
MODEL_ID = "gemini-2.5-flash"


# ============================================================
# FastAPI アプリ設定
# ============================================================

app = FastAPI(
    title="M-1風AI審査バックエンド（Gemini版）",
    description="ユーザーのネタを5人のAI審査員が採点するAPI（フロントも配信）",
    version="1.0.0",
)

# 同一オリジンでフロントも配信するので、CORS はほぼ不要ですが残しておきます
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 必要に応じて絞り込んでください
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Pydantic モデル定義
# ============================================================

class EvaluateRequest(BaseModel):
    text: str


class JudgeResult(BaseModel):
    judge_name: str
    score: Optional[int]
    comment: str


class EvaluateResponse(BaseModel):
    average_score: Optional[float]
    valid_count: int
    results: List[JudgeResult]


# ============================================================
# 審査員設定
# ============================================================

JUDGES = [
    {
        "name": "辛口ベテラン漫才師",
        "style": "ボケとツッコミの構成、伏線回収、オチのキレを厳しく見る。テンポ感にもこだわる。",
    },
    {
        "name": "理論派ツッコミ芸人",
        "style": "論理的なボケの積み上げ、ツッコミの位置、言葉選びを分析的に評価する。",
    },
    {
        "name": "感性派ピン芸人",
        "style": "感情の起伏・キャラクター性・勢い・インパクトを重視して評価する。",
    },
    {
        "name": "若手構成作家",
        "style": "構成・設定の新しさ、展開のメリハリ、客層へのハマり方などを総合的に見る。",
    },
    {
        "name": "お笑いマニア審査員",
        "style": "過去の名作ネタとの比較、新規性、細かい工夫などをマニア目線で評価する。",
    },
]


# ============================================================
# プロンプト作成
# ============================================================

def build_prompt(judge: dict, user_text: str) -> str:
    return f"""
あなたは漫才コンテストの審査員です。
キャラクター: {judge["style"]}

これから、お笑いコンテストのネタ（日本語の文章）が与えられます。
このネタの「面白さ」を、次の条件で評価してください。

- 0〜100点の整数のみで採点する（高いほど面白い）
- {judge["name"]} というキャラクターとしてコメントを書く
- 日本語で短くコメントする（1〜3文程度）

[ネタ]
{user_text}

出力フォーマットは必ず次の1行だけにしてください：
score=点数; comment=コメント

例:
score=85; comment=テンポが良くてオチもわかりやすく、とても笑えました。
""".strip()


# ============================================================
# Gemini API 呼び出し
# ============================================================

def call_model(prompt: str) -> str:
    """
    Gemini API にプロンプトを投げて、テキストを1本返します。
    """
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
    )
    # google-genai のレスポンスは .text にまとめてテキストが入ります
    return response.text or ""


# ============================================================
# スコアとコメントのパース
# ============================================================

def parse_score_and_comment(output_text: str) -> Tuple[Optional[int], str]:
    """
    モデル出力から score と comment を抽出します。
    想定フォーマット: score=85; comment=コメント
    """
    score_match = re.search(r"score\s*=\s*(\d{1,3})", output_text, re.IGNORECASE)
    score: Optional[int] = None
    if score_match:
        value = int(score_match.group(1))
        value = max(0, min(100, value))
        score = value

    comment_match = re.search(
        r"comment\s*=\s*(.*)$", output_text, re.IGNORECASE | re.DOTALL
    )
    if comment_match:
        comment = comment_match.group(1).strip()
    else:
        comment = output_text.strip()

    return score, comment


# ============================================================
# API エンドポイント
# ============================================================

@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.post("/api/evaluate", response_model=EvaluateResponse)
def evaluate(request: EvaluateRequest) -> EvaluateResponse:
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text が空です。")

    results: List[JudgeResult] = []
    total_score = 0
    valid_count = 0

    for judge in JUDGES:
        prompt = build_prompt(judge, text)
        try:
            output = call_model(prompt)
            score, comment = parse_score_and_comment(output)
        except Exception as e:
            score = None
            comment = f"APIエラー: {e}"

        if score is not None:
            total_score += score
            valid_count += 1

        results.append(
            JudgeResult(
                judge_name=judge["name"],
                score=score,
                comment=comment,
            )
        )

    if valid_count > 0:
        average_score: Optional[float] = round(total_score / valid_count, 1)
    else:
        average_score = None

    return EvaluateResponse(
        average_score=average_score,
        valid_count=valid_count,
        results=results,
    )


# ============================================================
# フロントエンド配信（StaticFiles）
# ============================================================

# backend/ から見て ../frontend/ を探す
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

if not FRONTEND_DIR.is_dir():
    raise RuntimeError(f"frontend ディレクトリが見つかりません: {FRONTEND_DIR}")

# "/" でフロントエンドを配信（index.html をトップに）
app.mount(
    "/",
    StaticFiles(directory=str(FRONTEND_DIR), html=True),
    name="static",
)


# ローカル起動用
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
