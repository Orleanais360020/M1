import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
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
# 推奨: Render などの環境変数で GEMINI_API_KEY を設定しておく。
# ローカルで試すだけなら、下の "YOUR_GEMINI_API_KEY_HERE" を
# 自分のキーに書き換えても動きます（公開リポジトリには上げないこと）。
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

# タイムゾーン（ランキングの「本日」判定用・JST）
JST = timezone(timedelta(hours=9))


# ============================================================
# FastAPI アプリ設定
# ============================================================

app = FastAPI(
    title="M-1風AI審査アプリ",
    description="ユーザーのネタを5人のAI審査員が採点し、ランキングも集計するAPI",
    version="1.0.0",
)

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
    entry_name: str  # コンビ名・芸名など


class JudgeResult(BaseModel):
    judge_name: str
    score: Optional[int]
    comment: str


class EvaluateResponse(BaseModel):
    average_score: Optional[float]
    valid_count: int
    results: List[JudgeResult]


class LeaderboardEntry(BaseModel):
    rank: int
    entry_name: str
    average_score: float


class LeaderboardResponse(BaseModel):
    daily_top: List[LeaderboardEntry]
    all_time_top: List[LeaderboardEntry]


# ============================================================
# 内部用 評価記録モデル（メモリ内にのみ保持）
# ============================================================

@dataclass
class EvaluationRecord:
    id: int
    entry_name: str
    average_score: float
    created_at: datetime  # JST


EVALUATIONS: List[EvaluationRecord] = []
NEXT_EVAL_ID: int = 1


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
# プロンプト作成（審査員ごと）
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
# ランキング関連関数
# ============================================================

def add_evaluation(entry_name: str, average_score: float) -> None:
    """
    メモリ上の評価リストに1件追加します。
    """
    global NEXT_EVAL_ID
    now = datetime.now(JST)
    record = EvaluationRecord(
        id=NEXT_EVAL_ID,
        entry_name=entry_name,
        average_score=average_score,
        created_at=now,
    )
    NEXT_EVAL_ID += 1
    EVALUATIONS.append(record)


def compute_leaderboard(now: datetime) -> LeaderboardResponse:
    """
    本日分TOP3と通算TOP3を計算して返します。
    """
    if not EVALUATIONS:
        return LeaderboardResponse(daily_top=[], all_time_top=[])

    today_date = now.astimezone(JST).date()

    daily_records = [
        r for r in EVALUATIONS
        if r.created_at.astimezone(JST).date() == today_date
    ]

    daily_sorted = sorted(
        daily_records,
        key=lambda r: (-r.average_score, r.created_at),
    )
    all_sorted = sorted(
        EVALUATIONS,
        key=lambda r: (-r.average_score, r.created_at),
    )

    def to_entries(records: List[EvaluationRecord]) -> List[LeaderboardEntry]:
        entries: List[LeaderboardEntry] = []
        for idx, r in enumerate(records[:3]):
            entries.append(
                LeaderboardEntry(
                    rank=idx + 1,
                    entry_name=r.entry_name,
                    average_score=round(r.average_score, 1),
                )
            )
        return entries

    return LeaderboardResponse(
        daily_top=to_entries(daily_sorted),
        all_time_top=to_entries(all_sorted),
    )


# ============================================================
# API エンドポイント
# ============================================================

@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.get("/api/leaderboard", response_model=LeaderboardResponse)
def get_leaderboard() -> LeaderboardResponse:
    now = datetime.now(JST)
    return compute_leaderboard(now)


@app.post("/api/evaluate", response_model=EvaluateResponse)
def evaluate(request: EvaluateRequest) -> EvaluateResponse:
    text = request.text.strip()
    entry_name = request.entry_name.strip()

    if not text:
        raise HTTPException(status_code=400, detail="text が空です。")
    if not entry_name:
        raise HTTPException(status_code=400, detail="entry_name が空です。")

    # 名前は長すぎると表示が崩れるのでサーバ側でも一応制限しておく
    entry_name = entry_name[:100]

    results: List[JudgeResult] = []
    total_score = 0
    valid_count = 0

    # 5人の審査員でそれぞれ採点
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
        add_evaluation(entry_name=entry_name, average_score=average_score)
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

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

if not FRONTEND_DIR.is_dir():
    raise RuntimeError(f"frontend ディレクトリが見つかりません: {FRONTEND_DIR}")

app.mount(
    "/",
    StaticFiles(directory=str(FRONTEND_DIR), html=True),
    name="static",
)


# ローカル起動用
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
