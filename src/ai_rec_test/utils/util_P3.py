from pathlib import Path
import numpy as np
import pandas as pd
from ..utils.util_P2 import rmse


# (u,i)공간에서 계산
def micro_rmse(df: pd.DataFrame, y_col: str = "rating", pred_col: str = "pred") -> float:
    """전체 (u, i) 테스트 평점 기준의 마이크로 RMSE를 계산합니다."""
    return rmse(df[y_col].to_numpy(), df[pred_col].to_numpy())


# u 공간에서 계산
def macro_rmse(
    df: pd.DataFrame,
    user_col: str = "user_id",
    y_col: str = "rating",
    pred_col: str = "pred",
) -> float:
    """유저별로 테스트 평점의 RMSE를 각각 계산한 후 그 평균(매크로 RMSE)을 반환합니다."""
    def _user_rmse(group: pd.DataFrame) -> float:
        return rmse(group[y_col].to_numpy(), group[pred_col].to_numpy())

    user_rmses = df.groupby(user_col, observed=True).apply(_user_rmse)
    return float(user_rmses.mean())


def split_users_into_quartiles(train: pd.DataFrame) -> pd.DataFrame:
    """유저를 학습 평점 수 기준 4개 분위(Q1~Q4) 그룹으로 분할합니다.
    
    반환: user_id와 quartile 그룹 정보를 포함하는 DataFrame
    """
    user_counts = (
        train.groupby("user_id", observed=True)
        .size()
        .reset_index(name="train_rating_count")
    )

    labels = ["Q1 (하위 25%)", "Q2 (25~50%)", "Q3 (50~75%)", "Q4 (상위 25%)"]
    user_counts["quartile"] = pd.qcut(
        user_counts["train_rating_count"],
        q=4,
        labels=labels,
        duplicates="drop",
    )
    return user_counts


def render_ascii_bar_chart(
    data: dict[str, float],
    title: str = "그룹별 RMSE 막대그래프",
    max_bar_width: int = 40,
) -> str:
    """콘솔에 출력할 수 있는 텍스트 막대그래프를 생성합니다."""
    lines = [f"\n=== {title} ==="]
    if not data:
        return "\n".join(lines)

    max_val = max(data.values())
    for label, val in data.items():
        bar_len = int((val / max_val) * max_bar_width) if max_val > 0 else 0
        bar = "#" * bar_len
        lines.append(f"{label:<16} | {bar:<{max_bar_width}} {val:.4f}")
    lines.append("=" * (max_bar_width + 26))
    return "\n".join(lines)


def plot_group_rmse_figure(
    data: dict[str, float],
    save_path: str | Path | None = None,
) -> str | None:
    """matplotlib을 사용하여 그룹별 RMSE 막대그래프 이미지를 저장합니다."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive 백엔드 사용
        import matplotlib.pyplot as plt
        import platform

        # 한글 폰트 설정 (Windows: Malgun Gothic, Mac: AppleGothic, 그 외 Nanum/기본)
        system_name = platform.system()
        if system_name == "Windows":
            plt.rcParams["font.family"] = "Malgun Gothic"
        elif system_name == "Darwin":
            plt.rcParams["font.family"] = "AppleGothic"
        plt.rcParams["axes.unicode_minus"] = False

        if save_path is None:
            # 기본 저장 경로
            output_dir = Path("outputs")
            output_dir.mkdir(parents=True, exist_ok=True)
            save_path = output_dir / "problem3_group_rmse.png"
        else:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

        labels = list(data.keys())
        values = list(data.values())

        plt.figure(figsize=(8, 5))
        bars = plt.bar(labels, values, color=["#4A90E2", "#50E3C2", "#F5A623", "#7ED321"], width=0.55)
        plt.title("학습 평점 수 분위 그룹별 모델 C 테스트 RMSE", fontsize=14, fontweight="bold", pad=15)
        plt.xlabel("학습 평점 수 4개 분위 그룹 (Quartiles)", fontsize=11, labelpad=10)
        plt.ylabel("Test RMSE", fontsize=11)
        plt.ylim(0, max(values) * 1.18)
        plt.grid(axis="y", linestyle="--", alpha=0.5)

        # 막대 위에 수치 텍스트 표시
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.015,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()
        return str(save_path)
    except Exception as e:
        # matplotlib이 실패하더라도 콘솔 파이프라인 중단 방지
        return None
