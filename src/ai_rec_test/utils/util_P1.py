from typing import Any
import numpy as np
import pandas as pd


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """split 열을 기준으로 train과 test 데이터프레임으로 분할합니다."""
    train = df[df["split"] == "train"].copy()
    test = df[df["split"] == "test"].copy()
    return train, test


def get_data_summary(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """학습 데이터와 테스트 데이터의 평점 수, 유저 수, 아이템 수 요약을 생성합니다."""
    summary_data = {
        "데이터셋": ["Train", "Test"],
        "평점 수": [len(train), len(test)],
        "유저 수": [train["user_id"].nunique(), test["user_id"].nunique()],
        "아이템 수": [train["movie_id"].nunique(), test["movie_id"].nunique()],
    }
    return pd.DataFrame(summary_data)


def fit_baseline(train: pd.DataFrame) -> dict[str, Any]:
    """학습 데이터로 세 베이스라인 모델에 필요한 파라미터(mu, b_i, b_u)를 추정합니다.
    
    - mu: 학습 평점의 전체 평균
    - b_i: 아이템 i의 (r - mu) 평균
    - b_u: 유저 u의 (r - mu - b_i) 평균
    """
    mu = float(train["rating"].mean())

    # b_i: 아이템 i의 (r - mu) 평균
    item_residuals = train["rating"] - mu
    b_i = item_residuals.groupby(train["movie_id"]).mean()

    # b_u: 유저 u의 (r - mu - b_i) 평균
    train_with_bi = train[["user_id", "movie_id", "rating"]].copy()
    train_with_bi["b_i"] = train_with_bi["movie_id"].map(b_i).fillna(0.0)
    user_residuals = train_with_bi["rating"] - mu - train_with_bi["b_i"]
    b_u = user_residuals.groupby(train_with_bi["user_id"]).mean()

    model = {
        "mu": mu,
        "b_i": b_i,  # pd.Series (index: movie_id)
        "b_u": b_u,  # pd.Series (index: user_id)
    }
    return model


def predict(model: dict[str, Any], df: pd.DataFrame, kind: str = "C") -> np.ndarray:
    """학습된 모델로 (user_id, movie_id) 쌍의 평점을 예측합니다.
    
    - 모델 A: r_hat = mu
    - 모델 B: r_hat = mu + b_i
    - 모델 C: r_hat = mu + b_u + b_i
    
    학습 데이터에 없는 유저·아이템의 편향은 0으로 처리합니다.
    """
    kind = kind.upper()
    mu = model["mu"]
    n = len(df)

    if kind == "A":
        return np.full(n, mu, dtype=float)

    b_i_series: pd.Series = model["b_i"]
    item_biases = df["movie_id"].map(b_i_series).fillna(0.0).to_numpy(dtype=float)

    if kind == "B":
        return mu + item_biases

    if kind == "C":
        b_u_series: pd.Series = model["b_u"]
        user_biases = df["user_id"].map(b_u_series).fillna(0.0).to_numpy(dtype=float)
        return mu + user_biases + item_biases

    raise ValueError(f"지원하지 않는 모델 종류입니다: {kind}. ('A', 'B', 'C' 중 선택)")
