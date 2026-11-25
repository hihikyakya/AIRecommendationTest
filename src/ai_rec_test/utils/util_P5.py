from typing import Any, Sequence
import numpy as np
import pandas as pd


def select_target_item(user_test_ratings: pd.DataFrame) -> int:
    """테스트 정답(평점 >= 4) 중 평점이 가장 높은 1개 아이템을 선택합니다.
    
    동점 규칙:
    1. 평점 가장 높음 (내림차순)
    2. timestamp 가장 늦음 (내림차순)
    3. movie_id 가장 작음 (오름차순)
    """
    pos_items = user_test_ratings[user_test_ratings["rating"] >= 4].copy()
    if pos_items.empty:
        raise ValueError("테스트 평점 4 이상인 정답 아이템이 없습니다.")

    pos_items = pos_items.sort_values(
        by=["rating", "timestamp", "movie_id"],
        ascending=[False, False, True],
    )
    return int(pos_items.iloc[0]["movie_id"])


def sample_candidates(
    user_id: int,
    user_test_ratings: pd.DataFrame,
    all_evaluated_items: set[int],
    all_movies: Sequence[int],
    rng: np.random.Generator,
) -> tuple[int, list[int]]:
    """유저의 최우선 테스트 정답 1개와 학습·테스트에서 평가하지 않은 아이템 99개로 구성된 100개 후보를 생성합니다.
    
    반환: (target_item, candidate_100_items)
    """
    # 1. 정답 아이템 1개 선택
    target_item = select_target_item(user_test_ratings)

    # 2. 유저가 학습·테스트에서 평가하지 않은 후보 아이템 풀 구성
    unrated_pool = np.array(
        [m for m in all_movies if m not in all_evaluated_items],
        dtype=int,
    )

    if len(unrated_pool) < 99:
        raise ValueError(f"유저 {user_id}의 미평가 아이템 수가 99개 미만입니다: {len(unrated_pool)}")

    # 3. rng로 99개 비복원 추출
    sampled_negatives = rng.choice(unrated_pool, size=99, replace=False)

    candidates = [target_item] + list(sampled_negatives)
    return target_item, candidates


def rank_candidates(
    candidates: Sequence[int],
    scores_dict: dict[int, float],
) -> list[int]:
    """100개 후보 아이템을 점수 내림차순, movie_id 오름차순으로 정렬합니다."""
    cand_scores = [(item, scores_dict.get(item, 0.0)) for item in candidates]
    cand_scores.sort(key=lambda x: (-x[1], x[0]))
    return [item for item, _ in cand_scores]
