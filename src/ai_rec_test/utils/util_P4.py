from typing import Any, Iterable, Sequence
import numpy as np
import pandas as pd


def recommend(
    scores: dict[int, float] | pd.Series | np.ndarray,
    seen: set[int] | Iterable[int],
    k: int = 10,
    all_item_ids: Sequence[int] | None = None,
) -> list[int]:
    """유저가 학습 데이터에서 평가하지 않은 전체 아이템을 점수 순으로 정렬하여 Top-k 추천 목록을 생성합니다.
    
    점수가 같을 경우 movie_id 오름차순으로 안정적으로 정렬합니다.
    """
    seen_set = set(seen)

    if isinstance(scores, pd.Series):
        # pandas Series인 경우 (index가 movie_id)
        # seen 제외
        cand_series = scores[~scores.index.isin(seen_set)]
        # 점수 내림차순, movie_id 오름차순 정렬
        sorted_indices = cand_series.index[
            np.lexsort((cand_series.index.to_numpy(), -cand_series.to_numpy()))
        ]
        return list(sorted_indices[:k])

    elif isinstance(scores, dict):
        candidates = [
            (item_id, score) for item_id, score in scores.items() if item_id not in seen_set
        ]
        # score 내림차순, item_id 오름차순 정렬
        candidates.sort(key=lambda x: (-x[1], x[0]))
        return [item_id for item_id, _ in candidates[:k]]

    elif isinstance(scores, (np.ndarray, list)):
        if all_item_ids is None:
            raise ValueError("scores가 배열 형태일 때는 all_item_ids를 제공해야 합니다.")
        candidates = [
            (item_id, float(scores[idx]))
            for idx, item_id in enumerate(all_item_ids)
            if item_id not in seen_set
        ]
        candidates.sort(key=lambda x: (-x[1], x[0]))
        return [item_id for item_id, _ in candidates[:k]]

    raise TypeError(f"지원하지 않는 scores 타입입니다: {type(scores)}")


def precision_at_k(recommended: Sequence[int], actual: set[int] | Sequence[int], k: int = 10) -> float:
    """Top-k 추천 목록의 정밀도 (Precision@k)를 계산합니다."""
    if k <= 0:
        return 0.0
    rec_k = recommended[:k]
    actual_set = set(actual)
    hits = len(set(rec_k) & actual_set)
    return float(hits / k)


def recall_at_k(recommended: Sequence[int], actual: set[int] | Sequence[int], k: int = 10) -> float:
    """Top-k 추천 목록의 재현율 (Recall@k)를 계산합니다."""
    actual_set = set(actual)
    if not actual_set:
        return 0.0
    rec_k = recommended[:k]
    hits = len(set(rec_k) & actual_set)
    return float(hits / len(actual_set))


def reciprocal_rank(recommended: Sequence[int], actual: set[int] | Sequence[int]) -> float:
    """추천 목록에서 첫 번째 정답 아이템이 등장하는 위치의 역수 (RR)를 계산합니다."""
    actual_set = set(actual)
    for rank, item in enumerate(recommended, start=1):
        if item in actual_set:
            return float(1.0 / rank)
    return 0.0


def mean_reciprocal_rank(recommendations: Sequence[Sequence[int]], actuals: Sequence[set[int] | Sequence[int]]) -> float:
    """전체 유저에 대한 Mean Reciprocal Rank (MRR)를 계산합니다."""
    if not recommendations:
        return 0.0
    rr_list = [reciprocal_rank(rec, act) for rec, act in zip(recommendations, actuals)]
    return float(np.mean(rr_list))


def average_precision_at_k(
    recommended: Sequence[int],
    actual: set[int] | Sequence[int],
    k: int = 10,
) -> float:
    """Top-k 추천 목록에 대한 Average Precision (AP@k)를 계산합니다."""
    actual_set = set(actual)
    if not actual_set or k <= 0:
        return 0.0

    rec_k = recommended[:k]
    hits = 0
    score_sum = 0.0

    for rank, item in enumerate(rec_k, start=1):
        if item in actual_set:
            hits += 1
            score_sum += hits / rank

    normalizer = min(len(actual_set), k)
    return float(score_sum / normalizer) if normalizer > 0 else 0.0


def ndcg_at_k(
    recommended: Sequence[int],
    actual_ratings: dict[int, float] | set[int] | Sequence[int],
    k: int = 10,
) -> float:
    """Top-k 추천 목록에 대한 NDCG@k를 계산합니다.
    
    관련도(relevance):
    - actual_ratings가 dict인 경우, 테스트 평점 4 이상인 아이템의 rating을 관련도로 사용, 그 외는 0
    - actual_ratings가 set/list인 경우, 해당 집합에 속하면 관련도 1.0, 그 외는 0
    """
    if k <= 0:
        return 0.0

    rec_k = recommended[:k]

    # 관련도 딕셔너리 구성
    if isinstance(actual_ratings, dict):
        rel_dict = {
            item: float(r) for item, r in actual_ratings.items() if float(r) > 0.0
        }
    else:
        rel_dict = {item: 1.0 for item in actual_ratings}

    if not rel_dict:
        return 0.0

    # DCG@k 계산
    dcg = 0.0
    for rank, item in enumerate(rec_k, start=1):
        rel = rel_dict.get(item, 0.0)
        if rel > 0:
            dcg += rel / np.log2(rank + 1)

    # IDCG@k 계산: 가장 높은 관련도를 가진 아이템들을 상위 순위에 배치했을 때의 DCG
    ideal_rels = sorted(rel_dict.values(), reverse=True)[:k]
    idcg = sum(rel / np.log2(rank + 1) for rank, rel in enumerate(ideal_rels, start=1))

    return float(dcg / idcg) if idcg > 0 else 0.0


def compute_f1(precision_val: float, recall_val: float) -> float:
    """정밀도와 재현율로부터 F1 점수를 계산합니다."""
    if precision_val + recall_val == 0:
        return 0.0
    return float(2.0 * precision_val * recall_val / (precision_val + recall_val))


# 기존 및 과제 명세 함수명 호환용 별칭
precision = precision_at_k
recall = recall_at_k
average_precision = average_precision_at_k
ndcg = ndcg_at_k
