import numpy as np
import pandas as pd
from ..utils.util_P1 import fit_baseline, split_train_test
from ..utils.util_P4 import *
from .parse_data import parse_table


def solve(df_problem=None):
    print("=" * 60)
    print("문제 4. 랭킹 지표")
    print("=" * 60)

    if df_problem is None:
        df_problem = parse_table()

    train, test = split_train_test(df_problem)
    model = fit_baseline(train)

    all_movies = np.sort(df_problem["movie_id"].unique())

    # 1. 추천 모델 점수 준비
    # 1) 인기 추천: 학습 데이터의 평점 수
    pop_series = train.groupby("movie_id").size().reindex(all_movies, fill_value=0)

    # 2) 모델 C 추천: mu + b_u + b_i
    # 특정 유저 u에게 점수 = (mu + b_u) + b_i
    # 아이템 간 순위는 b_i에 의해 결정됨 (동점시 movie_id 오름차순)
    b_i_series = model["b_i"].reindex(all_movies, fill_value=0.0)

    # 유저별 학습 데이터 seen 아이템 딕셔너리
    user_seen = train.groupby("user_id")["movie_id"].apply(set).to_dict()

    # 테스트셋에서 유저별 평점 정보
    # 정답: 테스트 평점이 4 이상인 아이템
    test_ratings_dict = (
        test.groupby("user_id")
        .apply(lambda g: dict(zip(g["movie_id"], g["rating"])))
        .to_dict()
    )

    # 테스트 정답이 1개 이상인 유저 필터링
    eval_users = []
    actual_pos_dict = {}
    for user_id, ratings in test_ratings_dict.items():
        pos_items = {m for m, r in ratings.items() if r >= 4}
        if len(pos_items) >= 1:
            eval_users.append(user_id)
            actual_pos_dict[user_id] = pos_items

    eval_users.sort()
    print(f"\n- 전체 테스트 유저 수: {test['user_id'].nunique()}명")
    print(f"- 테스트 정답(평점 4 이상)이 1개 이상인 유저 수: {len(eval_users)}명")

    # 전역 랭킹 아이템 순서 사전 계산 (recommend 속도 최적화 및 정렬 일관성 보장)
    # 인기 추천 정렬 순서
    pop_sorted_items = all_movies[
        np.lexsort((all_movies, -pop_series.loc[all_movies].to_numpy()))
    ]
    # 모델 C 점수 정렬 순서 (아이템 편향 b_i 기준)
    model_c_sorted_items = all_movies[
        np.lexsort((all_movies, -b_i_series.loc[all_movies].to_numpy()))
    ]

    models_eval = {
        "인기 추천 (학습 평점 수)": pop_sorted_items,
        "모델 C 추천 (예측 평점)": model_c_sorted_items,
    }

    results = []

    for model_name, sorted_items in models_eval.items():
        precisions = []
        recalls = []
        f1_list = []
        reciprocal_ranks = []
        average_precisions = []
        ndcgs = []

        for user_id in eval_users:
            seen = user_seen.get(user_id, set())
            actual = actual_pos_dict[user_id]
            user_ratings = test_ratings_dict[user_id]

            # Top-10 추천 목록 생성: seen 제외 후 상위 10개
            rec_10 = []
            for item in sorted_items:
                if item not in seen:
                    rec_10.append(item)
                    if len(rec_10) == 10:
                        break

            # NDCG용 정답 평점 (테스트 평점 4 이상인 아이템의 rating을 관련도로 사용, 그 외는 0)
            pos_ratings = {m: float(user_ratings[m]) for m in actual}

            # 지표 계산
            p = precision_at_k(rec_10, actual, k=10)
            r = recall_at_k(rec_10, actual, k=10)
            f1 = compute_f1(p, r)
            rr = reciprocal_rank(rec_10, actual)
            ap = average_precision_at_k(rec_10, actual, k=10)
            ndcg_val = ndcg_at_k(rec_10, pos_ratings, k=10)

            precisions.append(p)
            recalls.append(r)
            f1_list.append(f1)
            reciprocal_ranks.append(rr)
            average_precisions.append(ap)
            ndcgs.append(ndcg_val)

        results.append({
            "추천 모델": model_name,
            "Precision@10": np.mean(precisions),
            "Recall@10": np.mean(recalls),
            "F1": np.mean(f1_list),
            "MRR": np.mean(reciprocal_ranks),
            "MAP@10": np.mean(average_precisions),
            "NDCG@10": np.mean(ndcgs),
        })

    results_df = pd.DataFrame(results)
    print("\n[두 추천 모델의 랭킹 평가 지표 비교 표]")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 문제 2 결과와 비교 설명
    print("\n[설명: 문제 2의 평점 예측 결과와 문제 4의 랭킹 평가 결과 비교]")
    explanation = (
        "1. 평점 예측(RMSE/MAE)과 Top-N 랭킹의 괴리:\n"
        "   - 문제 2에서 모델 C는 유저 편향(b_u)과 아이템 편향(b_i)을 모두 반영하여 가장 우수한 RMSE(낮은 오차)를 기록했습니다.\n"
        "   - 그러나 랭킹 관점(문제 4)에서 모델 C의 점수(mu + b_u + b_i)는 단일 유저 내에서 아이템 간 순위가 오직 b_i(아이템 편향)\n"
        "     에 의해서만 결정됩니다. 유저의 세부적인 아이템 선호도(Collaborative Filtering 상호작용)가 빠져 있어 실질적으로\n"
        "     '평균 평점이 높은 영화'를 모든 유저에게 일률적으로 추천하게 됩니다.\n"
        "2. 소수 평가 고평점 아이템 문제 vs 대중성(인기) 효과:\n"
        "   - 모델 C(b_i 기반)는 소수의 유저에게만 5점을 받아 b_i가 극단적으로 높게 추정된 니치(Niche) 영화들을 상위에 추천하기 쉽습니다.\n"
        "   - 반면 인기 추천(Popularity)은 많은 사람들이 실제로 소비하고 검증된 아이템을 추천하므로, 유저가 실제로 접하고\n"
        "     테스트셋에서 높은 평점을 남길 확률(Recall, Precision, NDCG)에서 모델 C를 크게 상회하거나 매우 강력한 베이스라인이 됩니다.\n"
        "3. 교훈:\n"
        "   - 평점 예측 오차(RMSE)의 최소화가 반드시 실제 추천 만족도나 Top-N 랭킹 품질 향상으로 이어지지 않음을 확인 수 있으며,\n"
        "     추천 시스템 목적에 맞는 랭킹 평가 지표(NDCG, Recall 등)의 도입이 필수적입니다."
    )
    print(explanation)
    print("-" * 60)

    return results_df
