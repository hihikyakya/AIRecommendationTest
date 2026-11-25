import numpy as np
import pandas as pd
from ..utils.util_P1 import fit_baseline, split_train_test
from ..utils.util_P4 import ndcg_at_k, recall_at_k
from ..utils.util_P5 import *
from .parse_data import parse_table


def solve(df_problem=None):
    print("=" * 60)
    print("문제 5. 표본 채점 (Sampled Metrics)")
    print("=" * 60)

    if df_problem is None:
        df_problem = parse_table()

    train, test = split_train_test(df_problem)
    model = fit_baseline(train)

    all_movies = np.sort(df_problem["movie_id"].unique())

    # 1. 대상 유저: 문제 4의 유저 (테스트 정답 평점 >= 4가 1개 이상인 유저)
    pos_test = test[test["rating"] >= 4]
    eval_user_ids = sorted(pos_test["user_id"].unique())
    print(f"\n- 평가 대상 유저 수 (문제 4와 동일): {len(eval_user_ids)}명")

    # 2. 유저별 전체 평가 아이템 (학습 + 테스트)
    all_seen_by_user = df_problem.groupby("user_id")["movie_id"].apply(set).to_dict()

    # 테스트 데이터 유저별 그룹화
    test_grouped = {uid: group for uid, group in test.groupby("user_id")}

    # 점수 딕셔너리 준비
    # 1) 인기 추천 점수: train 내 평점 수
    pop_scores = train.groupby("movie_id").size().to_dict()

    # 2) 모델 C 추천 점수: 아이템 편향 b_i
    # (유저 내에서는 mu + b_u가 동일하므로 b_i 기준 정렬과 완벽히 일치)
    b_i_scores = model["b_i"].to_dict()

    # 3. rng 초기화: 유저 ID 오름차순으로 추출
    rng = np.random.default_rng(42)

    # 두 모델별 지표 수집
    pop_recalls = []
    pop_ndcgs = []

    model_c_recalls = []
    model_c_ndcgs = []

    for user_id in eval_user_ids:
        user_test_df = test_grouped[user_id]
        evaluated_items = all_seen_by_user.get(user_id, set())

        # 후보 100개 구성 (정답 1개 + 미평가 99개)
        target_item, candidates_100 = sample_candidates(
            user_id=user_id,
            user_test_ratings=user_test_df,
            all_evaluated_items=evaluated_items,
            all_movies=all_movies,
            rng=rng,
        )

        target_actual = [target_item]
        # NDCG 계산용 관련도 1
        target_actual_dict = {target_item: 1.0}

        # 1) 인기 추천 모델 정렬
        pop_ranked = rank_candidates(candidates_100, pop_scores)
        pop_rec = recall_at_k(pop_ranked, target_actual, k=10)
        pop_nd = ndcg_at_k(pop_ranked, target_actual_dict, k=10)
        pop_recalls.append(pop_rec)
        pop_ndcgs.append(pop_nd)

        # 2) 모델 C 추천 모델 정렬
        model_c_ranked = rank_candidates(candidates_100, b_i_scores)
        c_rec = recall_at_k(model_c_ranked, target_actual, k=10)
        c_nd = ndcg_at_k(model_c_ranked, target_actual_dict, k=10)
        model_c_recalls.append(c_rec)
        model_c_ndcgs.append(c_nd)

    sampled_results = pd.DataFrame([
        {
            "추천 모델": "인기 추천 (학습 평점 수)",
            "Sampled Recall@10": np.mean(pop_recalls),
            "Sampled NDCG@10": np.mean(pop_ndcgs),
        },
        {
            "추천 모델": "모델 C 추천 (예측 평점)",
            "Sampled Recall@10": np.mean(model_c_recalls),
            "Sampled NDCG@10": np.mean(model_c_ndcgs),
        },
    ])

    print("\n[표본 채점 (100개 후보) 평가 결과 표]")
    print(sampled_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 문제 4의 전체 랭킹 결과와 비교 설명
    print("\n[설명: 표본 채점(Sampled Metrics)과 전체 랭킹(Full Ranking) 결과 비교]")
    explanation = (
        "1. 수치적 급상승 원인 (난이도 하락):\n"
        "   - 전체 랭킹(문제 4)에서는 3,700여 개의 전체 비평가 아이템 중에서 상위 10개를 선별해야 하므로\n"
        "     정답이 Top-10에 포함될 확률이 상대적으로 낮습니다.\n"
        "   - 반면 표본 채점(문제 5)에서는 단 1개의 정답 아이템과 무작위로 추출된 99개의 비관측 아이템(대부분 비인기 아이템)\n"
        "     총 100개 사이에서만 순위를 다투므로, 정답이 Top-10(상위 10%)에 안착할 확률(Recall, NDCG)이 극적으로 높아집니다.\n"
        "2. 표본 채점의 한계 및 주의점 (Sampled Metric Bias):\n"
        "   - 논문(Krichene & Rendle, 2020)에서 지적하듯, 무작위 음성 표본(Uniform Negative Sampling)은\n"
        "     실제 서비스 환경의 치열한 후보 경쟁을 과소평가합니다.\n"
        "   - 특히 인기 추천이나 대중적 편향이 강한 모델일수록, 99개의 무작위 아이템보다 점수가 월등히 높아\n"
        "     표본 채점에서 지나치게 유리하게 평가되는 왜곡(Popularity Bias 증폭)이 발생할 수 있습니다.\n"
        "3. 결론:\n"
        "   - 표본 채점은 대규모 서비스에서 계산 비용을 줄이기 위한 효율적인 근사 방법이지만,\n"
        "     모델 간의 진정한 랭킹 성능 순위를 왜곡할 수 있으므로 최종 평가 시에는 전체 랭킹 평가를 병행해야 합니다."
    )
    print(explanation)
    print("-" * 60)

    return sampled_results
