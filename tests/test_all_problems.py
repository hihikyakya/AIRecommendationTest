import unittest
import numpy as np
import pandas as pd

from ai_rec_test.services.parse_data import parse_table
from ai_rec_test.utils.util_P1 import (
    fit_baseline,
    predict,
    split_train_test,
    get_data_summary,
)
from ai_rec_test.utils.util_P2 import (
    rmse,
    mae,
    mse,
    calculate_improvement_rate,
)
from ai_rec_test.utils.util_P3 import (
    micro_rmse,
    macro_rmse,
    split_users_into_quartiles,
)
from ai_rec_test.utils.util_P4 import (
    recommend,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    average_precision_at_k,
    ndcg_at_k,
    compute_f1,
)
from ai_rec_test.utils.util_P5 import (
    select_target_item,
    sample_candidates,
    rank_candidates,
)


class TestRecommendationPipeline(unittest.TestCase):
    def setUp(self):
        # 소규모 모의 데이터셋 생성
        self.mock_df = pd.DataFrame({
            "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 2, 3],
            "movie_id": [10, 20, 30, 10, 20, 40, 20, 30, 40, 50, 50, 50],
            "rating": [5, 4, 3, 4, 3, 2, 5, 4, 3, 4, 5, 2],
            "timestamp": [100, 101, 102, 100, 101, 102, 100, 101, 102, 200, 200, 200],
            "split": [
                "train", "train", "train",
                "train", "train", "train",
                "train", "train", "train",
                "test", "test", "test",
            ],
        })

    def test_util_p1_baseline(self):
        train, test = split_train_test(self.mock_df)
        self.assertEqual(len(train), 9)
        self.assertEqual(len(test), 3)

        summary = get_data_summary(train, test)
        self.assertEqual(summary.loc[summary["데이터셋"] == "Train", "유저 수"].iloc[0], 3)

        model = fit_baseline(train)
        self.assertIn("mu", model)
        self.assertIn("b_i", model)
        self.assertIn("b_u", model)

        # 수동 계산 검증
        # train ratings: [5, 4, 3, 4, 3, 2, 5, 4, 3] -> sum=33, len=9, mu=33/9 = 3.6666...
        expected_mu = 33.0 / 9.0
        self.assertAlmostEqual(model["mu"], expected_mu, places=5)

        # predict 검증
        pred_a = predict(model, test, kind="A")
        np.testing.assert_allclose(pred_a, expected_mu)

        pred_b = predict(model, test, kind="B")
        self.assertEqual(len(pred_b), len(test))

        pred_c = predict(model, test, kind="C")
        self.assertEqual(len(pred_c), len(test))

        # Train에 없는 미관측 영화(예: 999)에 대해 편향이 0인지 검증
        unseen_test = pd.DataFrame({"user_id": [1], "movie_id": [999]})
        pred_unseen_b = predict(model, unseen_test, kind="B")
        self.assertAlmostEqual(pred_unseen_b[0], expected_mu, places=5)

    def test_util_p2_metrics(self):
        y = np.array([3.0, 4.0, 5.0])
        y_hat = np.array([2.0, 4.0, 7.0])  # diffs: [1.0, 0.0, -2.0]
        # squared diffs: [1.0, 0.0, 4.0] -> mean=5/3 -> rmse=sqrt(5/3)
        # abs diffs: [1.0, 0.0, 2.0] -> mean=3/3 = 1.0

        self.assertAlmostEqual(mse(y, y_hat), 5.0 / 3.0, places=5)
        self.assertAlmostEqual(rmse(y, y_hat), np.sqrt(5.0 / 3.0), places=5)
        self.assertAlmostEqual(mae(y, y_hat), 1.0, places=5)

        imp = calculate_improvement_rate(2.0, 1.5)
        self.assertAlmostEqual(imp, 25.0, places=5)

    def test_util_p3_micro_macro(self):
        eval_df = pd.DataFrame({
            "user_id": [1, 1, 2, 2],
            "rating": [4, 5, 3, 3],
            "pred": [4, 4, 3, 5],
        })
        # diffs: [0, 1, 0, -2]
        # 전체 micro RMSE: sqrt((0 + 1 + 0 + 4) / 4) = sqrt(5/4) = sqrt(1.25)
        self.assertAlmostEqual(micro_rmse(eval_df), np.sqrt(1.25), places=5)

        # user 1 diffs: [0, 1] -> RMSE_1 = sqrt(1/2) = 0.7071
        # user 2 diffs: [0, -2] -> RMSE_2 = sqrt(4/2) = sqrt(2) = 1.4142
        # macro RMSE: (sqrt(0.5) + sqrt(2)) / 2
        expected_macro = (np.sqrt(0.5) + np.sqrt(2.0)) / 2.0
        self.assertAlmostEqual(macro_rmse(eval_df), expected_macro, places=5)

    def test_util_p4_ranking_metrics(self):
        rec = [10, 20, 30, 40, 50]
        actual = {20, 50}  # 2개 hit (순위 2, 순위 5)

        # Precision@5 = 2 / 5 = 0.4
        self.assertAlmostEqual(precision_at_k(rec, actual, k=5), 0.4)
        # Recall@5 = 2 / 2 = 1.0
        self.assertAlmostEqual(recall_at_k(rec, actual, k=5), 1.0)
        # RR = 1 / 2 = 0.5 (첫 정답이 순위 2에 있음)
        self.assertAlmostEqual(reciprocal_rank(rec, actual), 0.5)

        # AP@5:
        # rank 1 (10): miss
        # rank 2 (20): hit, P@2 = 1/2
        # rank 3 (30): miss
        # rank 4 (40): miss
        # rank 5 (50): hit, P@5 = 2/5
        # AP = (1/2 + 2/5) / min(2, 5) = (0.5 + 0.4) / 2 = 0.45
        self.assertAlmostEqual(average_precision_at_k(rec, actual, k=5), 0.45)

        # NDCG@5 (관련도 1.0인 경우)
        # DCG@5 = 1/log2(2+1) + 1/log2(5+1) = 1/log2(3) + 1/log2(6)
        # IDCG@5 = 1/log2(1+1) + 1/log2(2+1) = 1 + 1/log2(3)
        dcg_expected = 1.0 / np.log2(3) + 1.0 / np.log2(6)
        idcg_expected = 1.0 / np.log2(2) + 1.0 / np.log2(3)
        self.assertAlmostEqual(ndcg_at_k(rec, actual, k=5), dcg_expected / idcg_expected, places=5)

        # recommend 함수 검증
        scores = {10: 1.5, 20: 3.0, 30: 2.5, 40: 2.5}
        seen = {20}
        # unseen: 10(1.5), 30(2.5), 40(2.5)
        # 정렬: 30(2.5), 40(2.5) -> 점수 동점시 movie_id 오름차순: 30, 40, 그 다음 10
        top2 = recommend(scores, seen, k=2)
        self.assertEqual(top2, [30, 40])

    def test_util_p5_sampling(self):
        # target item 선택 규칙 검증
        # 1. rating 가장 높음 -> 2. timestamp 가장 늦음 -> 3. movie_id 가장 작음
        user_test = pd.DataFrame({
            "movie_id": [101, 102, 103, 104],
            "rating": [4, 5, 5, 3],
            "timestamp": [10, 20, 20, 30],
        })
        # 5점: 102(ts 20), 103(ts 20) -> rating 5 동점, timestamp 20 동점 -> movie_id 작은 102 선택!
        target = select_target_item(user_test)
        self.assertEqual(target, 102)

        # 후보 샘플링 시드 재현성 검증
        all_movies = list(range(1, 1000))
        seen = {101, 102, 103, 104}
        rng1 = np.random.default_rng(42)
        t1, c1 = sample_candidates(1, user_test, seen, all_movies, rng1)

        rng2 = np.random.default_rng(42)
        t2, c2 = sample_candidates(1, user_test, seen, all_movies, rng2)

        self.assertEqual(t1, t2)
        self.assertEqual(c1, c2)
        self.assertEqual(len(c1), 100)
        self.assertEqual(c1[0], target)
        self.assertEqual(len(set(c1)), 100)  # 중복 없음

    def test_actual_dataset_execution(self):
        # 실제 데이터셋 로드 검증
        df = parse_table()
        self.assertEqual(len(df), 1000209)
        self.assertEqual(df["split"].value_counts()["train"], 939809)
        self.assertEqual(df["split"].value_counts()["test"], 60400)


if __name__ == "__main__":
    unittest.main()
