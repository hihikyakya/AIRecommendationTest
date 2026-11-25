import pandas as pd
from ..utils.util_P1 import fit_baseline, predict, split_train_test
from ..utils.util_P2 import *
from .parse_data import parse_table


def solve(df_problem=None):
    print("=" * 60)
    print("문제 2. RMSE 와 MAE")
    print("=" * 60)

    if df_problem is None:
        df_problem = parse_table()

    train, test = split_train_test(df_problem)
    model = fit_baseline(train)

    y_test = test["rating"].to_numpy()

    # 세 모델의 예측값 계산
    pred_A = predict(model, test, kind="A")
    pred_B = predict(model, test, kind="B")
    pred_C = predict(model, test, kind="C")

    # RMSE 및 MAE 계산
    rmse_A = rmse(y_test, pred_A)
    mae_A = mae(y_test, pred_A)

    rmse_B = rmse(y_test, pred_B)
    mae_B = mae(y_test, pred_B)

    rmse_C = rmse(y_test, pred_C)
    mae_C = mae(y_test, pred_C)

    # 표로 출력
    results_df = pd.DataFrame({
        "모델": ["모델 A (mu)", "모델 B (mu + b_i)", "모델 C (mu + b_u + b_i)"],
        "RMSE": [rmse_A, rmse_B, rmse_C],
        "MAE": [mae_A, mae_B, mae_C],
    })
    print("\n[세 모델의 테스트 성능 비교 표]")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 모델 B 대비 모델 C의 RMSE 개선율 계산
    improvement = calculate_improvement_rate(rmse_B, rmse_C)
    print(f"\n[모델 B 대비 모델 C의 RMSE 개선율]: {improvement:.2f}%")

    # 두 지표의 차이 설명
    print("\n[설명: RMSE와 MAE 두 지표의 차이와 의미]")
    explanation = (
        "1. 수식 및 오차 가중치 차이:\n"
        "   - MAE는 예측 오차의 절댓값 평균(|y - y_hat|)으로, 모든 오차에 대해 선형적인 페널티를 부여합니다.\n"
        "   - RMSE는 오차의 제곱 평균의 제곱근(sqrt(mean((y - y_hat)^2)))으로, 오차가 클수록 제곱에 비례하여\n"
        "     훨씬 더 큰 페널티를 부과합니다.\n"
        "2. 이상치(Outlier) 및 대형 예측 실패에 대한 민감도:\n"
        "   - RMSE는 소수의 매우 큰 예측 오차(예: 실제 평점 5점을 1점으로 예측)에 크게 민감하게 반응합니다.\n"
        "   - 따라서 큰 실패를 치명적으로 피해야 하는 시스템에서는 RMSE가 더 엄격한 평가 지표가 됩니다.\n"
        "3. 지표 간 괴리와 해석:\n"
        "   - RMSE는 항상 MAE 이상(RMSE >= MAE)이며, 오차의 분산(극단적 오차의 빈도)이 클수록 두 지표의 차이가 벌어집니다.\n"
        "   - 표에서 모델 A -> B -> C로 갈수록 유저와 아이템 편향이 반영되면서 MAE뿐 아니라 특히 RMSE가 크게 감소하는 것은,\n"
        "     유저별 평가 성향 ( 및 아이템 성향) (후함/박함)을 고려함으로써 대형 오차가 효과적으로 줄어들었음을 의미합니다."
    )
    print(explanation)
    print("-" * 60)

    return results_df
