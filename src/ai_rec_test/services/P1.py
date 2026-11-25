from ..utils.util_P1 import *
from .parse_data import parse_table


def solve(df_problem=None):
    print("=" * 60)
    print("문제 1. 평가 프로토콜과 베이스라인")
    print("=" * 60)

    if df_problem is None:
        df_problem = parse_table()

    # 1. 학습·테스트 데이터 분할
    train, test = split_train_test(df_problem)

    # 2. 평점 수, 유저 수, 아이템 수 출력
    summary_df = get_data_summary(train, test)
    print("\n[데이터 분할 요약]")
    print(summary_df.to_string(index=False))

    # 3. 세 예측 모델 학습 (fit_baseline)
    model = fit_baseline(train)
    print("\n[학습된 베이스라인 파라미터]")
    print(f"- 전체 평균 (mu): {model['mu']:.4f}")
    print(f"- 학습 아이템 편향(b_i) 개수: {len(model['b_i'])} (평균: {model['b_i'].mean():.4f})")
    print(f"- 학습 유저 편향(b_u) 개수: {len(model['b_u'])} (평균: {model['b_u'].mean():.4f})")

    # 예측 샘플 확인
    test_sample = test.head(5).copy()
    pred_A = predict(model, test_sample, kind="A")
    pred_B = predict(model, test_sample, kind="B")
    pred_C = predict(model, test_sample, kind="C")

    sample_eval = test_sample[["user_id", "movie_id", "rating"]].copy()
    sample_eval["pred_A"] = np.round(pred_A, 3)
    sample_eval["pred_B"] = np.round(pred_B, 3)
    sample_eval["pred_C"] = np.round(pred_C, 3)
    print("\n[테스트 샘플 예측 결과 (상위 5건)]")
    print(sample_eval.to_string(index=False))

    # 4. 시간 기준 분할 이유 설명
    print("\n[설명: 무작위 분할 대신 시간 기준으로 분할하는 이유]")
    explanation = (
        "1. 실제 서비스 환경(현실성) 반영:\n"
        "   추천 시스템은 과거의 유저 행동 데이터를 학습하여 '미래'의 취향과 행동을 예측해야 합니다.\n"
        "2. 시간 역행 데이터 누출(Data Leakage) 방지:\n"
        "   무작위 분할(Random split)을 적용하면 미래 시점의 평점 데이터가 학습 셋에 포함되어,\n"
        "   시간적으로 앞선 테스트 평점을 예측하는 '시간 역행 누출'이 발생합니다.\n"
        "3. 트렌드 및 취향 드리프트(Concept Drift) 평가:\n"
        "   시간 분할을 사용해야 시간이 흐름에 따라 변화하는 유저의 취향 및 아이템의 트렌드 변화에\n"
        "   추천 모델이 얼마나 강건하고 적응력이 있는지 올바르게 평가할 수 있습니다."
    )
    print(explanation)
    print("-" * 60)

    return model, train, test
