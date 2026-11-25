import numpy as np


def rmse(y, y_hat) -> float:
    """Root Mean Squared Error (RMSE)를 계산합니다."""
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def mae(y, y_hat) -> float:
    """Mean Absolute Error (MAE)를 계산합니다."""
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    return float(np.mean(np.abs(y - y_hat)))


def mse(y, y_hat) -> float:
    """Mean Squared Error (MSE)를 계산합니다."""
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    return float(np.mean((y - y_hat) ** 2))


def calculate_improvement_rate(baseline_error: float, new_error: float) -> float:
    """개선율 (%)을 계산합니다: ((baseline - new) / baseline) * 100"""
    return float((baseline_error - new_error) / baseline_error * 100.0)


# 기존 템플릿 함수 호환성 지원
RMSE = rmse
MAE = mae
MSE = mse
