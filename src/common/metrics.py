"""
AI 음악 탐지 모델을 위한 평가 메트릭.

이진 분류 모델 평가를 위한 다양한 메트릭들을 제공하며,
특히 AI 생성 음악 탐지에 최적화되어 있습니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    classification_report, roc_auc_score,
    precision_recall_fscore_support
)
from typing import List, Tuple, Optional, Union, Dict
import warnings
warnings.filterwarnings('ignore')

# Seaborn 스타일 설정
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12

def calculate_accuracy(predictions: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    """
    이진 분류의 정확도를 계산합니다.

    Args:
        predictions: 모델 예측값 (확률 또는 로짓)
        labels: 실제 레이블 (0 또는 1)
        threshold: 이진 분류를 위한 결정 경계값

    Returns:
        0과 1 사이의 정확도 점수

    Examples:
        >>> preds = np.array([0.7, 0.3, 0.8, 0.2])
        >>> labels = np.array([1, 0, 1, 0])
        >>> acc = calculate_accuracy(preds, labels)
        >>> print(f"Accuracy: {acc:.4f}")
        Accuracy: 1.0000
    """
    # 확률값을 이진 예측으로 변환
    if predictions.ndim > 1:
        predictions = predictions[:, 1] # positive class 확률

    binary_preds = (predictions >= threshold).astype(int)

    # 정확도 계산
    correct = (binary_preds == labels).sum()
    total = len(labels)

    return correct / total if total > 0 else 0.0


def calculate_metrics(predictions: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    이진 분류를 위한 종합적인 메트릭을 계산합니다.

    Args:
        predictions: 모델 예측값
        labels: 실제 레이블
        threshold: 결정 경계값

    Returns:
        다양한 메트릭을 포함한 딕셔너리

    Note:
        Class 0 = Human (정상 음악)
        Class 1 = Deepfake (AI 생성 음악)
    """
    # 이진 예측으로 변환
    if predictions.ndim > 1:
        predictions = predictions[:, 1]

    binary_preds = (predictions >= threshold).astype(int)

    # Confusion Matrix 요소들
    tn, fp, fn, tp = confusion_matrix(labels, binary_preds).raval()

    # 각종 메트릭 계산
    metrics = {
        'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0, # TPR
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0, # TNR
        'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0, # False Positive Rate
        'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0, # False Negative Rate
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

    return metrics

# AUC-ROC

def calculate_auc_roc(predictions: np.ndarray, labels: np.ndarray, return_curve: bool = False) -> Union[float, Tuple[float, np.ndarray, np.ndarray]]:
    """
    AUC-ROC 점수를 계산하고 선택적으로 ROC 곡선을 반환합니다.

    Args:
        predictions: 모델 예측값 (확률)
        labels: 실제 레이블
        return_curve: FPR과 TPR 배열 반환 여부

    Returns:
        AUC 점수, 또는 return_curve=True일 때 (AUC, FPR, TPR) 튜플

    Examples:
        >>> auc_score = calculate_auc_roc(preds, labels)
        >>> auc_score, fpr, tpr = calculate_auc_roc(preds, labels, return_curve=True)
    """
    if predictions.ndim > 1:
        predictions = predictions[:, 1]

    # ROC curve 계산
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    auc_score = auc(fpr, tpr)

    if return_curve:
        return auc_score, fpr, tpr
    return auc_score

# EER

def calculate_eer(predictions: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """
    Equal Error Rate (EER) - FPR과 FNR이 같아지는 지점을 계산합니다.

    Args:
        predictions: 모델 예측값
        labels: 실제 레이블

    Returns:
        (EER 값, EER 임계값) 튜플

    Note:
        EER은 보안 시스템에서 중요한 메트릭으로,
        사람 음악을 AI로 오인하는 비율과
        AI 음악을 사람으로 놓치는 비율이 같아지는 지점입니다.
    """
    if predictions.ndim > 1:
        predictions = predictions[:, 1]

    # ROC curve 계산
    fpr, tpr, thresholds = roc_curve(labels, predictions)

    # FNR = 1 - TPR
    fnr = 1 - tpr

    # FPR과 FNR의 차이가 최소인 지점 찾기
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    eer_threshold = thresholds[eer_index]

    return eer, eer_threshold

    


