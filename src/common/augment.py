"""
Audio Data Augmentation Module

AI 생성 음악 탐지 모델의 성능 향상을 위한 데이터 증강 기법들.
과적합을 방지하고 모델의 일반화 능력을 향상시킵니다.

주요 증강 기법:
- Time Domain: stretching, shifting, volume control
- Frequency Domain: pitch shifting, filtering, masking
- Noise: gaussian, background, environmental
- Advanced: mix-up, spectral augmentation

References:
    - Park et al. "SpecAugment: A Simple Data Augmentation Method" (2019)
    - Ko et al. "Audio Augmentation for Speech Recognition" (2015)
"""

import numpy as np
import librosa
from typing import Optional, Tuple, Union, List, Dict
import random
from scipy import signal
import warnings

# Time Domain Augmentations

def time_stretch(audio: np.ndarray, sr: int, rate: float = 1.0) -> np.ndarray:
    """
    오디오의 재생 속도를 변경합니다 (피치는 유지).

    Phase vocoder를 사용하여 피치를 유지하면서 템포만 변경합니다.
    AI 생성 음악이 다양한 템포에서도 탐지되도록 학습시킵니다.

    Args:
        audio: 입력 오디오 신호 (1차원 numpy 배열)
        sr: 샘플레이트 (Hz)
        rate: 속도 변경 비율
            - 0.5: 절반 속도 (2배 길이)
            - 1.0: 원본 속도
            - 2.0: 2배 속도 (절반 길이)

    Returns:
        np.ndarray: 속도 변경된 오디오

    Raises:
        ValueError: rate가 0.1보다 작거나 10보다 큰 경우

    Example:
        >>> stretched = time_stretch(audio, sr, rate=1.2) # 20% 빠르게
        >>> print(f"Length change: {len(audio)} -> {len(stretched)}")

    Note:
        - rate가 1에서 멀어질수록 아티팩트가 발생할 수 있음
        - 극단적인 rate (< 0.5 또는 > 2.0)는 음질 저하 가능
    """
    if rate <= 0.1 or rate >= 10.0:
        raise ValueError(f"Rate should be between 0.1 and 10.0, got {rate}")

    if rate == 1.0:
        return audio

    # librosa의 phase vocoder 사용
    return librosa.effects.time_stretch(audio, rate=rate)

def time_shift(audio: np.ndarray, sr: int, shift_ms: float = 0.0, fill_mode: str = 'zeros') -> np.ndarray:
    """
    오디오를 시간축에서 이동시킵니다.

    순환 이동 또는 제로 패딩을 통해 시간축 이동을 구현합니다.
    모델이 오디오의 시작 위치에 민감하지 않도록 학습시킵니다.
    
    Args:
        audio: 입력 오디오 신호
        sr: 샘플레이트 (Hz)
        shift_ms: 이동할 밀리초
            - 양수: 오른쪽 이동 (지연)
            - 음수: 왼쪽 이동 (앞당김)
        fill_mode: 빈 공간 채우기 방식
            - 'zeros': 0으로 채우기
            - 'wrap': 순환 이동
    
    Returns:
        np.ndarray: 시간 이동된 오디오 (원본과 같은 길이)

    Example:
        >>> shifted = time_shift(audio, sr, shift_ms=100) # 100ms 지연
        >>> shifted_wrap = time_shift(audio, sr, -50, fill_mode='wrap')
    
    Note:
        - 이동량이 오디오 길이를 초과하면 무음 또는 완전 순환
        - wrap 모드는 리듬 패턴 유지에 유용
    """
    shift_samples = int(sr * shift_ms / 1000)

    if shift_samples == 0:
        return audio

    if fill_mode == 'wrap':
        # 순환 이동
        return np.roll(audio, shift_samples)
    else:   #  zeros
        if shift_samples > 0:
            # 오른쪽 이동 (앞에 0 추가)
            if shift_samples >= len(audio):
                return np.zeros_like(audio)
            return np.pad(audio[:-shift_samples], (shift_samples, 0), mode='constant')
        else:
            # 왼쪽 이동 (뒤에 0 추가)
            shift_samples = abs(shift_samples)
            if shift_samples >= len(audio):
                return np.zeros_like(audio)
            return np.pad(audio[shift_samples:], (0, shift_samples), mode='constant')

def change_volume(audio: np.ndarray, gain_db: float = 0.0, normalize: bool = False) -> np.ndarray:
    """
    오디오 음량을 조절합니다.
    
    데시벨 단위로 Gain을 적용하여 음량을 조절합니다.
    다양한 녹음 레벨에 강건한 모델을 학습시킵니다.

    Args:
        audio: 입력 오디오 신호
        gain_db: Gain (dB 단위)
            - 양수: 증폭
            - 음수: 감쇄
            - 6dB ≈ 2배 음량
            - -6dB ≈ 절반 음량
        normalize: True 면 클리핑 방지를 위해 정규화

    Returns:
        np.ndarray: 음량 조절된 오디오

    Example:
        >>> louder = change_volume(audio, gain_db=6) # 2배 증폭
        >>> quieter = change_volume(audio, gain_db=-10, normalize=True)
    
    Warning:
        Gain이 너무 크면 클리핑이 발생할 수 있음.
        normalize=True 사용 권장.
    """

    gain = 10 ** (gain_db / 20)
    augmented = audio * gain

    if normalize and np.max(np.abs(augmented)) > 1.0:
        # 클리핑 방지
        augmented = augmented / np.max(np.abs(augmented))

    return augmented

def polarity_inversion(audio: np.ndarray) -> np.ndarray:
    """
    오디오 신호의 극성을 반전시킵니다.

    위상을 180도 반전시켜 신호의 극성을 바꿉니다.
    사람 귀에는 차이가 없지만 모델 학습에 도움이 됩니다.

    Args:
        audio: 입력 오디오 신호
    
    Returns:
        np.ndarray: 극성 반전된 오디오

    Example:
        >>> inverted = polarity_inversion(audio)
        >>> print(f"Max before: {np.max(audio)}, max after: {np.max(inverted)}")

    """
    return -audio


# Frequency Domain Augmentations

def pitch_shift(audio: np.ndarray, sr: int, n_steps: float = 0.0, bins_per_octave: int = 12) -> np.ndarray:
    """
    오디오 피치를 변경합니다. (속도는 유지)

    STFT와 phase vocoder를 사용하여 템포를 유지하면서 피치만 변경합니다.
    다양한 키의 음악에 대해 강건한 모델을 학습시킵니다.

    Args:
        audio: 입력 오디오 신호
        sr: 샘플레이트 (Hz)
        n_steps: 반음 단위 변경량
            - 양수: 높은 음으로 (예: 2 = 2반음 올림)
            - 음수: 낮은 음으로 (예: -3 = 3반음 내림)
            - 12 = 1옥타브
        bins_per_octave: 옥타브당 빈 수 (미세 조정용)

    Returns:
        np.ndarray: 피치 변경된 오디오

    Example:
        >>> higher = pitch_shift(audio, sr, n_steps=2) # 2반음 올리기
        >>> lower = pitch_shift(audio, sr, n_steps=-5) # 5반음 내리기
        >>> octave_up = pitch_shift(audio, sr, n_steps=12) # 1옥타브 올리기

    Note:
        - 극단적인 피치 변경 (±12 이상)은 아티팩트 발생 가능
        - 실시간 처리에는 부적합 (계산량 많음)
    """
    if n_steps == 0:
        return audio

    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps, bins_per_octave=bins_per_octave)

def frequency_mask(audio: np.ndarray, sr: int, freq_mask_range: Tuple[float, float] = (0, 0), mask_type: str = 'zero') -> np.ndarray:
    """
    특정 주파수 대역을 마스킹합니다.

    STFT 도메인에서 특정 주파수 대역을 제거하거나 감쇄시킵니다.
    주파수 대역 손실에 강건한 모델을 학습시킵니다.

    Args:
        audio: 입력 오디오 신호
        sr: 샘플레이트 (Hz)
        freq_mask_range: 마스킹할 주파수 범위 (Hz)
            - (low_freq, high_freq) 튜플
        mask_type: 마스킹 방식
            - 'zero': 완전 제거
            - 'attenuate': 50% 감쇄

    Returns:
        np.ndarray: 주파수 마스킹된 오디오

    Example:
        >>> # 보컬 주파수 대역 마스킹
        >>> masked = frequency_mask(audio, sr, (85, 255))
        >>> # 고주파 제거
        >>> no_highs = frequency_mask(audio, sr, (4000, 8000))

    Note:
        - 극단적인 마스킹은 음질 저하
        - 저주파 마스킹은 베이스/드럼 제거 효과
        - 중간 주파수 마스킹은 보컬/멜로디 제거 효과
    """
    if freq_mask_range == (0, 0):
        return audio

    # STFT 변환
    D = librosa.stft(audio)
    freqs = librosa.fft_frequencies(sr=sr)

    # 마스킹 인덱스 찾기
    mask_idx = np.where((freqs >= freq_mask_range[0]) & (freqs <= freq_mask_range[1]))[0]

    if len(mask_idx) > 0:
        if mask_type == 'zero':
            D[mask_idx, :] = 0
        elif mask_type == 'attenuate':
            D[mask_idx, :] *= 0.5

    # 역변환
    return librosa.istft(D)

def apply_filter(audio: np.ndarray, sr: int, filter_type: str = 'lowpass', cutoff_freq: float = 1000.0, order: int = 5) -> np.ndarray:
    """
    다양한 주파수 필터를 적용합니다.

    Butterworth 필터를 사용하여 특정 주파수 대역을 필터링합니다.
    다양한 주파수 특성에 강건한 모델을 학습시킵니다.

    Args:
        audio: 입력 오디오 신호
        sr: 샘플레이트 (Hz)
        filter_type: 필터 종류
            - 'lowpass': 저역통과 (고주파 제거)
            - 'highpass': 고역통과 (저주파 제거)
            - 'bandpass': 대역통과 (특정 대역만 통과)
        cutoff_freq: 차단 주파수 (Hz)
            - bandpass의 경우 (low, high) 튜플
        order: 필터 차수 (높을수록 가파른 차단)

    Returns:
        np.ndarray: 필터링된 오디오

    Example:
        >>> # 저역통과 필터 (전화 음질 시뮬레이션)
        >>> filtered = apply_filter(audio, sr, 'lowpass', 3400)
        >>> # 고역통과 필터 (베이스 제거)
        >>> no_bass = apply_filter(audio, sr, 'highpass', 200)

    Note:
        - 너무 낮은/높은 차단 주파수는 무음 생성 가능
        - order가 높으면 ringing 아티팩트 발생 가능
    """
    nyquist = sr / 2

    if filter_type == 'bandpass':
        if not isinstance(cutoff_freq, tuple):
            raise ValueError("For bandpass, cutoff_freq must be (low, high) tuple")
        low, high = cutoff_freq
        low = low / nyquist
        high = high / nyquist
        b, a = signal.butter(order, [low, high], btype='band')
    else:
        normalized_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(order, normalized_cutoff, btype=filter_type)

    # 필터 적용
    filtered = signal.filtfilt(b, a, audio)
    return filtered




