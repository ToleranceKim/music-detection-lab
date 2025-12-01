# Development Plan - Music Detection Lab

AI 생성 음악 탐지 연구를 위한 기술 개발 계획서
- Nicholas Sunday "Detecting Musical Deepfakes" (2025)
- Darius Afchar "AI-Generated Music Detection" (2025)

---

## 프로젝트 개요

### 목표
1. 두 가지 주요 AI 음악 탐지 논문의 재현
2. 성능 비교 및 분석
3. 통합 데모 시스템 구축

### 기술 스택
- **Python**: 3.12
- **PyTorch**: 2.2.2
- **Audio**: librosa 0.11.0, torchaudio
- **Web**: Streamlit
- **Package Manager**: uv

---

## Phase 1: 환경 설정 (완료)

- python 3.12 환경 구성
- uv 패키지 매니저 설치
- pyproject.toml 작성
- 기본 의존성 설치 (PyTorch, librosa)
- Git 저장소 초기화

---

## Phase 2: 공통 유틸리티 모듈 (진행중)

### 2.1 오디오 I/O (`src/common/audio_io.py`)

**목적**: 오디오 파일 로딩 및 전처리 표준화

**구현 함수**:
- `load_audio(file_path, sr=22050)` - 오디오 로딩 및 리샘플링
- `save_audio(audio, sr, file_path)` - 오디오 저장
- `split_audio(audio, sr, segment_duration=10)` - 세그먼트 분할
- `normalize_audio(audio)` - 정규화

### 2.2 특징 추출 (`src/common/features.py`)

**목적**: Mel-spectrogram 및 기타 특징 추출

**구현 함수**:
- `extract_mel_spectrogram(audio, sr, n_mels=128, n_fft=2048, hop_length=512)`
- `extract_mfcc(audio, sr, n_mfcc=13)`
- `spectrogram_to_image(spec)` - CNN 입력용 변환

### 2.3 데이터 증강 (`src/common/augment.py`)

**목적**: 강건성 향상을 위한 오디오 변조

**구현 함수**:
- `pitch_shift(audio, sr, n_steps)` - 피치 변조 (-2 ~ +2 semitones)
- `time_stretch(audio, rate)` - 템포 변조 (0.9 ~ 1.1)
- `add_noise(audio, noise_factor=0.005)` - 노이즈 추가
- `random_augment(audio, sr)` - 랜덤 증강 조합

### 2.4 평가 메트릭 (`src/common/metrics.py`)

**목적**: 모델 성능 평가 표준화

**구현 함수**:
- `calculate_accuracy(predictions, labels)`
- `calculate_auc_roc(predictions, labels)`
- `calculate_eer(predictions, labels)` - Equal Error Rate
- `plot_confusion_matrix(predictions, labels)`

**예상 소요 시간**: 3-4일

---

## Phase 3: Sunday 논문 재현

### 3.1 데이터셋 준비

**FakeMusicCaps 데이터셋**
- 다운로드 및 구조 파악
- 메타데이터 파싱
- Train/Val/Test 분할 (60/20/20)

### 3.2 모델 구현 (`src/sunday/model.py`)

```python
class SundayDetector(nn.Module):
    def __init__(self):
        # ResNet18 backbone
        # Binary classification head
```

**아키텍처**:
- Backbone: ResNet18 (ImageNet pretrained)
- Input: Mel-spectrogram (128 x T)
- Output: Binary (Real/Fake)

### 3.3 데이터 로더 (`src/sunday/dataset.py`)

```python
class FakeMusicCapsDataset(Dataset):
    def __init__(self, root_dir, split='train', augment=False):
        # 데이터 로딩
        # 전처리 파이프라인
```

### 3.4 학습 스크립트 ('src/sunday/train.py')

**학습 설정**:
- Optimizer : Adam (lr=le-4)
- Loss: Binary Cross Entropy
- Epochs: 50
- Batch size: 32

### 3.5 평가 ('src/sunday/evaluate.py')

**평가 시나리오**:
1. Clean test set
2. Pitch-shifted test set
3. Tempo-stretched test set
4. Combined augmentation

**예상 소요 시간**: 1주일

---

## Phase 4: Afchar 논문 재현

### 4.1 데이터셋 준비

**옵션**:
1. FMA (Free Music Archive) 사용
2. FakeMusicCaps 재사용
3. Custom Dataset 구성

### 4.2 오토인코더 모델 (`src/afchar/model.py`)

```python
class AfcharAutoencoder(nn.Module):
    def __init__(self):
        # Encoder network
        # Decoder network

class AfcharDatector(nn.Module):
    def __init__(self):
        # Raconstruction error 기반 탐지
```

**아키텍처**:
- Encoder: Conv layers -> Latent space
- Decoder: Deconv layers -> Reconstruction
- Detection: Reconstruction error threshold

### 4.3 학습 파이프라인 (`src/afchar/train.py`)

**2단계 학습**:
1. Autoencoder 학습 (정상 음악만)
2. Threshold 학습 (정상 vs AI 생성)

### 4.4 평가 (`src/afchar/evaluate.py`)

**비교 실험**:
- Sunday 모델과 동일 테스트셋
- 일반화 성능 비교
- 계산 효율성 비교

**예상 소요 시간**: 1주일

---

## Phase 5: 통합 및 비교 분석

### 5.1 통합 평가 프레임워크 (`experiments/compare.py`)

**비교 메트릭**:
- Accuracy, Precision, Recall, F1
- ROC-AUC, EER
- Inference time
- Model size

### 5.2 교차 데이터셋 실험

**실험 설계**:
- Sunday 모델 -> Afchar 데이터
- Afchar 모델 -> Sunday 데이터
- Domain adaptation 분석

### 5.3 앙상블 실험 (`experiments/ensemble.py`)

**앙상블 방법**:
- Voting ensemble
- Weighted average
- Stacking

**예상 소요 시간**: 3-4일

---

## Phase 6: 데모 개발

### 6.1 Streamlit 웹 앱 (`demos/app.py`)

**기능**:
- 오디오 파일 업로드
- 실시간 분석
- 두 모델 결과 비교
- 시각화 (스펙트로그램, 신뢰도 점수)

### 6.2 CLI ehrn (`demos/cli.py`)

```bash
python detect.py --model sunday --input audio.wav
python detect.py --model afchar --input audio.wav
python detect.py --model ensemble --input audio.wav
```

**예상 소요 시간**: 2-3일

---

## 리스크 및 대응

### 기술적 리스크
1. **데이터셋 접근 문제**
    - 대응: 대체 데이터셋 사용

2. **GPU 자원 부족**
    - 대응: 모델 크기 축소, Colab, Runpod 활용

3. **재현 성능 미달**
    - 대응: 하이퍼파라미터 튜닝, 저자 문의

---

상세 구현 사항은 dev_log.md 에 기록하며 진행합니다. 