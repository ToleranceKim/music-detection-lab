# AI-Generated Music Detection & Unlearnable Data 연구 계획서

본 문서는 AI 생성 음악 탐지기(AIGM detection)와 Unlearnable Data 및 Defensive Data Poisoning(이하 ULD/DDP) 연구를 장기적으로 수행하기 위한 전체 로드맵을 정리한 계획서임.

---

## 1. 연구 목표 및 전체 구조

### 1.1 최종 목표

- AI 생성 음악 탐지와 음악·음성 데이터 방어 기법을 직접 재현하고 변형하는 것을 목표로 함.
- Python 기반 코드와 간단한 웹 데모 수준까지 구현하는 것을 1차 실용 목표로 설정.
  - 오디오 처리 파이프라인 구축
  - 모델 학습 및 평가 코드 구현
  - Streamlit 기반 인터랙티브 데모 앱 제작

### 1.2 연구 축 구분

연구는 크게 두 개의 축으로 진행함.

1. **축 A. AI Generated Music Detection (AIGM 탐지기)**
   - Text to Music 기반 합성 음악과 인간 음악을 구분하는 탐지 모델 연구
   - 대표 논문
     - Nicholas Sunday, *Detecting Musical Deepfakes* (2025)
     - Darius Afchar et al., *AI-Generated Music Detection and its Challenges* (ICASSP 2025)

2. **축 B. Unlearnable Data / Defensive Data Poisoning (ULD/DDP)**
   - 음악·음성 데이터가 생성 모델이나 화자 인식 모델에 학습되지 못하도록 보호하는 기법 연구
   - 대표 논문
     - Syed I. A. Meerza et al., *HarmonyCloak: Making Music Unlearnable for Generative AI* (2024)
     - Zhisheng Zhang et al., *SafeSpeech: Robust and Universal Voice Protection Against Malicious Speech Synthesis* (USENIX Security 2025)

---

## 2. 공통 기술 스택 및 환경

### 2.1 개발 환경

- 언어
  - Python 3.10 이상
- 주요 라이브러리
  - 오디오 전처리: `librosa`, `torchaudio`
  - 딥러닝: `torch`, `torchvision`
  - 데이터 처리: `numpy`, `pandas`
  - 시각화 및 분석: `matplotlib`, `seaborn`(필요 시), `scikit-learn`
  - 웹 데모: `streamlit`
- 실험 관리(선택)
  - `wandb` 또는 `tensorboard`

### 2.2 하드웨어 가정

- 기본 가정
  - CPU 환경에서 돌아갈 수 있는 축소 실험을 우선 설계
  - 데이터 서브셋, 짧은 오디오 클립, 작은 batch size 사용
- 확장 가정
  - Sunday 전체 실험, Afchar 전체 파이프라인, SafeSpeech 전 범위 재현 등은 GPU 사용이 요구될 가능성이 높음
  - 연구 중요도와 난이도를 고려하여 필요 시 GPU 환경을 추가 확보하는 것을 장기 옵션으로 둠

### 2.3 공통 유틸리티 구조

향후 여러 논문을 하나의 코드베이스에서 다루기 위해 다음과 같은 공통 모듈을 설계함.

- `audio_io.py`
  - 오디오 파일 읽기, 리샘플링, 채널 모노 변환
- `features.py`
  - Mel 스펙트로그램, STFT, 기타 스펙트럼 특징 추출 함수
- `augment.py`
  - 피치 시프트, 템포 스트레치, 노이즈 추가 등 데이터 변조 함수
- `train_utils.py`
  - 공통 학습 루프, 로그 기록, 체크포인트 저장
- `eval_utils.py`
  - 정확도, ROC AUC, EER, Confusion matrix 등 공통 평가 함수

---

## 3. 축 A: AI Generated Music Detection 연구 계획

### A 단계 전체 구조

- **A1. Sunday 재현**
  FakeMusicCaps 서브셋 기반, Mel 스펙트로그램 CNN 탐지기 재현 및 Streamlit 데모 구현
- **A2. Afchar 재현**
  Deezer 오토인코더 기반 탐지기 재현 및 Sunday와 비교, 일반화·강건성 실험
- **A3. 심화 확장**
  Fourier 아티팩트, 가사 기반 멀티뷰 탐지, SONICS 데이터셋 활용 연구로 확장

---

### A1. Detecting Musical Deepfakes (Sunday) 재현

#### A1.1 목표

- FakeMusicCaps 기반의 간단한 AIGM 탐지 파이프라인을 재현함.
- CPU 환경에서도 동작 가능한 축소 버전 모델을 구현하고, Streamlit 데모까지 연결하는 것을 1차 목표로 함.

#### A1.2 참고 논문 및 코드

- Nicholas Sunday, *Detecting Musical Deepfakes* (2025)
  FakeMusicCaps 데이터셋과 ResNet18 계열 CNN을 사용한 딥페이크 음악 탐지기를 제안하고 코드와 실험을 공개한 논문으로 알려져 있음.

#### A1.3 데이터 설계

- 데이터셋
  - FakeMusicCaps에서 인간 음악과 AI 합성 음악을 균형 있게 샘플링
  - CPU 기준 실험을 위해 각 클래스당 수천 곡 수준의 서브셋으로 시작
- 전처리
  - 오디오를 10초 혹은 15초 단위 클립으로 분할
  - 샘플링레이트 통일 후 Mel 스펙트로그램 추출

#### A1.4 모델 및 학습

- 모델
  - ResNet18 또는 경량 CNN 구조
- 학습 절차
  - 기본 설정
    - 손실 함수: Binary cross entropy
    - 옵티마이저: Adam
    - 간단한 학습 스케줄
  - 실험 시나리오
    1. 변조 없는 데이터로 학습 및 테스트
    2. 피치 시프트, 템포 스트레치 등의 변조를 포함한 학습 및 테스트
- 평가 지표
  - 정확도, ROC AUC, Confusion matrix
  - 변조 강도별 성능 변화 비교

#### A1.5 Streamlit 데모 계획

- 입력
  - 사용자가 업로드한 오디오 파일
- 처리
  - 동일한 전처리 파이프라인 통해 Mel 스펙트로그램 생성
  - 학습된 모델로 AI 생성 가능성 추정
- 출력
  - AI 생성 음악일 확률
  - 간단한 결과 요약 텍스트

---

### A2. AI Generated Music Detection and its Challenges (Afchar, Deezer) 재현

#### A2.1 목표

- Deezer Research의 AI 음악 탐지기 구조를 실제로 재현함.
- Sunday 파이프라인과 비교하여, 오토인코더 기반 탐지기의 특성과 일반화 문제를 직접 확인함.

#### A2.2 참고 논문 및 코드

- Darius Afchar et al., *AI-Generated Music Detection and its Challenges* (ICASSP 2025)
  오토인코더로 재구성된 오디오와 원본 오디오를 구분하는 탐지기를 통해 약 99.8퍼센트의 높은 정확도를 보고하며, 일반화 및 강건성 문제를 강조하는 논문으로 알려져 있음.
- Deezer GitHub 레포지토리
  논문 관련 코드와 실험 스크립트가 PyTorch 기반으로 공개되어 있음.

#### A2.3 데이터 및 파이프라인

- 데이터
  - FMA(Free Music Archive) 데이터셋의 small 또는 medium 서브셋 사용
- 파이프라인
  1. FMA 곡에 대해 오토인코더를 학습하여 재구성본 생성
  2. 원본 vs 재구성본 이진 분류 모델 학습
  3. 탐지기가 실제 AI 생성 음악에 대해 어느 정도 성능을 보이는지 추가 실험 설계 가능

#### A2.4 실험 및 비교

- Sunday vs Deezer 비교
  - 동일하거나 유사한 테스트셋에서 두 모델의 성능과 오류 패턴 비교
  - 어떤 종류의 변조나 공격에 더 취약한지 분석
- 강건성 실험
  - 피치 변조, 템포 변조, 노이즈 추가 등 다양한 변형 조건에서 성능 변화 측정

#### A2.5 확장 데모 아이디어

- Streamlit 대시보드에서
  - Sunday 모델 점수
  - Afchar 모델 점수를 동시에 표시
  - 두 모델의 판단 차이를 시각적 또는 수치적으로 비교

---

### A3. 심화 과제(중기 이후)

- **A3.1 Fourier 기반 아티팩트 설명**
  - Afchar, *A Fourier Explanation of AI-music Artifacts*
    생성기 구조로 인해 주파수 영역에 나타나는 특유의 피크와 체커보드 아티팩트를 분석하고, 간단한 스펙트럼 기반 탐지기를 구현하는 심화 과제로 설정.
- **A3.2 Double Entendre 멀티뷰 탐지**
  - Frohmann et al., *Double Entendre*
    자동 전사된 가사와 음성 임베딩을 결합해 노이즈에 강한 멀티뷰 탐지기를 구현하는 것을 중장기 목표로 설정.
- **A3.3 SONICS, FakeMusicCaps 대규모 확장**
  - SONICS와 FakeMusicCaps 전체 스케일에서 성능과 일반화를 확인하는 실험은 GPU 자원이 안정적으로 확보된 이후 중기 과제로 둠.

---

## 4. 축 B: Unlearnable Data 및 Defensive Data Poisoning 연구 계획

### U 단계 전체 구조

- **U0. ULD 이론 정리**
  - Unlearnable Examples, A Survey on Unlearnable Data 등 이론적 기반 정리
- **U1. SafeSpeech 재현**
  - 실용적이고 코드가 공개된 오디오 보호 ULD 기법 우선 재현
- **U2. PosCUDA 및 HiddenSpeaker 토이 실험**
  - 오디오 분류 및 화자 인식에 대한 ULD 실험
- **U3. HarmonyCloak 개념적 실험**
  - 음악 생성 모델에 대한 ULD를 장기 과제로 설정

---

### U0. ULD 이론 정리

#### U0.1 주요 문헌

- Hanxun Huang et al., *Unlearnable Examples: Making Personal Data Unexploitable* (ICLR 2021)
- Yujing Jiang et al., *Unlearnable Examples for Time Series* (PAKDD 2024)
- Jiahao Li et al., *A Survey on Unlearnable Data* (2025, 서베이 논문)
- Syed I. A. Meerza et al., *HarmonyCloak: Making Music Unlearnable for Generative AI*

#### U0.2 정리 목표

- Unlearnable Data를
  - adversarial attack
  - machine unlearning
  - backdoor 공격
  과 비교하여 개념적 위치를 명확히 정리
- 평가 지표
  - unlearnability
  - imperceptibility
  - robustness
  - computational efficiency
  네 축으로 정리
- 이후 SafeSpeech, HarmonyCloak, PosCUDA를 이 틀 안에 배치

---

### U1. SafeSpeech 재현

#### U1.1 목표

- SafeSpeech 파이프라인을 축소 버전으로 재현하여, 사용자의 음성을 보호하는 ULD 전처리 모듈을 구현.
- 간단한 Streamlit 데모를 통해 "음성 업로드 → 보호된 음성 다운로드" 흐름을 구현.

#### U1.2 참고 논문 및 코드

- Zhisheng Zhang et al., *SafeSpeech: Robust and Universal Voice Protection Against Malicious Speech Synthesis* (USENIX Security 2025)
  음성 합성 및 voice cloning 모델이 학습하지 못하도록 SPEC(Speech Perturbative Concealment)을 삽입하는 프레임워크를 제안하며, 높은 방어율과 실험 결과를 보고하고 코드와 artifact를 공개한 것으로 알려져 있음.

#### U1.3 데이터 및 모델

- 데이터
  - LibriTTS 또는 VCTK의 일부 서브셋 사용
  - 짧은 발화 중심으로 축소된 실험 세트 구성
- 모델
  - 공개된 VITS 또는 FastSpeech 계열 TTS 모델 한 종을 baseline으로 사용

#### U1.4 실험 설계

- 단계
  1. Clean 음성 데이터로 TTS baseline 구축
  2. SafeSpeech 코드에 따라 음성에 SPEC 노이즈 삽입
  3. 보호된 음성으로 동일 TTS를 재학습 또는 테스트
- 평가
  - 청취 품질
    - 주관적 MOS 스타일 평가 혹은 간단한 내부 스코어
  - 방어 효과
    - 화자 유사성 감소 정도를 임베딩 거리 등으로 측정
    - 공격자 관점에서 voice cloning이 얼마나 실패하는지 정량화

#### U1.5 Streamlit 데모

- 입력
  - 사용자 음성 파일 업로드
- 처리
  - SafeSpeech 기반 보호 노이즈 삽입
- 출력
  - 보호된 음성 파일 다운로드 링크
  - 간단한 보호 상태 메시지

---

### U2. PosCUDA 및 HiddenSpeaker 토이 실험

#### U2.1 PosCUDA

- Vignesh Gokul, Shlomo Dubnov, *PosCUDA: Position Based Convolution for Unlearnable Audio Datasets*
  SpeechCommands 등 오디오 분류 데이터셋에 대해 클래스별 위치 기반 컨볼루션 블러를 적용해 학습을 방해하는 방법을 제안한 논문으로 알려져 있음.
- 계획
  - SpeechCommands 또는 FSDD 같은 작은 데이터셋에서
    - Clean 학습
    - PosCUDA 처리 후 학습
  - 분류 정확도의 하락과 unlearnability 정도를 비교
- 주의
  - 코드 공개 여부는 아직 확실하지 않으며, 논문 설명만으로 구현해야 할 가능성이 있음

#### U2.2 HiddenSpeaker

- Yujing Jiang et al., *HiddenSpeaker: Generating Imperceptible Unlearnable Audio to Protect Speaker Identity*
  화자 검증 모델에 대해 화자 음성을 unlearnable하게 만드는 기법을 제안하는 연구로 알려져 있음.
- 계획
  - SafeSpeech 이후 화자 인식 환경에서 유사 개념을 실험하는 토이 과제로 고려
- 주의
  - 코드 공개 여부가 확실하지 않기 때문에, 실제 구현 난도는 논문 세부를 다시 확인해야 함

---

### U3. HarmonyCloak 개념적 실험(장기 과제)

#### U3.1 개요

- HarmonyCloak은 음악 데이터를 텍스트 투 뮤직 모델이 학습하지 못하도록 error minimizing 노이즈를 삽입하는 최초의 음악 특화 ULD 프레임워크로 제안되었다고 알려져 있음.
- MuseGAN, SymphonyNet, MusicLM 계열 모델을 대상으로 한 실험 결과를 보고.

#### U3.2 장기 계획

- 단기
  - 논문의 수식, 손실 함수, 노이즈 설계 방식을 상세히 정리
- 이후
  - 공개된 MusicGen 계열 모델을 surrogate로 사용하여
    - 짧은 멜로디 또는 루프를 대상으로 축소 실험
  - 노이즈 삽입 전후의 생성 품질과 프롬프트 정합도 변화를 조사
- 이 단계는 계산 비용과 구현 난도가 높으므로, A1 A2 U1 U2를 충분히 수행한 이후 박사급 장기 과제로 설정

---

## 5. 단계별 우선순위 정리

### 5.1 1차 우선순위

1. **A1. Detecting Musical Deepfakes (Sunday) 재현**
   - FakeMusicCaps 서브셋, Mel 스펙트로그램, ResNet18 기반 탐지기
   - 간단한 Streamlit 데모 포함
2. **U1. SafeSpeech 재현**
   - LibriTTS 또는 VCTK 서브셋, 단일 TTS 모델 기준 축소 실험
   - 보호 전후 음성 비교 및 간단한 데모 구현

### 5.2 2차 우선순위

1. **A2. Afchar AI Generated Music Detection 재현**
   - Deezer 오토인코더 기반 탐지기 재현
   - Sunday 모델과의 비교 및 강건성 실험
2. **U2. PosCUDA 토이 실험**
   - SpeechCommands 등 소규모 데이터셋에서 오디오 분류용 ULD 실험

### 5.3 3차 우선순위

1. **A3. Fourier 아티팩트, Double Entendre, SONICS 심화 연구**
2. **U3. HarmonyCloak 개념적 실험 및 MusicGen 기반 축소 구현**

---

## 6. 바로 다음 실행 단계 체크리스트

1. Python 가상환경 생성 및 필수 패키지 설치
2. 공통 유틸리티 모듈 스켈레톤 작성
   - 오디오 로딩, 스펙트로그램 변환, 간단 학습 루프
3. Sunday 실험을 위한 FakeMusicCaps 서브셋 다운로드 및 구조 파악
4. SafeSpeech GitHub 레포지토리와 USENIX artifact 문서를 읽고 축소 재현 경로 파악
5. 위 네 가지가 정리되면
   - `README.md`에 본 계획서를 포함하고
   - 각 단계별로 별도의 디렉토리 및 노트북 혹은 스크립트를 추가

---

## 7. 이 계획에서 불확실한 부분

- HarmonyCloak, PosCUDA, HiddenSpeaker의 **코드 공개 범위와 최신 구현 상태**는 논문과 공식 페이지를 다시 확인해야 함.
  현재 계획서는 코드가 완전히 공개되지 않았을 가능성을 전제로 작성되었으며, 실제 구현 난도는 추후 확인이 필요함.
- SafeSpeech, Sunday, Afchar 등은 코드 및 데이터셋이 공개된 것으로 알려져 있으나,
  실제 실행 시 요구되는 GPU 메모리와 학습 시간은 실험 규모와 설정에 따라 달라질 수 있으므로,
  초기에는 작은 서브셋 기준으로 탐색하는 것이 안전함.