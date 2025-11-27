# AI Music Defense Lab

**AI 생성 음악 탐지(AIGM)**와 **학습불가능 오디오 데이터/방어적 데이터 포이즈닝(ULD/DDP)** 연구실

## 개요

이 저장소는 다음 두 축의 실험과 코드를 모으기 위한 연구용 레포지토리입니다:

### 1. AI Generated Music Detection (AIGM)
- 텍스트 투 뮤직 기반 합성 음악과 인간 음악을 구분하는 탐지기 구현
- Nicholas Sunday, Darius Afchar, Yupei Li 등의 연구 재현 및 변형 실험

### 2. Unlearnable Data / Defensive Data Poisoning for Audio (ULD/DDP)
- 음악과 음성 데이터가 생성 모델과 화자 인식 모델 등에 학습되지 않도록 보호하는 기법 구현
- HarmonyCloak, SafeSpeech, PosCUDA, HiddenSpeaker 및 관련 ULD 이론 재현

## 프로젝트 구조 (예정)

초기 단계에서는 간단하게 시작하여 필요에 따라 확장 예정:
```
ai-music-defense-lab/
├── docs/                   # 연구 계획서 및 문서
│   └── research_plan.md    # 상세 연구 로드맵
├── notebooks/              # 실험용 Jupyter 노트북
├── src/                    # 핵심 코드 (추후 모듈별 확장)
├── data/                   # 데이터셋 (gitignore에 포함)
└── requirements.txt        # 의존성 패키지
```

## 연구 계획

상세 연구 목표와 방법론, 구현 로드맵:
- [**연구 계획서 전문**](docs/research_plan.md)

## 시작하기

### 요구사항
- Python 3.10+
- PyTorch
- 오디오 처리 라이브러리 (librosa, torchaudio)

### 설치

```bash
# 저장소 클론
git clone https://github.com/yourusername/ai-music-defense-lab.git
cd ai-music-defense-lab

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

## 현재 집중 과제

### 1단계 (우선순위)
- **A1**: Detecting Musical Deepfakes (Nicholas Sunday) 재현
  - FakeMusicCaps 데이터셋, CNN 기반 탐지기
- **U1**: SafeSpeech 재현
  - 악의적인 음성 합성으로부터 음성을 보호하는 기법

### 2단계
- **A2**: AI-Generated Music Detection (Darius Afchar/Deezer)
  - 오토인코더 기반 탐지
- **U2**: PosCUDA
  - 오디오 분류 작업을 위한 학습불가능 데이터

## 주요 참고 문헌

### AIGM 탐지
- Nicholas Sunday, "Detecting Musical Deepfakes" (2025)
- Darius Afchar et al., "AI-Generated Music Detection and its Challenges" (ICASSP 2025)

### Unlearnable Data
- Syed I. A. Meerza et al., "HarmonyCloak: Making Music Unlearnable for Generative AI" (2024)
- Zhisheng Zhang et al., "SafeSpeech: Robust and Universal Voice Protection" (USENIX Security 2025)