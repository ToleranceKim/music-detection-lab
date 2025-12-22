# Detecting Musical Deepfakes

- title: "[논문 리뷰] Detecting Musical Deepfakes – FakeMusicCaps와 ResNet18 기반 AI 음악 딥페이크 탐지"
- date: 2025-12-05
- tags: [AI Music Detection, Deepfake, FakeMusicCaps, ResNet18]
---

# Detecting Musical Deepfakes: FakeMusicCaps와 ResNet18으로 살펴본 음악 딥페이크 탐지

## 리뷰를 시작하며

오늘 다룰 Nicholas Sunday의 「Detecting Musical Deepfakes」는 FakeMusicCaps 데이터셋의 오디오를 Mel Spectrogram으로 변환한 뒤 ResNet18 이진 분류기를 학습하는 기본적인 탐지 파이프라인을 제안합니다. 논문과 함께 공식 GitHub 레포지토리가 공개되어 있어 코드를 내려받아 곧바로 학습과 평가를 재현해 볼 수 있고, 전처리와 실험 설정을 바꾸어 보는 것도 비교적 수월합니다.

저자는 Deezer의 연구(Afchar et al, 2024)와 SONICS, FakeMusicCaps 논문을 주요 관련 연구로 검토합니다. 특히 Deezer 연구에서 사용된 피치 쉬프트와 템포 스트레치 조작 시나리오를 FakeMusicCaps와 ResNet18 구성에 다시 적용해, 이런 단순 효과가 탐지 성능을 얼마나 흔들 수 있는지 실험합니다. 이 점에서 Sunday 논문은, SONICS 수준의 복잡한 아키텍처로 넘어가기 전에 "짧은 10초 클립, Mel Spectrogram, 범용 CNN" 이라는 직관적인 조합이 어디까지 작동하는지 보여 주는 출발점 역할을 합니다.

이 리뷰에서는 Sunday가 제안한 파이프라인을 실제로 실행해 본 경험을 전제로, 이 구성이 AI Music Detection을 이해하기 위한 직관적인 시작점으로서 어느 정도까지 유효한지, 또 어떤 한계를 드러내는지를 정리합니다. 사용자 재현 실험의 구체적인 수치와 비교표는 향후 별도의 포스트에서 다룰 예정이며, 이 글에서는 원 논문이 보고하는 내용과 구조 분석에 집중합니다. 

---

> 핵심 요약
> Sunday의  「Detecting Musical Deepfakes」는 FakeMusicCaps 데이터셋과 Mel Spectrogram + ResNet18 이진 분류기를 사용해 사람vs딥페이크 음악 탐지 문제를 실험적으로 분석한 연구입니다. 논문에 따르면, 10초 단위 오디오 클립 10,746개(사람 5,373 / 딥페이크 5,373)를 학습과 평가에 사용했을 때 모든 실험에서 F1, Accuracy, Recall, Precision이 80%를 상회하는 비교적 높은 성능을 달성합니다. 또한 Deezer 연구를 참조해 피치 쉬프트와 템포 스트레치 같은 단순 조작이 탐지 성능을 얼마나 떨어뜨리는지 측정하고 여러 조작 데이터셋을 연속적으로 학습하는 Continuous Learning 설정이 사람 음악 재현율을 높이는 대신 오탐률을 크게 증가시키는 트레이드오프를 보여 줍니다.

---

## 논문 정보

|항목|내용|
|---|---|
|제목|Detecting Musical Deepfakes|
|저자|Nicholas Sunday, Department of Computer Science, The University of Texas at Austin, USA|
|소속|Department of Computer Science, The University of Texas at Austin, USA|
|유형|arXiv preprint (cs.SD, cs.LG), UT Austin coursework 기반 연구|
|논문 링크|[arXiv:2505.09633](https://arxiv.org/abs/2505.09633)|
|코드|[GitHub Repository](https://github.com/nicksunday/deepfake-music-detector)|
|주요 데이터셋|FakeMusicCaps[1], MusicLM 관련 MusicCaps[2]|
|주요 관련 연구|SONICS[3], "Detecting music deepfakes is easy but actually hard"[4]|

**재현 참고사항**: GitHub 저장소에는 Jupyter notebook 형태의 실험 코드가 포함되어 있으며, 
본 재현 구현은 논문에 기술된 아키텍처와 하이퍼파라미터를 기반으로 PyTorch로 재구현한 것입니다.

---

# 1. 연구 배경

## 1.1 Text-to-Music와 음악 딥페이크의 등장

논문에 따르면 최근 Text to Music 플랫폼의 발전으로 짧은 텍스트 프롬프트만으로도 사람 연주와 구분하기 어려운 수준의 음악을 생성할 수 있는 환경이 빠르게 확산되고 있습니다. Sunday는 이 흐름을 이미지 딥페이크와 음성 딥페이크가 만들어낸 상황과 유사한 맥락에서 이해합니다. 플랫폼은 창작 접근성을 높이고 새로운 실험을 가능하게 하지만, 동시에 다음과 같은 문제를 동반합니다.

- 저작권 침해
- 거짓 저자 표기와 크레딧 왜곡
- 예술적 진정성에 대한 신뢰 저하

저자는 이러한 문제를 완전히 새로운 영역이라기보다, 기존 딥페이크 논의가 음악 영역으로 확장된 사례로 봅니다. 특히 "누가 이 음악을 만들었는지"와 "얼마나 사람 창작에 의존했는지"를 둘러싼 법적, 윤리적 논의가 본격화되면서, 플랫폼 차원에서 사람 음악과 생성 음악을 구분해 주는 기술이 정책과 운영 모두에서 중요해지고 있다는 점을 강조합니다.

## 1.2 FakeMusicCaps와 MusicCaps

이 실험은 Politecnico di Milano 연구진이 제안한 FakeMusicCaps 데이터셋 [1]에 기반합니다. FakeMusicCaps는 Google의 MusicCaps [2] 에서 제공하는 텍스트 설명을 바탕으로 여러 Text to Music 모델이 생성한 딥페이크 음악을 모아 구축된 데이터셋입니다.

저자가 정리한 FakeMusicCaps의 구조는 다음과 같습니다.

- 사람 음악
    - MusicCaps 기반 사람 연주 10초 오디오 클립 5,373개
- 딥페이크 음악
    - 다섯개 TTM 플랫폼(MusicGen, audioldm2, musicdm, mustango, stable_audio_open)이 생성한 10초 딥페이크 트랙 5,521개
    - Suno 플랫폼에서 생성된 더 긴 딥페이크 트랙 63개

이 가운데 저자는 딥페이크 5,373개를 무작위로 선택해 사람 음악 5,373개와 균형을 맞춘 뒤, 총 10,746개 샘플을 "사람 vs 딥페이크"이진 분류 데이터셋으로 재구성합니다. 이때 개별 TTM 플랫폼을 구분하지 않고, 모든 플랫폼을 하나의 "딥페이크" 클래스로 합쳐 버리는 설계 선택을 합니다. 이 선택 덕분에 모델 구조와 실험 설계가 단순해지는 대신, 플랫폼별 특성과 차이를 분석하는 가능성은 일부 포기하는 형태가 됩니다.

## 1.3 SONICS, Deezer 연구와 Sunday 논문의 위치

Sunday는 관련 연구로 SONICS [3]와 Deezer 연구(Afchar et al., 2024) [4]를 핵심 축으로 인용합니다.

- SONICS
    - SONICS는 Syntetic Or Not, Identifying Counterfeit Songs라는 이름 그대로 전체 곡 단위 딥페이크를 대상으로 SpecTTTra라는 새로운 아키텍처와 풀 길이 데이터셋을 제안합니다.
    - 이 연구는 짧은 클립 기반 데이터셋이 곡 구조, 가사 편곡 같은 문맥 정보를 충분히 담지 못한다고 지적하고, 풀 곡 수준에서의 구조적 다양성과 맥락 모델링을 강조합니다.

- Deezer 연구(Afchar et al., 2024)
    - Deezer연구는 Sunday가 인용한 바에 따르면, 사람 음악과 딥페이크 음악을 CNN으로 분류하는 작업은 기본 설정에서는 비교적 어렵지 않지만, 피치 쉬프트와 템포 스트레치 같은 단순 조작만으로도 탁월한 모델이 쉽게 무력화될 수 있다는 점을 보여줍니다.
    - 이 연구는 "탐지 자체가 중요한 것이 아니라 조작에 대한 견고함이 핵심 문제"라는 관점을 제시합니다.

Sunday의 Detecting Musical Deepfakes는 이 두 축 사이 어딘가에 위치합니다.

- SONICS처럼 전체 곡 단위 복잡한 아키텍처로 가지 않고
- Deezer 연구에서 중요하게 다룬 피치와 템포 조작 시나리오를 그대로 가져와
- FakeMusicCaps와 ResNet18이라는 비교적 단순한 조합이 조작에 얼마나 취약한지 다시 점검합니다.

이 관점에서 보면 Sunday 논문은, "단순한 Mel Spectrogram과 범용 CNN만으로 어느 정도까지 버틸 수 있는가"를 살펴보는 중간 단계의 정리 작업으로 이해할 수 있습니다.

## 1.4 법, 윤리, 정책 논의의 개요

논문 후반부에서 Sunday는 Text to Music 플랫폼이 만들어낼 수 있는 법적, 윤리적 이슈를 개관합니다. 논문에 따르면 대표적으로 다음과 같은 경우들이 문제로 제기됩니다.

- 기존 곡을 모사하거나, 특정 아티스트의 스타일을 과도하게 모방하는 생성물로 인한 저작권 침해 가능성
- AI가 만든 곡을 사람의 창작물인 것처럼 크레딧을 붙이고 판매하거나 공개하는 경우 발생할 수 있는 사기와 신뢰 훼손 문제
- 아티스트의 음성을 무단으로 학습해 목소리 자체를 복제하는 경우 퍼블리시티 권리 침해 가능성

동시에 Sunday는 모든 딥페이크가 악용으로 이어지는 것은 아니라는 점도 사례를 통해 보여 줍니다. 질병이나 사고로 목소리를 잃은 사람이 AI 기반 음성 복원을 통해 다시 노래를 부를 수 있게 되는 사례, 사망한 아티스트의 미완성 작업을 유족과 레이블이 함께 AI를 통해 마무리하는 사례 등은 기술의 긍정적 가능성을 보여 주는 예로 소개됩니다.

이 논문은 명시적인 법률 해석을 제시하기보다는, 이러한 사례들이 "누가 이 음악을 만들었는지"와 "어떤 맥락에서 허용 가능한지"에 대한 사회적 합의와 규범을 요구한다는 점을 강조합니다. 그리고 이런 논의의 한 축으로서 "탐지 기술"이 필요하다는 문제의식을 전면에 두고 있습니다.

## 1.5 이 리뷰의 관점

이 블로그의 관점에서 Sunday 논문은 다음과 같은 이유로 첫 리뷰 대상으로 선택되었습니다.

1. FakeMusicCaps, Mel Spectrogram, ResNet18이라는 조합이 비교적 직관적이고, 코드가 공개되어 있어 재현과 변형이 용이합니다.
2. Afchar 등 Deezer 연구와 SONICS, FakeMusicCaps 논문을 자연스럽게 엮으면서, Text to Music 딥페이크 탐지 연구의 흐름 속에서 자신을 위치시키고 있습니다.
3. 짧은 10초 클립이라는 제약과 피치, 템포 조작에 대한 취약성을 동시에 보여 주기 때문에, 이후 더 복잡한 모델과 방어 기법을 다룰 때 "어디까지가 단순 모델로 가능한 영역인지"를 감각적으로 이해하는 데 도움이 됩니다.

이 리뷰는 논문이 제시하는 기술 구성과 실험 결과, 법과 윤리 논의를 가능한 한 충실히 정리하고, 향후 AI 음악 탐지 및 방어 논문 리뷰로 확장해 가기 위한 기반을 마련하는 것을 목표로 합니다.

---

# 2. 방법론 분석

## 2.1 문제 정의와 수식

저자가 정의하는 문제는 "10초 길이의 음악 오디오 클립이 사람 음악인지, Text to Music 기반 딥페이크인지 이진 분류하는 것"입니다.

- 입력
    - 길이 10초의 음악 오디오 클립
    - FakeMusicCaps 기반 Mel Spectrogram 이미지
- 출력
    - 레이블 0: 사람 연주 음악
    - 레이블 1: TTM 기반 딥페이크 음악

 Mel Spectrogram으로 표현된 입력을 다음과 같이 두면

$$
\mathbf{x} \in \mathbb{R}^{C \times H \times W}
$$

여기서

- (C)는 채널 수
- (H), (W)는 주파수 축과 시간 축 해상도입니다.

레이블은
$$
y \in \{0, 1\}
$$
으로 정의됩니다.

- (y = 0): 사람 음악
- (y = 1): 딥페이크 음악

ResNet18 기반 분류기 $f_{\theta}$는

$$
f_{\theta}: \mathbb{R}^{C \times H \times W} \rightarrow [0, 1]^2
$$

형태의 확률 벡터를 출력합니다. 학습 목표는 교차 엔트로피 손실을 최소화하는 것입니다.

$$
\min_{theta} \ \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}}
\left[ \mathcal{L}_{\mathrm{CE}}\big(f_{\theta}(\mathbf{x}), y\big) \right]
$$
여기서
- $\mathcal{D}$ 는 Sunday가 재구성한 FakeMusicCaps 이진 분류 데이터셋
- $\mathcal{L}_{\mathrm{CE}}$ 는 이진 분류용 교차 엔트로피 손실입니다.

직관적으로는 "Mel Spectrogram 이미지를 입력받아 사람 음악인지 딥페이크인지 구분하는 이미지 분류 모델을 학습하는 문제"로 정리할 수 있습니다.

## 2.1.1 표기 정리

이 리뷰에서 사용하는 표기와 의미는 다음과 같습니다.

|기호|의미|
|---|---|
| $\mathbf{x}$                | Mel Spectrogram으로 표현된 입력 오디오 텐서 |
| $y$                         | 이진 레이블(0: 사람, 1: 딥페이크)          |
| $f_{\theta}$                | ResNet18 기반 분류기                 |
| $\theta$                    | 모델 파라미터                         |
| $\mathcal{D}$               | FakeMusicCaps 기반 학습 데이터셋        |
| $\mathcal{L}_{\mathrm{CE}}$ | 교차 엔트로피 손실 함수                   |

## 2.2 데이터셋 구성과 전처리

논문에 따르면 데이터셋 구성은 다음과 같이 이루어집니다.

- 전체 샘플 수: 10,746개
    - 사람 음악: 5,373개
    - 딥페이크 음악: 5,373개

데이터 분할은 텐서 개수를 기준으로

- 학습: 8,599개
- 검증: 1,075개
- 테스트: 1,074

로 이루어집니다.

모든 오디오는 librosa를 활용해 Mel Spectrogram으로 변환됩니다. 이후 PyTorch 텐서로 변환하고, ImageNet 사전학습 ResNet18에 맞추어 다음 평균과 표준편차로 정규화합니다.

- 평균: [0.485, 0.456, 0.406]
- 표준편차: [0.229, 0.224, 0.225]

논문에 따르면 Mel Spectrogram을 텐서로 변환한 뒤 ImageNet 정규화(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])를 적용합니다. ResNet18이 3채널 입력을 요구하므로 구현 시 채널 변환이 필요하지만, 논문에서는 구체적인 방식을 명시하지 않았습니다.

## 2.3 모델 구조와 학습 설정

모델 구조는 torchvision의 ResNet18을 그대로 사용하되, 마지막 완전연결층을 이진 분류에 맞게 수정하는 방식입니다.

- Backbond: ResNet18 (ImageNet1k v1 사전학습 가중치)
- 수정: 마지막 fc 레이어의 출력 차원을 2로 교체
- 손실 함수: Cross Entropy Loss
- 옵티마이저: Adam
- 배치 크기: 32
- 학습 epoch 수: 20

학습 자체는 전형적인 이미지 분류 설정과 거의 동일합니다. 이러한 선택은

1. Audio 전용 아키텍처를 도입하기 전에, "가장 단순한 범용 CNN"으로 어느 정도까지 성능이 나오는지 확인하기 위한 의도
2. 아키텍처를 바꾸지 않고도 데이터셋과 조작 시나리오를 비교하기 위한 기준 구조 확보

라는 두 가지 관점에서 이해할 수 있습니다.

## 2.4 악의적 조작 시나리오와 데이터셋 변형

논문에 따르면, 이 연구는 Deezer연구 [4]에서 제안한 피치와 템포 조작 아이디어를 FakeMusicCaps에 적용해 "조작된 딥페이크가 탐지 모델을 얼마나 쉽게 속일 수 있는가"를 평가합니다.

논문에서 정의한 네 가지 데이터셋은 다음과 같습니다.

1. Base
    - 조작을 가하지 않은 원본 FakeMusicCaps

2. Pitch
    - 각 클립에 대해 -2에서 2사이의 난수를 반음 단위로 샘플링
    - 해당 값만큼 피치 쉬프트 적용

3. Tempo
    - 각 클립에 대해 0.8에서 1.2 사이의 난수를 샘플링
    - 해당 비율만큼 재생 속도를 늘리거나 줄이는 템포 스트레치 적용

4. PitchTempo
    - 피치 쉬프트와 템포 스트레치를 모두 적용

이때 피치와 템포 변조는 모두 librosa의 시그널 처리 함수를 사용해 구현합니다. 각 클립마다 독립적인 난수를 샘플링하므로, 데이터셋 전체적으로 다양한 조작 조합이 등장하게 됩니다.

## 2.5 평가지표 정의

저자는 각 실험 설정에 대해 Accuracy, Precision, Recall, F1 Score, False Positive Rate (FPR), False Negative Rate (FNR)를 사용해 모델 성능을 평가합니다.

**Label 정의**: 논문과 GitHub 저장소 설명을 통해 Class 0 = Human (사람 음악), Class 1 = Deepfake (AI 생성 음악)으로 정의됨을 확인했습니다.

혼동 행렬을

- TP: 딥페이크를 딥페이크로 올바르게 분류
- TN: 사람 음악을 사람 음악으로 올바르게 분류
- FP: 사람 음악을 딥페이크로 잘못 분류
- FN: 딥페이크를 사람 음악으로 잘못 분류

라고 할 때, 지표는 다음과 같이 정의됩니다.

$$
\mathrm{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$
$$
\mathrm{Precision} = \frac{TP}{TP + FP}
$$
$$
\mathrm{Recall} = \frac{TP}{TP + FN}
$$
$$
\mathrm{F1} =
2 \cdot \frac{\mathrm{Precision} \cdot \mathrm{Recall}}
{\mathrm{Precision} + \mathrm{Recall}}
$$


$$
\mathrm{FPR} = \frac{FP}{FP + TN}
$$
$$
\mathrm{FNR} = \frac{FN}{FN + TP}
$$

Sunday는 이러한 지표를 네 가지 데이터셋(Base, Pitch, Tempo, PitchTempo)에 각각 학습한 모델과, 네 데이터셋을 순차적으로 학습한 Continuous Learning 모델에 대해 비교합니다. 논문 본문에는 Recall 88.89퍼센트, FNR 11.11퍼센트처럼 일부 수치가 직접 언급되며, 나머지는 "모든 지표가 80퍼센트 이상" 또는 "Baseline 대비 3-4 퍼센트포인트 이내" 같은 형태로 정성적 설명이 제공됩니다. PDF 표에 포함된 상세 지표는 텍스트로 일일이 풀어 쓰여 있지 않기 때문에, 이 리뷰에서는 저자가 직접 언급한 수치를 중심으로 정리합니다.

---

# 3. 재현 실험 설계

이 섹션은 Sunday의 GitHub 레포지토리를 기반으로 한 사용자 재현 실험 계획을 정리한 부분입니다. 실제 수치와 그래프는 추후 별도의 포스트에서 다룰 예정이며, 여기서는 "어떤 구조로 재현할 것인지"만 기록합니다.

## 3.1 재현 목표

1. Baseline 재현
- FakeMusicCaps에서 사람과 딥페이크 각각 5,373개를 추출
- Mel Spectrogram과 ImageNet 정규화 적용
- ResNet18을 epoch 학습해 Base 데이터셋 기준 Sunday 논문과 유사한 지표 달성 여부 확인

2. 조작 시나리오 재현
- Pitch, Tempo, PitchTempo 세 가지 조작 데이터셋 생성
- 각 데이터셋에 대해 독립적으로 ResNet18 학습
- Base 모델 대비 성능 하락 폭과 Deezer 연구에서 보고된 경향 비교

3. Continuous Learning 재현
- Base -> Pitch -> Tempo -> PitchTempo 순으로 동일 모델을 순차 학습
- Recall과 FNR, FPR의 변화 추이를 Sunday 논문과 비교

4. 확장 실험 초안
- 동일 파이프라인에서 ResNet50, Audio Transformer 등 아키텍처 변경 실험 설계
- 공격 강도(피치 변조 폭, 템포 비율)의 단계별 변화에 따른 성능 곡선 설계

## 3.2 환경 설정 메모

아래 표는 원 논문 설정과 사용자 재현 환경을 대응시키기 위한 메모용입니다. 사용자 실험이 진행되면 오른쪽 열에 구체적인 내용을 채워 넣을 계획입니다.

| 항목 | 원 논문 설정 | 사용자 재현 계획 |
|-----|------------|--------------|
|프로그래밍 언어|Python, PyTorch, librosa|동일 스택 사용|
|입력 표현|Mel Spectrogram + ImageNet 정규화|동일 설정 재현|
|모델|ResNet18(ImageNet1k v1), 출력 2차원|동일 모델로 시작 후 다른 아키텍처 확장|
|데이터 분할|학습 8,599, 검증 1,075, 테스트 1,074|가능한 한 동일 분할 규칙 적용|
|하이퍼파라미터|배치 32, epoch 20, Adam, Cross Entropy|초기에는 동일 설정, 이후 epoch와 학습률 조정 실험|

---

# 4. 비판적 분석 

## 4.1 장점

1. 직관적이고 재현 가능한 베이스라인
    Sunday 논문과 GitHub 코드에 따르면 FakeMusicCaps, Mel Spectrogram, ResNet18이라는 조합은 구조가 단순하고 구현이 명확해, 추후 연구자와 실무자가 같은 설정에서 출발하기에 적합합니다. 복잡한 아키텍처를 도입하기 전에 "TTM 딥페이크를 어느 정도까지 기본 CNN으로 걸러낼 수 있는지"를 빠르게 확인할 수 있는 출발점이라는 점이 장점입니다.

2. 조작에 대한 취약성 문제를 정면에서 다룸
    Deezer 연구에 따르면 피치와 템포 조작이 탐지 모델을 크게 약화사시킬 수 있다는 점이 반복해서 강조됩니다. Sunday는 이 시나리오를 FakeMusicCaps와 ResNet18 설정에 그대로 적용해, 단일 모델에서 조작 전후 성능을 비교합니다. Baseline 대비 Pitch, Tempo, PitchTempo 설정에서 3-4포인트 수준의 성능 저하가 일관되게 나타난다는 점은, "테스트셋 성능만 보고 모델을 신뢰하는 것은 위험하다"는 메시지를 다시 상기시킵니다. 

3. Continuous Learning을 통한 운영 상 트레이드오프 제시
    논문 Table 1에 따르면 Continuous Learning 실험은 Recall 0.889로 딥페이크 탐지율이 높지만, FPR 0.217로 사람 음악의 21.7%를 딥페이크로 오인합니다. 이는 반복 학습으로 모델이 과도하게 민감해져, 정상 음악까지 의심하는 경향을 보이는 것입니다. Precision 0.800, Specificity 0.783으로 전체적인 정확도는 감소했지만, 딥페이크를 놓치는 것보다 오탐을 감수하는 보수적인 접근으로 볼 수 있습니다. 실제 서비스에서는 이러한 트레이드오프를 고려하여 임계값을 조정해야 합니다.

# 4.2 한계와 주의점

1. 10초 클립 중심 설계의 한계
    FakeMusicCaps는 기본적으로 길이 10초 클립으로 구성되어 있습니다. 논문에 따르면 이 구조는 학습과 배치 처리에는 편리하지만, 실제 음악 소비 환경에서는 곡 전체 구조, 가사, 편곡, 믹싱 등 더 긴 시간 스케일의 정보가 중요합니다. SONICS가 강조하듯, 전체 곡 단위 모델링이 필요한 상황에서는 Sunday의 설정만으로는 충분하지 않습니다.

2. ResNet18 단일 아키텍처에 대한 의존
    Sunday는 오디오 특화 네트워크나 Transformer 기반 모델과의 비교 없이, ResNet18 하나만을 사용합니다. Sunday가 제시한 결과만으로는 "Mel Spectrogram과 CNN이라는 선택이 이 문제에 최적화된 것인지"를 판단하기 어렵습니다. RawNet2나 SpecTTTra, 오디오 Transformer 같은 모델과의 비교 실험이 없다 보니, 모델 구조가 바뀌었을 때 조작에 대한 취약성이 어떻게 달라지는지는 추가 연구가 필요합니다.

3. 데이터셋 외삽 일반화에 대한 논의 부족
    FakeMusicCaps는 MusicCaps 기반으로 구성되어 있어, 장르와 스타일, 녹음 환경에 특정 편향이 존재할 가능성이 있습니다. 논문은 이러한 데이터셋 바깥 상황(로켈 인디 음악, 라이브 녹음, 방송 음원 등)에 대한 실험을 수행하지 않기 때문에, 모델이 현실 세계 전체 음악 분포에서 얼마나 잘 일반화되는지에 대해서는 조심스럽게 해석해야 합니다.

4. 조작 강도의 범위가 비교적 제한적
    Sunday는 Deezer 연구에서 사용한 것과 유사한 범위의 피치 템포와 조작을 사용합니다.(피치 ±2세미톤, 템포 0.8-1.2배). 논문 결론에서 저자 스스로 더 극단적인 조작, 다른 종류의 오디오 효과(리버브, 에코, 디스토션 등)를 고려할 필요가 있다고 언급합니다. 따라서 현재 결과는 "적당한 수준의 피치와 템포 조작에 대한 강건성"만을 평가한 것에 가깝고, 보다 공격적인 공격 시나리오에 대한 평가는 후속 과제로 남습니다.

5. Image 전이학습의 근본적 한계
    Mel Spectrogram의 주파수(Y축)와 시간(X축)은 일반 이미지의 공간 정보와 의미가 다릅니다. ImageNet으로 학습된 특징이 오디오 스펙트로그램에 최적화되지 않았을 가능성이 있으며, 이는 피치/템포 조작에 취약한 원인일 수 있습니다.

## 4.3 음악 산업 적용 가능성

1. 업로드 단계 선별 필터의 초기 버전으로 활용 가능
    Sunday 결과에 따르면 Baseline 설정에서 F1 0.878, Accuracy 0.885, Precision 0.911, Specificity 0.922 정도의 성능을 얻을 수 있습니다. 이는 "업로드 단계에서 의심스러운 딥페이크를 후보로 올리는 필터"로는 충분히 의미 있는 수준입니다. 다만 피치와 템포 조작에 대한 취약성, Continuous Learning에서 FPR 증가를 고려하면, 이 모델 하나만으로 저작권 판단이나 강한 제재를 자동화하는 것은 적절하지 않아 보입니다.

2. 강건성 평가 템플릿으로서의 가치
    네 가지 데이터셋(Base, Pitch, Tempo, PitchTempo)과 Continuous Learning 구성을 나란히 비교하는 구조는 다른 플랫폼에도 그대로 적용할 수 있는 평가 템플릿입니다. 예를 들어 실제 서비스에서는 여기에 이퀄라이저 조작, 잡음 추가, 동영상 플랫폼 리인코딩 등 현실적인 변형까지 확장해 실험할 수 있습니다. Sunday의 결과는 그러한 확장 실험을 설계할 때 유용한 출발점이 됩니다.

3. 비기술 이해관계자 대상 설명 자료
    ResNet18과 Mel Spectrogram은 이미지 분류 비유로 설명하기 쉽기 때문에, 규제 기관이나 저작권 단체, 레이블 실무자에게 "기계가 생성 음악을 어떻게 보고 있는지"를 설명하는 데 도움이 됩니다. "조작 전에는 80퍼센트 이상 잘 맞추지만, 피치와 템포를 약간만 바꾸면 성능이 떨어진다"는 메시지는 딥페이크 탐지 기술의 잠재력과 한계를 동시에 전달할 수 있습니다.

---

# 5. 결론 및 향후 연구

Sunday의 「Detecting Musical Deepfakes」는 FakeMusicCaps와 ResNet18이라는 단순하고 재현 가능한 설정에서, 사람 음악과 TTM 기반 딥페이크 음악을 구분하는 기본 모델을 구축하고, 피치와 템포 조작이 탐지 성능에 어떤 영향을 미치는지 정리한 연구입니다. 논문에 따르면 모든 실험에서 F1, Accuracy, Recall, Precision이 80퍼센트 이상을 기록했다는 점은 긍정적이지만, Deezer 연구와 마찬가지로 피치와 템포 조작에는 여전히 취약한 모습을 보입니다.

이 리뷰의 관점에서 보면 Sunday 논문은 AI Music Detection 축에서 다음과 같은 의미를 갖습니다.

- ResNet18과 Mel Spectrogram이라는 기본 조합으로 어느 정도 성능을 기대할 수 있는지에 대한 "실용적인 기준선"을 제공합니다.

- 단순한 조작 시나리오만으로도 탐지 모델이 쉽게 흔들릴 수 있다는 점을 다시 보여 주며, 강건성 평가의 중요성을 상기시킵니다.

- SONICS 수준의 전체 곡 모델링, Unlearnable Data나 Defensive Data Poisoning 같은 방어 기법을 검토할 때 "어디까지는 단순 모델로 해결되고, 어디서부터는 데이터 측 방어 혹은 더 정교한 모델이 필요한가"를 생각해 볼 수 있는 출발점을 제공합니다.

---
아래 번호는 리뷰 논문의 참고문헌과 정합성을 최대한 맞추어 정리했습니다.

[1] G. Comanducci et al. "FakeMusicCaps: A Benchmark for Detecting AI Generated Music from Text Descriptions." arXiv, 2024.  
[2] A. Agostinelli et al. "MusicLM: Generating Music From Text." arXiv, 2023.  
[3] M. M. Rahman et al. "SONICS: Synthetic Or Not? Identifying Counterfeit Songs." arXiv, 2024.  
[4] D. Afchar et al. "Detecting music deepfakes is easy but actually hard." arXiv, 2024.  
[5] N. Sunday. "Detecting Musical Deepfakes." arXiv:2505.09633, 2025. (UT Austin coursework)
[6] nicksunday. "Deepfake Music Detection." GitHub Repository, 2025. [https://github.com/nicksunday/deepfake-music-detector](https://github.com/nicksunday/deepfake-music-detector?utm_source=chatgpt.com)

