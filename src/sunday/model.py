"""
Sunday 논문(2025) 재현을 위한 모델 구현.

ResNet18 기반 AI 음악 탐지 모델.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Tuple

class SundayDetector(nn.Module):
    """
    Sunday 논문의 딥페이크 음악 탐지 모델.

    Architecture:
        - Backbone: ResNet18 (ImageNet1k v1 pretrained)
        - Input: 3-channel Mel-spectrogram (224x224)
        - Output: Binary classification (Human=0, Deepfake=1)
    """

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        """
        모델 초기화.

        Args:
            pretrained: ImageNet pretrained weights 사용 여부
            freeze_backbone: ResNet18 backbone 가중치 고정 여부
        """
        super(SundayDetector, self).__init__()

        # ResNet18 backbone 로드
        self.backbone = models.resnet18(pretrained=pretrained)

        # Backbone 가중치 고정 (선택적)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # 분류 헤드 수정 (1000 classes -> 2 classes)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파.

        Args:
            x: 입력 텐서 (B, 3, 224, 224)

        Returns:
            로짓 (B, 2)
        """
        return self.backbone(x)

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        예측 (확률값 반환).

        Args:
            x: 입력 텐서 (B, 3, 224, 224)

        Returns:
            (예측 클래스, 확률값)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

        return preds, probs

    def get_feature_extractor(self) -> nn.Module:
        """
        특징 추출기 변환 (마지막 FC layer 제외).

        Returns:
            특징 추출 모델
        """
        # FC layer를 제외한 나머지 부분
        feature_extractor = nn.Sequential(*list(self.backbone.children())[:-1])
        return feature_extractor

class SundayLoss(nn.Module):
    """
    Sunday 논문의 손실 함수.

    Binary Cross Entropy with optional class weights.
    """

    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        """
        손실 함수 초기화.

        Args:
            class_weights: 클래스별 가중치 [human_weight, deepfake_weight]
        """
        super(SundayLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        손실 계산.

        Args:
            logits: 모델 출력 (B, 2)
            labels: 정답 레이블 (B,)

        Returns:
            손실값
        """
        return self.criterion(logits, labels)

def create_model(pretrained: bool = True, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> SundayDetector:
    """
    모델 생성 헬퍼 함수.

    Args:
        pretrained: ImageNet pretraiend 사용 여부
        device: 장치 (cuda/cpu)

    Returns:
        초기화된 모델
    """
    model = SundayDetector(pretrained=pretrained)
    model = model.to(device)
    return model

# 테스트 코드
if __name__ == "__main__":
    print("Sunday Model Test")
    print("=" * 50)

    # 1. 모델 생성 테스트
    model = create_model(pretrained=True, device='cpu')
    print(f"모델 생성 완료: {model.__class__.__name__}")

    # 2. 모델 구조 확인
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"전체 파라미터: {total_params:,}")
    print(f"학습 가능 파라미터: {trainable_params:,}")

    # 3. 순전파 테스트
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        output = model(dummy_input)
        print(f"\n입력 shape: {dummy_input.shape}")
        print(f"출력 shape: {output.shape}")

        # 예측 테스트
        preds, probs = model.predict(dummy_input)
        print(f"예측 클래스: {preds}")
        print(f"확률값 shape: {probs.shape}")

    # 4. 손실 함수 테스트
    loss_fn = SundayLoss()
    dummy_labels = torch.randint(0, 2, (batch_size,))
    loss = loss_fn(output, dummy_labels)
    print(f"\n손실값: {loss.item():.4f}")

    print("\n" + "=" * 50)
    print("All tests completed successfully!")

