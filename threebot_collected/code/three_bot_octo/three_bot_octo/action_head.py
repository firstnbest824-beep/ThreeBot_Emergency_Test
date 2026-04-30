"""Three-Bot Action Head: 5-DoF (Dual XY + Shared Z).
# Three-Bot의 액션 헤드 모듈: 5자유도 (XY 2쌍 + 공통 Z) 제어를 담당합니다.

Refactored for 5-DoF Hardware:
# 5-DoF 하드웨어 구성에 맞게 재설계되었습니다:
- Unit 1: X1, Y1 (왼쪽)
- Unit 2: X2, Y2 (오른쪽)
- Shared: Z (양쪽 통합 윈치)

Architecture:
# 아키텍처 특징:
- XY (4-DoF): Deterministic MLP (결정론적 다층 퍼셉트론)
- Z (1-DoF): Gaussian Head (Uncertainty-aware, low latency)
# Z축은 Gaussian Head를 사용하여 확률적인 예측(불확실성 관리)을 수행합니다. 
# 윈치 제어의 안정성을 위해 중요합니다.

Author: Architect_Sonnet @ Nova Robotics
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Tuple
from .gaussian_action_head import GaussianActionHead
# 별도로 정의된 GaussianActionHead(확률적 액션 헤드)를 임포트합니다.

class ThreeBotActionHead(nn.Module):
    """5-DoF Action Head for Dual Winch System.
    # 듀얼 윈치 시스템을 위한 5-DoF 액션 헤드 클래스입니다.
    # Flax(JAX)의 nn.Module을 상속받습니다.
    """
    
    hidden_dim: int = 256
    # 은닉층(Hidden Layer)의 차원 크기를 256으로 설정합니다.
    
    def setup(self):
        # Shared Feature Extractor
        # 공통 특징 추출기: 트랜스포머의 출력을 받아서 액션에 필요한 특징으로 변환합니다.
        self.feature_extractor = nn.Sequential([
            nn.Dense(self.hidden_dim, name='fc1'),  # 완전 연결 층 1
            nn.LayerNorm(name='ln1'),               # 레이어 정규화
            nn.gelu,                                # GELU 활성화 함수
            nn.Dense(self.hidden_dim, name='fc2'),  # 완전 연결 층 2
            nn.LayerNorm(name='ln2'),               # 레이어 정규화
            nn.gelu,                                # GELU 활성화 함수
        ])
        
        # XY Heads (Deterministic)
        # XY 축 제어 헤드 (결정론적 출력)
        # 출력: 4개 [ΔX1, ΔY1, ΔX2, ΔY2]
        self.xy_head = nn.Dense(4, name='xy_head')
        
        # Z Head (Gaussian / Probabilistic)
        # Z 축 제어 헤드 (확률적 출력 - 가우시안 분포)
        # 출력: 1개 [ΔZ_shared]
        # Z축은 중력 등의 영향으로 미세 조정이 중요하므로 확률적 접근을 사용합니다.
        self.z_head = GaussianActionHead(action_dim=1)
        
    @nn.compact
    def __call__(self, 
                 transformer_output: jnp.ndarray, 
                 train: bool = False) -> jnp.ndarray:
        """
        Forward Pass (순전파 수행)
        
        Args:
            transformer_output: [B, Seq, D] 형태의 텐서. (Seq = 시퀀스 길이)
                                트랜스포머 모델의 전체 출력입니다.
            train: 학습 모드인지 여부 (Dropout이나 샘플링 방식이 달라짐)
            
        Returns:
            actions: [B, 5] (배치 크기, 5) 형태의 액션 값.
                     순서대로 [X1, Y1, X2, Y2, Z] 입니다.
        """
        # Mean pooling over sequence dimension
        # 시퀀스 차원(Sequence dim)에 대해 평균을 내서 하나의 벡터로 압축합니다.
        # [B, Seq, D] -> [B, D]
        h = jnp.mean(transformer_output, axis=1)
        
        # 특징 추출기 통과
        feat = self.feature_extractor(h)  # [B, 256]
        
        # 1. XY Prediction (XY 축 예측)
        xy = self.xy_head(feat)  # [B, 4]
        xy = nn.tanh(xy) # 출력 범위를 [-1, 1] 사이로 제한합니다 (Tanh 사용).
        
        # 2. Z Prediction (Z 축 예측)
        # Gaussian Head를 통해 Z축 값을 예측합니다.
        # 학습 중일 때는 분포에서 샘플링하고, 추론 시에는 평균값을 쓸 수도 있습니다.
        z_out = self.z_head(feat, train=train)
        
        if train:
            # 학습 모드일 때는 z_out이 (샘플, 로그확률 등)의 튜플일 수 있으므로 첫 번째 요소(샘플)만 가져옵니다.
            z_sample = z_out[0] 
        else:
            # 추론 모드일 때는 z_out 자체가 액션 값입니다.
            z_sample = z_out
            
        # Combine [B, 4] + [B, 1] -> [B, 5]
        # XY 값 4개와 Z 값 1개를 합쳐서 최종 5-DoF 액션을 만듭니다.
        actions = jnp.concatenate([xy, z_sample], axis=-1)
        
        return actions

    def compute_loss(self,
                     transformer_output: jnp.ndarray,
                     ground_truth_actions: jnp.ndarray, # 정답 액션 데이터 [B, 5]
                     key: jax.random.PRNGKey) -> Tuple[jnp.ndarray, dict]:
        """
        Loss Calculation (손실 함수 계산 - 학습용)
        """
        
        # 학습 시에는 보통 마지막 토큰(가장 최근 정보)만 사용하거나 전체를 평균냅니다.
        # 여기서는 마지막 토큰(last token)을 사용하는 것으로 보입니다.
        h = transformer_output[:, -1, :] 
        feat = self.feature_extractor(h)
        
        # 1. XY Loss (MSE - 평균 제곱 오차)
        # XY축은 정답과의 단순 거리 차이를 줄이도록 학습합니다.
        xy_pred = nn.tanh(self.xy_head(feat))
        xy_true = ground_truth_actions[:, :4] # 처음 4개 컬럼이 XY 정답
        xy_loss = jnp.mean((xy_pred - xy_true) ** 2)
        
        # 2. Z Loss (NLL - 음의 로그 우도)
        # Z축은 가우시안 분포의 확률 밀도를 최대화하도록(NLL을 최소화하도록) 학습합니다.
        # 이는 모델이 자신의 예측 불확실성을 함께 배우게 합니다.
        z_true = ground_truth_actions[:, 4:5] # 5번째 컬럼이 Z 정답
        z_loss, z_metrics = self.z_head.compute_loss(feat, z_true, key)
        
        # Total Loss (총 손실)
        # 두 손실을 단순 합산합니다. (필요 시 가중치를 줄 수 있습니다)
        total_loss = xy_loss + z_loss
        
        # 모니터링을 위한 메트릭 딕셔너리 생성
        metrics = {
            'total_loss': total_loss,
            'xy_loss': xy_loss,
            **z_metrics # Z축 관련 메트릭(분산 등) 포함
        }
        
        return total_loss, metrics
