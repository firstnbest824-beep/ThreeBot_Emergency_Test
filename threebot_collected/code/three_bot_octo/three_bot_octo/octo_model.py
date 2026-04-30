"""Complete Three-Bot Octo Model (5-DoF Refactor).
# 전체 Three-Bot Octo 모델: 5-DoF 하드웨어에 맞게 재구성되었습니다.

Integrates (구성 요소):
- Vision Encoder: 3-camera multi-view (3대의 카메라 통합)
- Proprioception Encoder: 10-dim state (5 pos + 5 vel) (로봇 상태 정보)
- Action Head: 5-DoF (Dual XY + Shared Z) (최종 액션 결정)

Author: Architect_Sonnet @ Nova Robotics
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Optional, Tuple
from .vision_encoder import ThreeBotVisionEncoder
from .action_head import ThreeBotActionHead

class ProprioceptionEncoder(nn.Module):
    """Encodes 5-DoF robot state into 3 semantic tokens.
    # 5-DoF 로봇 상태(10차원)를 3개의 의미 있는 토큰으로 인코딩합니다.
    
    Output Structure (출력 구조):
        Token 0: Unit 1 (X1, Y1 + velocities) - 왼쪽 유닛 정보
        Token 1: Unit 2 (X2, Y2 + velocities) - 오른쪽 유닛 정보
        Token 2: Shared Z-axis (Z + velocity) - 공통 Z축 정보
    """
    
    out_dim: int = 128
    # 출력될 토큰의 차원 크기
    
    def setup(self):
        # XY 유닛을 위한 공통 인코더 (좌우 대칭성을 위해 가중치 공유)
        self.xy_unit_encoder = nn.Sequential([
            nn.Dense(64, name='xy_fc1'),
            nn.LayerNorm(name='xy_ln1'),
            nn.gelu,
            nn.Dense(self.out_dim, name='xy_fc2'),
        ])
        
        # 공통 Z축을 위한 별도의 인코더
        self.z_encoder = nn.Sequential([
            nn.Dense(64, name='z_fc1'),
            nn.LayerNorm(name='z_ln1'),
            nn.gelu,
            nn.Dense(self.out_dim, name='z_fc2'),
        ])
    
    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            state: [B, 10] = [x1, y1, x2, y2, z, vx1, vy1, vx2, vy2, vz]
        Returns:
            tokens: [B, 3, D] where D=out_dim (배치 크기, 토큰 3개, 차원 D)
        """
        # 상태 데이터를 위치(pos)와 속도(vel)로 분리
        pos = state[:, :5]   # [B, 5] = [x1, y1, x2, y2, z]
        vel = state[:, 5:]   # [B, 5] = [vx1, vy1, vx2, vy2, vz]
        
        # 1. 왼쪽 유닛 인코딩 (X1, Y1 + 속도)
        unit1_state = jnp.concatenate([
            pos[:, 0:2],   # 위치 정보
            vel[:, 0:2]    # 속도 정보
        ], axis=-1)  # [B, 4]
        unit1_token = self.xy_unit_encoder(unit1_state)  # [B, D]
        
        # 2. 오른쪽 유닛 인코딩 (X2, Y2 + 속도) - 인코더 재사용(Weight Sharing)
        unit2_state = jnp.concatenate([
            pos[:, 2:4],   
            vel[:, 2:4]    
        ], axis=-1)  # [B, 4]
        unit2_token = self.xy_unit_encoder(unit2_state)  # [B, D]
        
        # 3. 공통 Z축 인코딩
        z_state = jnp.concatenate([
            pos[:, 4:5],   
            vel[:, 4:5]    
        ], axis=-1)  # [B, 2]
        z_token = self.z_encoder(z_state)  # [B, D]
        
        # 3개의 토큰을 하나로 합침 [B, 3, D]
        tokens = jnp.stack([unit1_token, unit2_token, z_token], axis=1)
        
        return tokens


class TransformerBlock(nn.Module):
    """Standard Transformer Block.
    # 표준적인 트랜스포머 블록: 데이터를 고차원적으로 처리하고 관계를 학습합니다.
    """
    num_heads: int = 8
    mlp_dim: int = 2048
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        # Multi-Head Attention: 여러 각도에서 데이터 간의 연관성을 분석합니다.
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate if train else 0.0
        )(x, x)
        x = nn.LayerNorm()(x + attn_out) # 잔차 연결(Residual Connection) 및 정규화
        
        # MLP (Feed Forward): 분석된 특징을 비선형적으로 변환합니다.
        mlp_out = nn.Sequential([
            nn.Dense(self.mlp_dim),
            nn.gelu,
            nn.Dropout(rate=self.dropout_rate, deterministic=not train),
            nn.Dense(x.shape[-1]),
            nn.Dropout(rate=self.dropout_rate, deterministic=not train),
        ])(x)
        out = nn.LayerNorm()(x + mlp_out)
        return out

class ThreeBotOctoModel(nn.Module):
    """5-DoF Octo Model with Proprioception.
    # 5-DoF Three-Bot 시스템의 전체 Octo 모델 아키텍처입니다.
    """
    
    vision_feature_dim: int = 512
    transformer_dim: int = 768
    num_transformer_layers: int = 6
    num_attention_heads: int = 8
    
    def setup(self):
        # 1. 인코더 레이어 (시각 + 로봇 상태)
        # 3대의 카메라 영상을 처리하는 비전 인코더
        self.vision_encoder = ThreeBotVisionEncoder(
            num_cameras=3,
            feature_dim=self.vision_feature_dim
        )
        # 비전 특징을 트랜스포머 차원에 맞추는 프로젝션 레이어
        self.vision_proj = nn.Dense(self.transformer_dim, name='vision_proj')
        
        # 로봇 상태 정보를 트랜스포머 토큰으로 변환하는 인코더
        self.proprio_encoder = ProprioceptionEncoder(out_dim=self.transformer_dim)
        
        # 2. 트랜스포머 레이어 (6층 구조)
        # 이미지와 로봇 상태 간의 복잡한 상관관계를 학습합니다.
        self.transformer_layers = [
            TransformerBlock(
                num_heads=self.num_attention_heads,
                name=f'transformer_layer_{i}'
            )
            for i in range(self.num_transformer_layers)
        ]
        
        # 3. 액션 헤드 (최종 예측부)
        # 트랜스포머의 출력을 받아 5-DoF 액션(ΔX1, ΔY1, ΔX2, ΔY2, ΔZ)을 결정합니다.
        self.action_head = ThreeBotActionHead(hidden_dim=256)
        
    def encode(self,
               observations: Dict[str, jnp.ndarray],
               proprioception: jnp.ndarray, # [B, 10]
               train: bool = True) -> jnp.ndarray:
        """Fuse Vision + Proprioception -> Transformer Tokens.
        # 시각 정보와 로봇 상태 정보를 하나로 융합하여 토큰화합니다.
        """
        
        # 1. 비전 토큰 생성: [B, T, N, D_vis]
        visual_tokens = self.vision_encoder(observations, train=train)
        B, T, N, C = visual_tokens.shape
        
        # 트랜스포머 차원으로 변환 및 1차원으로 펼치기
        visual_tokens = self.vision_proj(visual_tokens) # [B, T, N, D_trans]
        visual_tokens = visual_tokens.reshape(B, T * N, self.transformer_dim)
        # Shape: [B, T*N (전체 패치 수), D_trans]
        
        # 2. 로봇 상태 토큰 생성: [B, 10] -> [B, 3, D_trans]
        proprio_tokens = self.proprio_encoder(proprioception)
        # Shape: [B, 3 (왼쪽, 오른쪽, Z축), D_trans]
        
        # 3. 토큰 결합: [로봇 상태×3 + 시각 정보×N]
        # 모든 정보가 하나의 시퀀스로 묶여 트랜스포머에 입력됩니다.
        tokens = jnp.concatenate([proprio_tokens, visual_tokens], axis=1)
        # Shape: [B, 3 + T*N, D_trans]
        
        # 4. 트랜스포머 레이어 통과
        for layer in self.transformer_layers:
            tokens = layer(tokens, train=train)
            
        return tokens
    
    @nn.compact
    def __call__(self,
                 observations: Dict[str, jnp.ndarray],
                 proprioception: jnp.ndarray, # [B, 10]
                 train: bool = False) -> jnp.ndarray:
        """
        모델의 메인 실행 루프입니다.
        데이터를 인코딩하고 최종 액션(명령)을 예측합니다.
        """
        
        # 인코딩 수행 (시각 + 상태 융합 및 트랜스포머 연산)
        tokens = self.encode(observations, proprioception, train=train)
        
        # 액션 헤드를 통해 최종 명령 생성
        # Octo 모델 구조에 따라 트랜스포머의 전체 토큰 정보를 사용하여 액션을 예측합니다.
        actions = self.action_head(tokens, train=train)
        
        return actions

def create_three_bot_model() -> ThreeBotOctoModel:
    """Three-Bot Octo 모델 인스턴스를 생성하는 헬퍼 함수."""
    return ThreeBotOctoModel()
