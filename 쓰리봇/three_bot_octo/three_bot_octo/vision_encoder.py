"""Multi-Camera Vision Encoder for Three-Bot.
# Three-Bot을 위한 멀티 카메라 비전 인코더:
# 3대의 카메라로부터 들어오는 영상을 처리하여 AI 모델이 이해할 수 있는 특징(Token)으로 변환합니다.

Optimized vision encoding for 3-camera setup (3카메라 설정 최적화):
- Camera 1: Gripper 1 (egocentric) - 왼쪽 그리퍼 시점
- Camera 2: Gripper 2 (egocentric) - 오른쪽 그리퍼 시점 
- Camera 3: Global view (allocentric) - 전체 작업 공간 시점

Architecture (아키텍처):
- Shared ResNet-18/34 backbone: 파라미터 효율성을 위해 3대 카메라가 하나의 신경망 공유
- Camera-specific learnable embeddings: 각 카메라 시점의 고유한 특징을 구분하기 위한 임베딩
- Optional cross-view attention: 서로 다른 카메라 시점 간의 공간적 대응 관계 학습
- Output: 트랜스포머 입력용 멀티뷰 비주얼 토큰

Author: Architect_Sonnet @ Nova Robotics
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Optional, Tuple
from functools import partial


class ResNetBlock(nn.Module):
    """Basic ResNet block with pre-activation.
    # 기본적인 ResNet 블록: 이미지의 특징을 유지하면서 깊은 층을 쌓을 수 있게 합니다 (잔차 연결).
    """
    
    features: int
    stride: int = 1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Args:
            x: [B, H, W, C] 입력 텐서
            train: 학습 모드 여부 (배치 정규화 등에 영향)
        Returns:
            out: 특징이 추출된 결과 텐서
        """
        residual = x
        
        # Pre-activation: 활성화 함수와 정규화를 먼저 수행하여 학습 안정성 향상
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        
        # 첫 번째 컨볼루션 레이어
        x = nn.Conv(self.features, (3, 3), strides=self.stride, 
                   padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        
        # 두 번째 컨볼루션 레이어
        x = nn.Conv(self.features, (3, 3), padding='SAME', use_bias=False)(x)
        
        # 잔차 연결 (Residual Connection): 
        # 입력값과 결과값의 크기가 다르면 크기를 맞춰서 더해줌
        if residual.shape != x.shape:
            residual = nn.Conv(self.features, (1, 1), strides=self.stride,
                              use_bias=False)(residual)
        
        return x + residual


class ResNet18Backbone(nn.Module):
    """ResNet-18 backbone for image feature extraction.
    # 이미지 특징 추출을 위한 ResNet-18 백본 신경망입니다.
    # 출력: [B, H/32, W/32, 512] 크기의 공간적 특징 맵
    """
    
    num_blocks: Tuple[int, ...] = (2, 2, 2, 2)  # ResNet-18 구조
    num_filters: Tuple[int, ...] = (64, 128, 256, 512)
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Args:
            x: [B, H, W, 3] RGB 이미지 (0~1 사이로 정규화됨)
        Returns:
            features: 추출된 공간 특징
        """
        # 초기 레이어: 큰 커널(7x7)을 사용하여 전반적인 형태 파악
        x = nn.Conv(64, (7, 7), strides=2, padding='SAME', 
                   use_bias=False, name='conv1')(x)
        x = nn.BatchNorm(use_running_average=not train, name='bn1')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
        
        # ResNet 단계별 블록 쌓기
        for stage_idx, (num_blocks, num_filters) in enumerate(
            zip(self.num_blocks, self.num_filters)
        ):
            for block_idx in range(num_blocks):
                stride = 2 if (stage_idx > 0 and block_idx == 0) else 1
                x = ResNetBlock(
                    features=num_filters,
                    stride=stride,
                    name=f'stage{stage_idx}_block{block_idx}'
                )(x, train=train)
        
        return x


class CrossViewAttention(nn.Module):
    """Multi-head attention across camera views.
    # 서로 다른 카메라 뷰 간의 어텐션: 
    # 왼쪽/오른쪽 그리퍼가 서로를 "인지"하고, 전체 시점의 맥락을 통합하게 해줍니다.
    """
    
    num_heads: int = 8
    qkv_features: int = 512
    
    @nn.compact
    def __call__(self, 
                 view_tokens: jnp.ndarray,
                 view_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Args:
            view_tokens: 모든 뷰에서 추출된 토큰들의 합집합
        """
        # Multi-head self-attention: 모든 시점의 정보를 서로 비교 연산
        attended = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            name='cross_view_attn'
        )(view_tokens, mask=view_mask)
        
        # 잔차 연결 및 레이어 정규화
        tokens = nn.LayerNorm()(view_tokens + attended)
        
        # Feed-forward network: 위치별 정보를 비선형적으로 변환
        ffn = nn.Sequential([
            nn.Dense(self.qkv_features * 4),
            nn.gelu,
            nn.Dense(self.qkv_features),
        ])
        
        tokens = nn.LayerNorm()(tokens + ffn(tokens))
        
        return tokens


class ThreeBotVisionEncoder(nn.Module):
    """Multi-camera vision encoder for Three-Bot.
    # Three-Bot 전용 멀티 카메라 비전 인코더입니다.
    
    3대의 동기화된 카메라 영상을 처리합니다:
    1. 왼쪽 그리퍼 카메라 (1인칭)
    2. 오른쪽 그리퍼 카메라 (1인칭)
    3. 글로벌 카메라 (3인칭)
    """
    
    num_cameras: int = 3
    backbone_type: str = 'resnet18'
    feature_dim: int = 512
    use_cross_view_attention: bool = True
    image_size: Tuple[int, int] = (224, 224)
    patch_size: int = 7  # 224/32 = 7x7 공간 해상도
    
    def setup(self):
        """백본 신경망과 각종 임베딩 초기화."""
        
        # 공유 백본 (ResNet-18)
        if self.backbone_type == 'resnet18':
            self.backbone = ResNet18Backbone()
        else:
            raise NotImplementedError(f"Backbone {self.backbone_type} not implemented")
        
        # 카메라별 고유 벡터 (Learnable Embeddings)
        # 어느 카메라에서 온 영상인지 AI가 구분할 수 있게 합니다.
        self.camera_embeddings = self.param(
            'camera_embed',
            nn.initializers.normal(stddev=0.02),
            (self.num_cameras, self.feature_dim)
        )
        
        # 공간적 위치 임베딩 (Spatial Positional Embeddings)
        # 영상 내의 어느 위치(좌상단, 우하단 등)인지 가르쳐줍니다.
        num_patches = self.patch_size ** 2
        self.spatial_pos_embed = self.param(
            'spatial_pos_embed',
            nn.initializers.normal(stddev=0.02),
            (num_patches, self.feature_dim)
        )
        
        # 시간적 위치 임베딩 (Temporal Positional Embeddings)
        # 연속된 프레임 중에서 몇 번째 시점인지 알려줍니다.
        self.temporal_pos_embed = self.param(
            'temporal_pos_embed',
            nn.initializers.normal(stddev=0.02),
            (16, self.feature_dim)
        )
        
        # 공통 차원으로 투영하는 레이어
        self.feature_proj = nn.Dense(self.feature_dim, name='feature_proj')
        
        # 크로스 뷰 어텐션 모듈
        if self.use_cross_view_attention:
            self.cross_view_attn = CrossViewAttention(
                num_heads=8,
                qkv_features=self.feature_dim
            )
    
    def encode_single_view(self,
                          image: jnp.ndarray,
                          camera_idx: int,
                          train: bool = True) -> jnp.ndarray:
        """개별 카메라 시점의 영상을 인코딩합니다."""
        batch_size, seq_len, H, W, C = image.shape
        
        # 백본 처리를 위해 배치와 시간 차원을 하나로 합침
        image_flat = image.reshape(batch_size * seq_len, H, W, C)
        
        # ResNet 통과하여 특징 추출
        features = self.backbone(image_flat, train=train)
        # Shape: [B*T, H', W', C'] where H'=W'=7 for 224x224 input
        
        _, H_feat, W_feat, C_feat = features.shape
        
        # 특징 맵을 패치(나열된 조각) 형태로 변환 [B, T, 패치수, 차원]
        features = features.reshape(
            batch_size, seq_len, H_feat * W_feat, C_feat
        )  # [B, T, num_patches, C_feat]
        
        # 공통 특징 차원으로 변환
        features = self.feature_proj(features)  # [B, T, num_patches, D]
        
        # 1. 카메라 정보 더하기
        cam_embed = self.camera_embeddings[camera_idx]  # [D,]
        features = features + cam_embed[None, None, None, :]
        
        # 2. 공간 위치 정보 더하기
        features = features + self.spatial_pos_embed[None, None, :, :]
        
        # 3. 시간 순서 정보 더하기
        temporal_embed = self.temporal_pos_embed[:seq_len, :]  # [T, D]
        features = features + temporal_embed[None, :, None, :]
        
        return features
    
    @nn.compact
    def __call__(self,
                 images: Dict[str, jnp.ndarray],
                 train: bool = True) -> jnp.ndarray:
        """멀티 카메라 관측값을 통합 인코딩합니다."""
        # 입력 데이터 확인
        assert 'gripper1' in images, "Missing gripper1 camera"
        assert 'gripper2' in images, "Missing gripper2 camera"
        assert 'global' in images, "Missing global camera"
        
        batch_size, seq_len = images['gripper1'].shape[:2]
        
        # 3곳의 카메라 고리에서 각자 인코딩 수행
        camera_names = ['gripper1', 'gripper2', 'global']
        all_tokens = []
        
        for cam_idx, cam_name in enumerate(camera_names):
            tokens = self.encode_single_view(
                images[cam_name],
                camera_idx=cam_idx,
                train=train
            )  # [B, T, num_patches, D]
            all_tokens.append(tokens)
        
        # 모든 시점을 하나로 묶음 [B, T, 전체 토큰 수, D]
        visual_tokens = jnp.concatenate(all_tokens, axis=2)
        
        # (선택 사항) 크로스 뷰 어텐션을 사용하여 시점 간 정보 교환
        if self.use_cross_view_attention:
            # Flatten batch and time for attention
            B, T, N, D = visual_tokens.shape
            tokens_flat = visual_tokens.reshape(B * T, N, D)
            
            # Apply cross-view attention
            tokens_flat = self.cross_view_attn(tokens_flat)
            
            # Reshape back
            visual_tokens = tokens_flat.reshape(B, T, N, D)
        
        return visual_tokens
    
    def get_output_shape(self, seq_len: int = 1) -> Tuple[int, int]:
        """최종 출력될 비주얼 토큰의 개수와 차원을 반환합니다."""
        num_patches = self.patch_size ** 2
        num_tokens = self.num_cameras * num_patches
        return (num_tokens, self.feature_dim)


class ImagePreprocessor:
    """Three-Bot 비전 인코더를 위한 이미지 전처리 도구입니다."""
    
    def __init__(self,
                 target_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True):
        """
        Args:
            target_size: (H, W) to resize images
            normalize: Whether to normalize to [0, 1]
        """
        self.target_size = target_size
        self.normalize = normalize
    
    def preprocess(self, image: jnp.ndarray) -> jnp.ndarray:
        """단일 이미지를 리사이징하고 정규화합니다."""
        # Resize (assume jax.image.resize or use PIL)
        # For JAX, use jax.image.resize
        from jax.image import resize
        
        # 이미지 크기 조정 (224x224)
        image = resize(image, 
                      shape=(self.target_size[0], self.target_size[1], 3),
                      method='bilinear')
        
        # 0.0 ~ 1.0 사이로 정규화
        if self.normalize:
            if image.dtype == jnp.uint8:
                image = image.astype(jnp.float32) / 255.0
            else:
                image = jnp.clip(image, 0.0, 1.0)
        
        return image
    
    def preprocess_batch(self, images: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """여러 대의 카메라 영상을 한꺼번에 전처리합니다."""
        processed = {}
        for cam_name, img_batch in images.items():
            # 배치 전체에 대해 병렬(vmap) 처리
            processed[cam_name] = jax.vmap(
                jax.vmap(self.preprocess)
            )(img_batch)
        
        return processed


# 팩토리 함수: 인코더를 쉽게 생성할 수 있게 도와줍니다.
def create_vision_encoder(
    num_cameras: int = 3,
    use_cross_view_attention: bool = True,
    feature_dim: int = 512
) -> ThreeBotVisionEncoder:
    """Create Three-Bot vision encoder.
    
    Args:
        num_cameras: Number of camera views
        use_cross_view_attention: Enable cross-view fusion
        feature_dim: Output feature dimension
    
    Returns:
        ThreeBotVisionEncoder instance
    """
    return ThreeBotVisionEncoder(
        num_cameras=num_cameras,
        use_cross_view_attention=use_cross_view_attention,
        feature_dim=feature_dim
    )
