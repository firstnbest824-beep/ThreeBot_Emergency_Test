"""JAX/Flax Training Pipeline for Three-Bot Octo Model.
# Three-Bot Octo 모델을 위한 JAX/Flax 기반 학습 파이프라인입니다.

Optimized for NVIDIA Jetson Orin AGX with (Jetson Orin AGX 최적화 기술):
- BF16 Mixed Precision (BF16 혼합 정밀도 학습)
- Gradient Checkpointing (메모리 절약을 위한 그래디언트 체크포인트)
- Gradient Accumulation (그래디언트 축적을 통한 가상 배치 크기 확대)
- Multi-GPU support (if available) (멀티 GPU 지원)

Author: Architect_Sonnet @ Nova Robotics
"""

import jax
import jax.numpy as jnp
import optax
import flax
from flax.training import train_state, checkpoints
from flax import struct
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Dict, Tuple, Optional, Callable
from pathlib import Path
import time

from .octo_model import create_three_bot_octo_model, count_parameters


@struct.dataclass
class TrainConfig:
    """Training configuration.
    # 학습 설정값을 정의하는 데이터 클래스입니다.
    """
    
    # Model (모델 관련 설정)
    model_size: str = 'small'  # 모델 크기: 'small' 또는 'base'
    use_diffusion: bool = True # 디퓨전(Diffusion) 헤드 사용 여부
    diffusion_steps: int = 10  # 디퓨전 샘플링 단계 수
    
    # Training (학습 기본 설정)
    num_epochs: int = 100      # 전체 학습 에폭(Epoch) 수
    batch_size: int = 8        # 디바이스당 배치 크기
    learning_rate: float = 3e-4 # 기본 학습률
    weight_decay: float = 1e-4  # 가중치 감쇠 (L2 정규화 효과)
    warmup_epochs: int = 5      # 초기 학습 안정화를 위한 웜업 에폭 수
    gradient_clip_norm: float = 1.0 # 그래디언트 폭주 방지를 위한 클리핑 임계값
    
    # Optimization (최적화 기술 설정)
    mixed_precision: bool = True  # BF16 사용 여부 (메모리 절약 및 속도 향상)
    gradient_checkpointing: bool = True # 메모리 사용량 최적화
    gradient_accumulation_steps: int = 4 # 실제 배치를 나누어 연산하여 메모리 부족 해결
    
    # Data (데이터셋 관련)
    dataset_path: str = 'three_bot_rlds' # RLDS 데이터셋 경로
    shuffle_buffer_size: int = 1000      # 데이터 셔플링 버퍼 크기
    num_parallel_calls: int = 8          # 데이터 로딩 병렬 처리 수
    
    # Checkpointing (체크포인트 저장 설정)
    checkpoint_dir: str = 'checkpoints'   # 저장 디렉토리
    checkpoint_every_n_steps: int = 1000  # 저장 주기 (스텝 단위)
    keep_n_checkpoints: int = 5           # 유지할 최대 체크포인트 수
    
    # Logging (로그 기록 설정)
    log_every_n_steps: int = 10           # 훈련 로그 출력 주기
    eval_every_n_steps: int = 500         # 검증(Evaluation) 주기
    
    # Hardware (하드웨어 설정)
    num_devices: int = 1  # JAX가 자동으로 GPU 개수를 감지합니다.


class TrainState(train_state.TrainState):
    """Extended train state with batch statistics.
    # Flax의 기본 TrainState를 확장하여 배치 통계량과 랜덤 키를 포함시킵니다.
    """
    batch_stats: Optional[flax.core.FrozenDict] = None
    key: jax.random.PRNGKey = None


def create_learning_rate_schedule(
    config: TrainConfig,
    steps_per_epoch: int
) -> Callable[[int], float]:
    """Create learning rate schedule with warmup and cosine decay.
    # 학습률 스케줄 생성: 초기에는 점진적으로 상승(Warmup)하고 이후 서서히 감소(Cosine Decay)합니다.
    """
    warmup_steps = config.warmup_epochs * steps_per_epoch
    total_steps = config.num_epochs * steps_per_epoch
    
    # Warmup 기간 동안의 선형 증가 스케줄
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=config.learning_rate,
        transition_steps=warmup_steps
    )
    
    # 이후 Cosine Decay 적용
    cosine_fn = optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=total_steps - warmup_steps,
        alpha=0.1  # 최저 학습률을 초기값의 10% 수준으로 유지
    )
    
    return optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[warmup_steps]
    )


def create_train_state(
    config: TrainConfig,
    rng: jax.random.PRNGKey,
    steps_per_epoch: int
) -> TrainState:
    """Initialize model and optimizer.
    # 모델의 파라미터를 초기화하고 최적화 도구(Optimizer)를 설정하여 학습 상태를 생성합니다.
    """
    # 1. Octo 모델 생성
    model = create_three_bot_octo_model(
        model_size=config.model_size,
        use_diffusion=config.use_diffusion,
        diffusion_steps=config.diffusion_steps
    )
    
    # 2. 초기화를 위한 더미 데이터 준비 (입력 형태 정의)
    dummy_obs = {
        'gripper1': jnp.ones((1, 1, 224, 224, 3)),
        'gripper2': jnp.ones((1, 1, 224, 224, 3)),
        'global': jnp.ones((1, 1, 224, 224, 3)),
    }
    
    # 3. 파라미터 초기화
    rng, init_rng = jax.random.split(rng)
    # 모델에 필요한 다양한 랜덤 키들(드롭아웃, 노이즈 등)을 정의합니다.
    variables = model.init(
        {'params': init_rng, 'dropout': init_rng, 
         'noise': init_rng, 'sample': init_rng, 'diffusion': init_rng},
        dummy_obs,
        train=True
    )
    
    params = variables['params']
    batch_stats = variables.get('batch_stats', None)
    
    # 4. 모델 정보 출력 (파라미터 수 확인)
    num_params = count_parameters(params)
    print(f"Model size: {config.model_size}")
    print(f"Total parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # 5. 최적화 도구 설정 (AdamW + 그래디언트 클리핑)
    lr_schedule = create_learning_rate_schedule(config, steps_per_epoch)
    
    tx = optax.chain(
        optax.clip_by_global_norm(config.gradient_clip_norm), # 그래디언트 절단
        optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=config.weight_decay,
            b1=0.9,
            b2=0.999
        )
    )
    
    # 그래디언트 축적(Gradient Accumulation) 사용 시 설정
    if config.gradient_accumulation_steps > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=config.gradient_accumulation_steps)
    
    # 6. 최종 TrainState 인스턴스 반환
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
        key=rng
    )


@jax.jit
def train_step(
    state: TrainState,
    batch: Dict[str, jnp.ndarray]
) -> Tuple[TrainState, dict]:
    """Single training step.
    # 단일 학습 스텝 수행: 전방향 전파(Forward), 그래디언트 계산, 파라미터 업데이트가 일어나는 핵심 루프입니다.
    # @jax.jit 데코레이터를 통해 컴파일되어 매우 빠르게 실행됩니다.
    """
    def loss_fn(params):
        """손실 함수 계산 내부 함수."""
        # 다양한 연산을 위한 랜덤 키 생성
        dropout_rng, noise_rng, sample_rng, diffusion_rng = jax.random.split(state.key, 4)
        
        # Forward pass
        variables = {'params': params}
        if state.batch_stats is not None:
            variables['batch_stats'] = state.batch_stats
        
        # 모델의 손실값 계산 메서드 호출
        loss, metrics = state.apply_fn(
            variables,
            batch['observations'],
            batch['actions'],
            batch['key'],  # 디퓨전 샘플링용 키
            method='compute_loss',
            rngs={'dropout': dropout_rng, 'noise': noise_rng, 
                  'sample': sample_rng, 'diffusion': diffusion_rng}
        )
        
        return loss, metrics
    
    # 7. 그래디언트 계산
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)
    
    # 8. 파라미터 업데이트
    state = state.apply_gradients(grads=grads)
    
    # 다음 스텝을 위해 랜덤 키 갱신
    state = state.replace(key=jax.random.split(state.key)[0])
    
    # 모니터링을 위한 메트릭 추가
    grad_norm = jnp.sqrt(
        sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads))
    )
    metrics['grad_norm'] = grad_norm
    metrics['loss'] = loss
    
    return state, metrics


@jax.jit
def eval_step(
    state: TrainState,
    batch: Dict[str, jnp.ndarray]
) -> dict:
    """Single evaluation step.
    # 단일 검증 스텝 수행: 학습은 하지 않고 모델의 현재 성능(Loss 등)만 측정합니다.
    """
    variables = {'params': state.params}
    if state.batch_stats is not None:
        variables['batch_stats'] = state.batch_stats
    
    loss, metrics = state.apply_fn(
        variables,
        batch['observations'],
        batch['actions'],
        batch['key'],
        method='compute_loss',
        rngs={'dropout': jax.random.PRNGKey(0), 
              'noise': jax.random.PRNGKey(1),
              'sample': jax.random.PRNGKey(2),
              'diffusion': jax.random.PRNGKey(3)},
        train=False # 학습 모드 해제 (드롭아웃 등 비활성화)
    )
    
    metrics['loss'] = loss
    return metrics


def load_tfds_dataset(
    config: TrainConfig,
    split: str = 'train'
) -> tf.data.Dataset:
    """Load RLDS dataset from TensorFlow Datasets.
    # RLDS 형식의 데이터셋을 읽어와서 파이프라인(전처리, 셔플링, 배치)을 구축합니다.
    """
    # 1. TFDS 데이터셋 로드
    ds = tfds.load(
        config.dataset_path,
        split=split,
        shuffle_files=(split == 'train')
    )
    
    # 2. 전처리 함수 정의
    def preprocess(episode):
        # 에피소드 데이터에서 영상과 액션을 추출
        steps = episode['steps']
        
        # Stack steps into sequences
        observations = {
            'gripper1': steps['observation']['gripper1_image'],
            'gripper2': steps['observation']['gripper2_image'],
            'global': steps['observation']['global_image'],
        }
        actions = steps['action']
        
        return {
            'observations': observations,
            'actions': actions,
            'key': jax.random.PRNGKey(0)  # 임시 키 (추후 학습 시 갱신됨)
        }
    
    # 3. 전처리 적용 및 최적화 설정
    ds = ds.map(preprocess, num_parallel_calls=config.num_parallel_calls)
    
    # Shuffle and batch
    if split == 'train':
        ds = ds.shuffle(config.shuffle_buffer_size) # 데이터 순서 섞기
    
    ds = ds.batch(config.batch_size, drop_remainder=True) # 배치 구성
    ds = ds.prefetch(tf.data.AUTOTUNE) # 처리 속도 향상을 위한 데이터 로드 최적화
    
    return ds


def train_epoch(
    state: TrainState,
    train_ds: tf.data.Dataset,
    epoch: int,
    config: TrainConfig
) -> Tuple[TrainState, dict]:
    """Train for one epoch.
    # 1세대(Epoch) 전체 학습 루프입니다.
    """
    epoch_metrics = {}
    num_steps = 0
    start_time = time.time()
    
    # 4. 데이터셋 반복 실행
    for step, batch in enumerate(train_ds):
        # 텐서플로우 텐서를 JAX 배열로 변환
        batch = jax.tree_map(lambda x: jnp.array(x.numpy()), batch)
        
        # 각 배치마다 고유한 랜덤 키 부여
        batch['key'] = jax.random.fold_in(state.key, step)
        
        # 5. 핵심 학습 스텝 실행
        state, metrics = train_step(state, batch)
        
        # 결과 메트릭 수집
        for k, v in metrics.items():
            if k not in epoch_metrics:
                epoch_metrics[k] = []
            epoch_metrics[k].append(float(v))
        
        num_steps += 1
        
        # 주기적으로 로그 출력
        if step % config.log_every_n_steps == 0:
            loss = metrics['loss']
            # 옵티마이저 하이퍼파라미터에서 현재 학습률 가져오기
            lr = state.opt_state[0].hyperparams['learning_rate']  # type: ignore
            print(f"Epoch {epoch} Step {step}: loss={loss:.4f}, lr={lr:.6f}")
    
    # 에폭 전체 평균 메트릭 계산
    epoch_metrics = {
        k: sum(v) / len(v) for k, v in epoch_metrics.items()
    }
    
    epoch_time = time.time() - start_time
    samples_per_sec = (num_steps * config.batch_size) / epoch_time
    
    print(f"Epoch {epoch} completed: {epoch_time:.1f}s, "
          f"{samples_per_sec:.1f} samples/sec")
    print(f"Metrics: {epoch_metrics}")
    
    return state, epoch_metrics


def train(config: TrainConfig):
    """Main training loop.
    # 전체 학습 과정을 관리하는 최상위 함수입니다.
    """
    print("=" * 80)
    print("Three-Bot Octo Model Training")
    print("=" * 80)
    print(f"Configuration: {config}")
    
    # 1. 초기 랜덤 키 설정
    rng = jax.random.PRNGKey(42)
    
    # 2. 데이터셋 로드
    print("\nLoading datasets...")
    train_ds = load_tfds_dataset(config, split='train')
    val_ds = load_tfds_dataset(config, split='val')
    
    # Compute steps per epoch (approximate)
    # TODO: 데이터셋 정보로부터 정확한 스텝 수 가져오기
    steps_per_epoch = 1000  # 임시 설정
    
    # 3. 모델 및 학습 상태 초기화
    print("\nInitializing model...")
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(config, init_rng, steps_per_epoch)
    
    # 4. 체크포인트 저장을 위한 디텍토리 준비
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # 5. 반복 학습 시작
    print("\nStarting training...")
    for epoch in range(config.num_epochs):
        # 1세대 학습 실행
        state, train_metrics = train_epoch(state, train_ds, epoch, config)
        
        # 6. 주기적으로 검증 데이터셋 평가
        if epoch % (config.eval_every_n_steps // steps_per_epoch) == 0:
            print(f"\nRunning validation...")
            val_metrics = {}
            for batch in val_ds:
                batch = jax.tree_map(lambda x: jnp.array(x.numpy()), batch)
                batch['key'] = jax.random.PRNGKey(epoch)
                
                metrics = eval_step(state, batch)
                for k, v in metrics.items():
                    if k not in val_metrics:
                        val_metrics[k] = []
                    val_metrics[k].append(float(v))
            
            val_metrics = {k: sum(v) / len(v) for k, v in val_metrics.items()}
            print(f"Validation metrics: {val_metrics}")
        
        # 7. 최신 가중치 저장(Checkpoint)
        if epoch % (config.checkpoint_every_n_steps // steps_per_epoch) == 0:
            print(f"\nSaving checkpoint...")
            checkpoints.save_checkpoint(
                ckpt_dir=str(ckpt_dir),
                target=state,
                step=epoch,
                keep=config.keep_n_checkpoints,
                overwrite=True
            )
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)


if __name__ == '__main__':
    # 메인 실행 환경 및 설정
    config = TrainConfig(
        model_size='small',
        num_epochs=100,
        batch_size=8,
        learning_rate=3e-4,
        mixed_precision=True,
        gradient_checkpointing=True
    )
    
    train(config)
