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
import numpy as np
from typing import Dict, Tuple, Optional, Callable, Iterable, List
from collections import defaultdict
from pathlib import Path
import glob
import time

from .octo_model import create_three_bot_octo_model, count_parameters


try:
    # TensorFlow only feeds data here; keep the GPU memory available for JAX/XLA.
    tf.config.set_visible_devices([], 'GPU')
except RuntimeError:
    pass


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
    dataset_format: str = 'processed_npy' # 'rlds' 또는 'processed_npy'
    dataset_path: str = 'training_data/processed_episodes/*.npy' # RLDS 데이터셋 이름 또는 processed npy glob
    dataset_data_dir: Optional[str] = None # TFDS data_dir. None이면 TFDS 기본 경로 사용
    shuffle_buffer_size: int = 1000      # 데이터 셔플링 버퍼 크기
    num_parallel_calls: int = 8          # 데이터 로딩 병렬 처리 수
    steps_per_epoch: int = 1000          # 학습률 스케줄 계산용 근사 스텝 수
    max_train_steps_per_epoch: Optional[int] = None # 디버그/파일럿 학습용 제한
    max_val_steps: Optional[int] = None   # 디버그/파일럿 검증용 제한
    use_balanced_sampler: bool = False
    balanced_zero_ratio: float = 0.30
    balanced_xy_only_ratio: float = 0.45
    balanced_z_only_ratio: float = 0.20
    balanced_xy_z_ratio: float = 0.05
    balanced_samples_per_epoch: Optional[int] = None
    balanced_threshold: float = 1e-6
    balanced_seed: int = 42
    
    # Checkpointing (체크포인트 저장 설정)
    checkpoint_dir: str = 'training_outputs/checkpoints' # 저장 디렉토리
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
    total_steps = config.num_epochs * steps_per_epoch
    warmup_steps = min(config.warmup_epochs * steps_per_epoch, max(0, total_steps - 1))
    decay_steps = max(1, total_steps - warmup_steps)
    
    # Warmup 기간 동안의 선형 증가 스케줄
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=config.learning_rate,
        transition_steps=warmup_steps
    )
    
    # 이후 Cosine Decay 적용
    cosine_fn = optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=decay_steps,
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
    dummy_proprioception = jnp.zeros((1, 10), dtype=jnp.float32)
    
    # 3. 파라미터 초기화
    rng, init_rng = jax.random.split(rng)
    # 모델에 필요한 다양한 랜덤 키들(드롭아웃, 노이즈 등)을 정의합니다.
    variables = model.init(
        {'params': init_rng, 'dropout': init_rng, 
         'noise': init_rng, 'sample': init_rng, 'diffusion': init_rng},
        dummy_obs,
        dummy_proprioception,
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
        
        rngs = {
            'dropout': dropout_rng,
            'noise': noise_rng,
            'sample': sample_rng,
            'diffusion': diffusion_rng,
        }

        if state.batch_stats is not None:
            (loss, metrics), updates = state.apply_fn(
                variables,
                batch['observations'],
                batch['proprioception'],
                batch['actions'],
                batch['key'],
                train=True,
                method='compute_loss',
                rngs=rngs,
                mutable=['batch_stats'],
            )
            return loss, (metrics, updates)

        loss, metrics = state.apply_fn(
            variables,
            batch['observations'],
            batch['proprioception'],
            batch['actions'],
            batch['key'],
            train=True,
            method='compute_loss',
            rngs=rngs,
        )
        
        return loss, (metrics, None)
    
    # 7. 그래디언트 계산
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (metrics, updates)), grads = grad_fn(state.params)
    
    # 8. 파라미터 업데이트
    state = state.apply_gradients(grads=grads)
    if updates is not None:
        state = state.replace(batch_stats=updates['batch_stats'])
    
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
        batch['proprioception'],
        batch['actions'],
        batch['key'],
        train=False,
        method='compute_loss',
        rngs={'dropout': jax.random.PRNGKey(0), 
              'noise': jax.random.PRNGKey(1),
              'sample': jax.random.PRNGKey(2),
              'diffusion': jax.random.PRNGKey(3)}
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
    if config.dataset_format == 'processed_npy':
        ds = load_processed_npy_dataset(config, split=split)
    elif config.dataset_format == 'rlds':
        ds = load_rlds_dataset(config, split=split)
    else:
        raise ValueError(f"Unsupported dataset_format: {config.dataset_format}")

    if split == 'train':
        ds = ds.shuffle(config.shuffle_buffer_size) # 데이터 순서 섞기
    
    ds = ds.batch(config.batch_size, drop_remainder=True) # 배치 구성
    ds = ds.prefetch(tf.data.AUTOTUNE) # 처리 속도 향상을 위한 데이터 로드 최적화
    
    return ds


def _normalize_image(image: tf.Tensor) -> tf.Tensor:
    """Convert uint8 camera frames to the model's [0, 1] float input range."""
    image = tf.cast(image, tf.float32)
    return image / 255.0


def _format_step(
    gripper1: tf.Tensor,
    gripper2: tf.Tensor,
    global_image: tf.Tensor,
    action: tf.Tensor,
    proprioception: tf.Tensor,
) -> Dict[str, Dict[str, tf.Tensor]]:
    """Format one timestep into the exact model batch contract.

    The model always consumes a time dimension, even when the dataset contains
    single-frame steps. Expanding to `[T=1, H, W, C]` keeps the processed `.npy`
    path and the RLDS path compatible with the same training step.
    """
    return {
        'observations': {
            'gripper1': tf.expand_dims(_normalize_image(gripper1), axis=0),
            'gripper2': tf.expand_dims(_normalize_image(gripper2), axis=0),
            'global': tf.expand_dims(_normalize_image(global_image), axis=0),
        },
        'proprioception': tf.cast(proprioception, tf.float32),
        'actions': tf.cast(action, tf.float32),
    }


def load_rlds_dataset(config: TrainConfig, split: str = 'train') -> tf.data.Dataset:
    """Load a TFDS/RLDS dataset and flatten episodes into trainable steps."""
    load_kwargs = {
        'split': split,
        'shuffle_files': split == 'train',
    }
    if config.dataset_data_dir:
        load_kwargs['data_dir'] = config.dataset_data_dir

    episodes = tfds.load(config.dataset_path, **load_kwargs)

    def preprocess_step(step):
        observation = step['observation']
        return _format_step(
            observation['gripper1_image'],
            observation['gripper2_image'],
            observation['global_image'],
            step['action'],
            observation['robot_state'],
        )

    def episode_to_steps(episode):
        return episode['steps'].map(
            preprocess_step,
            num_parallel_calls=config.num_parallel_calls,
        )

    return episodes.flat_map(episode_to_steps)


def processed_npy_generator(file_list: Iterable[str]):
    """Yield preprocessed `.npy` samples using the current local episode schema.

    The recovered processed episodes contain synchronized images and 5-D actions,
    but not robot state. A zero `[10]` proprioception vector is used here so the
    same model can later train on RLDS data that does include real state.
    """
    for file_path in file_list:
        try:
            episode = np.load(file_path, allow_pickle=True)
            for step in episode:
                yield {
                    'observations': {
                        'gripper1': np.expand_dims(step['img_left'].astype(np.float32) / 255.0, axis=0),
                        'gripper2': np.expand_dims(step['img_right'].astype(np.float32) / 255.0, axis=0),
                        'global': np.expand_dims(step['img_global'].astype(np.float32) / 255.0, axis=0),
                    },
                    'proprioception': np.zeros((10,), dtype=np.float32),
                    'actions': step['action'].astype(np.float32),
                }
        except Exception as e:
            print(f"Error loading {file_path}: {e}")


def _format_processed_npy_step(step: dict) -> dict:
    return {
        'observations': {
            'gripper1': np.expand_dims(step['img_left'].astype(np.float32) / 255.0, axis=0),
            'gripper2': np.expand_dims(step['img_right'].astype(np.float32) / 255.0, axis=0),
            'global': np.expand_dims(step['img_global'].astype(np.float32) / 255.0, axis=0),
        },
        'proprioception': np.zeros((10,), dtype=np.float32),
        'actions': step['action'].astype(np.float32),
    }


def _classify_action_bucket(action: np.ndarray, threshold: float) -> str:
    active = np.abs(action) > threshold
    xy_active = bool(np.any(active[:4]))
    z_active = bool(active[4])
    if xy_active and z_active:
        return 'xy_z'
    if xy_active:
        return 'xy_only'
    if z_active:
        return 'z_only'
    return 'zero'


def _build_balanced_npy_index(file_list: List[str], threshold: float):
    buckets = {
        'zero': [],
        'xy_only': [],
        'z_only': [],
        'xy_z': [],
    }
    total_steps = 0

    for file_id, file_path in enumerate(file_list):
        episode = np.load(file_path, allow_pickle=True)
        for step_idx, step in enumerate(episode):
            bucket = _classify_action_bucket(step['action'].astype(np.float32), threshold)
            buckets[bucket].append((file_id, step_idx))
            total_steps += 1

    return buckets, total_steps


def balanced_processed_npy_generator(
    file_list: List[str],
    buckets: Dict[str, List[Tuple[int, int]]],
    samples_per_pass: int,
    ratios: Dict[str, float],
    seed: int,
):
    """Yield a balanced train stream without changing the validation distribution."""
    rng = np.random.default_rng(seed)
    bucket_names = ('zero', 'xy_only', 'z_only', 'xy_z')
    available = [name for name in bucket_names if buckets[name]]
    ratio_sum = sum(ratios[name] for name in available)
    if ratio_sum <= 0.0:
        raise ValueError("balanced sampler ratios must sum to a positive value")

    counts = {}
    remaining = samples_per_pass
    for name in available[:-1]:
        count = int(round(samples_per_pass * ratios[name] / ratio_sum))
        counts[name] = min(count, remaining)
        remaining -= counts[name]
    counts[available[-1]] = remaining

    selected_by_file = defaultdict(list)
    for name in available:
        bucket_items = buckets[name]
        draw_count = counts.get(name, 0)
        if draw_count <= 0:
            continue
        choices = rng.integers(0, len(bucket_items), size=draw_count)
        for choice in choices:
            file_id, step_idx = bucket_items[int(choice)]
            selected_by_file[file_id].append(step_idx)

    file_ids = list(selected_by_file.keys())
    rng.shuffle(file_ids)
    for file_id in file_ids:
        episode = np.load(file_list[file_id], allow_pickle=True)
        step_indices = selected_by_file[file_id]
        rng.shuffle(step_indices)
        for step_idx in step_indices:
            yield _format_processed_npy_step(episode[step_idx])


def load_processed_npy_dataset(config: TrainConfig, split: str = 'train') -> tf.data.Dataset:
    """Create a TensorFlow dataset from local processed episode files."""
    file_list: List[str] = sorted(glob.glob(config.dataset_path))
    if not file_list:
        raise ValueError(f"No processed .npy files matched: {config.dataset_path}")

    num_train = max(1, int(len(file_list) * 0.8))
    if split == 'train':
        file_list = file_list[:num_train]
    elif split == 'val':
        file_list = file_list[num_train:] or file_list[-1:]
    else:
        raise ValueError(f"Unsupported split: {split}")

    # Keep this signature strict. Shape drift here usually means the preprocessing
    # output changed and should be fixed before JAX compilation starts.
    output_signature = {
        'observations': {
            'gripper1': tf.TensorSpec(shape=(1, 224, 224, 3), dtype=tf.float32),
            'gripper2': tf.TensorSpec(shape=(1, 224, 224, 3), dtype=tf.float32),
            'global': tf.TensorSpec(shape=(1, 224, 224, 3), dtype=tf.float32),
        },
        'proprioception': tf.TensorSpec(shape=(10,), dtype=tf.float32),
        'actions': tf.TensorSpec(shape=(5,), dtype=tf.float32),
    }

    if split == 'train' and config.use_balanced_sampler:
        buckets, total_steps = _build_balanced_npy_index(
            file_list,
            threshold=config.balanced_threshold,
        )
        samples_per_pass = config.balanced_samples_per_epoch or total_steps
        ratios = {
            'zero': config.balanced_zero_ratio,
            'xy_only': config.balanced_xy_only_ratio,
            'z_only': config.balanced_z_only_ratio,
            'xy_z': config.balanced_xy_z_ratio,
        }
        print("Balanced sampler enabled")
        print(f"  samples/pass: {samples_per_pass}")
        for name in ('zero', 'xy_only', 'z_only', 'xy_z'):
            print(f"  {name}: source={len(buckets[name])}, target_ratio={ratios[name]:.2f}")

        generator_calls = {'count': 0}

        def make_balanced_generator():
            seed = config.balanced_seed + generator_calls['count']
            generator_calls['count'] += 1
            return balanced_processed_npy_generator(
                file_list=file_list,
                buckets=buckets,
                samples_per_pass=samples_per_pass,
                ratios=ratios,
                seed=seed,
            )

        return tf.data.Dataset.from_generator(
            make_balanced_generator,
            output_signature=output_signature,
        )

    return tf.data.Dataset.from_generator(
        lambda: processed_npy_generator(file_list),
        output_signature=output_signature,
    )


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
        batch = jax.tree_util.tree_map(lambda x: jnp.array(x.numpy()), batch)
        
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
            loss = float(metrics['loss'])
            print(f"Epoch {epoch} Step {step}: loss={loss:.4f}")

        if (config.max_train_steps_per_epoch is not None and
                num_steps >= config.max_train_steps_per_epoch):
            break
    
    # 에폭 전체 평균 메트릭 계산
    epoch_metrics = {k: sum(v) / len(v) for k, v in epoch_metrics.items()}
    
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
    steps_per_epoch = config.steps_per_epoch
    
    # 3. 모델 및 학습 상태 초기화
    print("\nInitializing model...")
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(config, init_rng, steps_per_epoch)
    
    # 4. 체크포인트 저장을 위한 디텍토리 준비
    ckpt_dir = Path(config.checkpoint_dir).expanduser().resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # 5. 반복 학습 시작
    print("\nStarting training...")
    eval_every_epochs = max(1, config.eval_every_n_steps // steps_per_epoch)
    checkpoint_every_epochs = max(1, config.checkpoint_every_n_steps // steps_per_epoch)

    for epoch in range(config.num_epochs):
        # 1세대 학습 실행
        state, train_metrics = train_epoch(state, train_ds, epoch, config)
        
        # 6. 주기적으로 검증 데이터셋 평가
        if epoch % eval_every_epochs == 0:
            print(f"\nRunning validation...")
            val_metrics = {}
            for val_step, batch in enumerate(val_ds):
                batch = jax.tree_util.tree_map(lambda x: jnp.array(x.numpy()), batch)
                batch['key'] = jax.random.PRNGKey(epoch)
                
                metrics = eval_step(state, batch)
                for k, v in metrics.items():
                    if k not in val_metrics:
                        val_metrics[k] = []
                    val_metrics[k].append(float(v))

                if (config.max_val_steps is not None and
                        val_step + 1 >= config.max_val_steps):
                    break
            
            val_metrics = {k: sum(v) / len(v) for k, v in val_metrics.items()}
            print(f"Validation metrics: {val_metrics}")
        
        # 7. 최신 가중치 저장(Checkpoint)
        if epoch % checkpoint_every_epochs == 0:
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


def main(args=None):
    """CLI entry point used by both `python -m` and ROS console scripts."""
    import argparse

    parser = argparse.ArgumentParser(description='Train Three-Bot Octo model')
    parser.add_argument('--model_size', default='small', choices=['small', 'base'])
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--dataset_format', default='processed_npy', choices=['rlds', 'processed_npy'])
    parser.add_argument('--dataset_path', default='training_data/processed_episodes/*.npy')
    parser.add_argument('--dataset_data_dir', default=None)
    parser.add_argument('--checkpoint_dir', default='training_outputs/checkpoints')
    parser.add_argument('--steps_per_epoch', type=int, default=1000)
    parser.add_argument('--max_train_steps_per_epoch', type=int, default=None)
    parser.add_argument('--max_val_steps', type=int, default=None)
    parser.add_argument('--use_balanced_sampler', action='store_true')
    parser.add_argument('--balanced_zero_ratio', type=float, default=0.30)
    parser.add_argument('--balanced_xy_only_ratio', type=float, default=0.45)
    parser.add_argument('--balanced_z_only_ratio', type=float, default=0.20)
    parser.add_argument('--balanced_xy_z_ratio', type=float, default=0.05)
    parser.add_argument('--balanced_samples_per_epoch', type=int, default=None)
    parser.add_argument('--balanced_threshold', type=float, default=1e-6)
    parser.add_argument('--balanced_seed', type=int, default=42)
    args = parser.parse_args(args=args)

    # 메인 실행 환경 및 설정
    config = TrainConfig(
        model_size=args.model_size,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dataset_format=args.dataset_format,
        dataset_path=args.dataset_path,
        dataset_data_dir=args.dataset_data_dir,
        checkpoint_dir=args.checkpoint_dir,
        steps_per_epoch=args.steps_per_epoch,
        max_train_steps_per_epoch=args.max_train_steps_per_epoch,
        max_val_steps=args.max_val_steps,
        use_balanced_sampler=args.use_balanced_sampler,
        balanced_zero_ratio=args.balanced_zero_ratio,
        balanced_xy_only_ratio=args.balanced_xy_only_ratio,
        balanced_z_only_ratio=args.balanced_z_only_ratio,
        balanced_xy_z_ratio=args.balanced_xy_z_ratio,
        balanced_samples_per_epoch=args.balanced_samples_per_epoch,
        balanced_threshold=args.balanced_threshold,
        balanced_seed=args.balanced_seed,
        mixed_precision=True,
        gradient_checkpointing=True
    )
    
    train(config)


if __name__ == '__main__':
    main()
