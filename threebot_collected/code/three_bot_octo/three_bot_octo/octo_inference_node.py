"""Real-Time Octo Inference Node for ROS 2.
# ROS 2용 실시간 Octo 추론 노드: 
# 학습된 Three-Bot Octo 모델을 사용하여 실시간으로 로봇을 제어합니다.

Author: Architect_Sonnet @ Nova Robotics
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
try:
    # 로봇의 현재 상태(위치, 속도 등)를 정의한 커스텀 메시지 임포트
    from three_bot_core.msg import RobotState
except ImportError:
    RobotState = None  # 메시지가 없는 경우 예외 처리
from cv_bridge import CvBridge
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from threading import Lock
import time

from .octo_model import create_three_bot_octo_model
from .vision_encoder import ImagePreprocessor
from flax.training import checkpoints


class OctoInferenceNode(Node):
    """Real-time Octo model inference for Three-Bot control.
    # Three-Bot 제어를 위한 실시간 Octo 모델 추론 노드 클래스입니다.
    
    Subscribes to (구독하는 토픽):
        /three_bot/camera/gripper1/image_raw (Image): 왼쪽 그리퍼 카메라
        /three_bot/camera/gripper2/image_raw (Image): 오른쪽 그리퍼 카메라
        /three_bot/camera/global/image_raw (Image): 전체 작업 공간 카메라
        /three_bot/control/robot_state (RobotState): 로봇의 현재 관절/엔더 상태 (Proprioception)
    
    Publishes to (발행하는 토픽):
        /three_bot/policy/action (Float32MultiArray): 예측된 액션 [ΔX₁, ΔY₁, ΔX₂, ΔY₂, ΔZ_shared]
    
    Parameters (설정 파라미터):
        ~checkpoint_path: 학습된 모델 가중치 파일 경로
        ~control_rate: 제어 루프 주기 (Hz, 기본값: 20)
        ~model_size: 모델 크기 ('small' 또는 'base')
        ~use_gpu: GPU 추론 사용 여부 (기본값: True)
    """
    
    def __init__(self):
        super().__init__('octo_inference_node')
        
        # ROS 2 파라미터 선언 및 기본값 설정
        self.declare_parameter('checkpoint_path', 'checkpoints/final')
        self.declare_parameter('control_rate', 20.0)
        self.declare_parameter('model_size', 'small')
        self.declare_parameter('use_gpu', True)
        
        # 설정된 파라미터 값 가져오기
        checkpoint_path = self.get_parameter('checkpoint_path').value
        control_rate = self.get_parameter('control_rate').value
        model_size = self.get_parameter('model_size').value
        use_gpu = self.get_parameter('use_gpu').value
        
        self.get_logger().info("=" * 80)
        self.get_logger().info("Three-Bot Octo Inference Node")
        self.get_logger().info("=" * 80)
        self.get_logger().info(f"Checkpoint: {checkpoint_path}")
        self.get_logger().info(f"Control rate: {control_rate} Hz")
        self.get_logger().info(f"Model size: {model_size}")
        
        # JAX 환경 설정: GPU 미사용 시 CPU 강제 로드
        if not use_gpu:
            jax.config.update('jax_platform_name', 'cpu')
        
        # 1. Octo 모델 생성
        self.get_logger().info("Loading model...")
        from .octo_model import create_three_bot_model
        self.model = create_three_bot_model()
        
        # 2. 체크포인트(학습된 가중치) 로드
        self.params = checkpoints.restore_checkpoint(
            ckpt_dir=checkpoint_path,
            target=None
        )
        
        if self.params is None:
            self.get_logger().error(f"Failed to load checkpoint from {checkpoint_path}")
            raise RuntimeError("Checkpoint loading failed")
        
        self.get_logger().info("Model loaded successfully!")
        
        # 이미지 전처리기 초기화 (모델 입력 크기인 224x224로 리사이징)
        self.preprocessor = ImagePreprocessor(target_size=(224, 224))
        
        # ROS 이미지와 OpenCV 이미지를 변환하는 브릿지
        self.bridge = CvBridge()
        
        # 멀티스레드 환경에서 데이터 안전을 위한 Lock 및 버퍼 설정
        self.image_lock = Lock()
        self.latest_images = {
            'gripper1': None,
            'gripper2': None,
            'global': None,
        }
        self.image_timestamps = {
            'gripper1': None,
            'gripper2': None,
            'global': None,
        }
        
        # 로봇 고유 상태(관절 위치 등) 저장을 위한 변수 및 Lock
        self.proprio_lock = Lock()
        self.latest_proprio = None
        
        # --- 구독자(Subscribers) 설정 ---
        # 3대의 카메라로부터 이미지 수신
        self.create_subscription(
            Image,
            '/three_bot/camera/left_gripper/image_raw',
            lambda msg: self._image_callback(msg, 'gripper1'),
            10
        )
        self.create_subscription(
            Image,
            '/three_bot/camera/right_gripper/image_raw',
            lambda msg: self._image_callback(msg, 'gripper2'),
            10
        )
        self.create_subscription(
            Image,
            '/three_bot/camera/global/image_raw',
            lambda msg: self._image_callback(msg, 'global'),
            10
        )
        
        # 로봇 내부 상태(관절 정보 등) 수신
        if RobotState is not None:
            self.create_subscription(
                RobotState,
                '/three_bot/control/robot_state',
                self._state_callback,
                10
            )
            self.get_logger().info("Subscribed to /robot_state for proprioception")
        else:
            self.get_logger().warn("RobotState message not available. Using dummy proprioception.")
        
        # --- 발행자(Publisher) 설정 ---
        # 예측된 액션 명령 발행
        self.action_pub = self.create_publisher(
            Float32MultiArray,
            '/three_bot/policy/action',
            10
        )
        
        # 메인 제어 루프 타이머 설정 (예: 20Hz면 0.05초마다 호출)
        timer_period = 1.0 / control_rate
        self.timer = self.create_timer(timer_period, self._control_loop)
        
        # 통계 관리를 위한 변수들
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.last_stats_print = time.time()
        
        self.get_logger().info("Node initialized. Waiting for images...")
    
    def _state_callback(self, msg: 'RobotState'):
        """로봇 상태 피드백 수신 콜백 함수 (Proprioception)."""
        # 10자유도 상태 추출: [위치(x1, y1, x2, y2, z), 속도(vx1, vy1, vx2, vy2, vz)]
        with self.proprio_lock:
            self.latest_proprio = jnp.array([
                msg.position[0], msg.position[1], msg.position[2], msg.position[3], msg.position[4],
                msg.velocity[0], msg.velocity[1], msg.velocity[2], msg.velocity[3], msg.velocity[4]
            ], dtype=jnp.float32)
    
    def _image_callback(self, msg: Image, camera_name: str):
        """카메라 이미지 수신 콜백 함수."""
        # ROS Image 메시지를 OpenCV(numpy) 이미지로 변환
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        
        # 버퍼에 저장 (Lock을 사용하여 제어 루프와 충돌 방지)
        with self.image_lock:
            self.latest_images[camera_name] = img
            self.image_timestamps[camera_name] = msg.header.stamp
    
    def _control_loop(self):
        """메인 제어 루프: 데이터를 모아서 모델 추론을 실행하고 결과를 발행합니다."""
        # 1. 모든 카메라의 이미지가 최소 한 번은 들어왔는지 확인
        with self.image_lock:
            if any(img is None for img in self.latest_images.values()):
                # 아직 초기 데이터가 다 모이지 않음
                return
            
            # 이미지 간의 동기화 확인 (이미지들의 타임스탬프 차이가 100ms 이내여야 함)
            timestamps = [ts.nanosec for ts in self.image_timestamps.values() if ts is not None]
            if timestamps:
                time_span = (max(timestamps) - min(timestamps)) / 1e9
                if time_span > 0.1:  # 100ms 임계값
                    self.get_logger().warn(
                        f"Images not synchronized: {time_span*1000:.1f}ms span"
                    )
            
            # 처리 도중에 이미지가 바뀌지 않도록 복제본 생성
            images = {k: v.copy() for k, v in self.latest_images.items()}
        
        # 2. 이미지 전처리
        observations = {}
        for cam_name, img in images.items():
            # 리사이징 및 정규화
            img_processed = self.preprocessor.preprocess(jnp.array(img))
            # 모델 입력 형태에 맞게 차원 추가: [1, 1, H, W, 3] (배치, 시간, 높이, 너비, 채널)
            observations[cam_name] = img_processed[None, None, ...]
        
        # 3. 모델 추론 실행
        start_time = time.time()
        
        actions = self._predict(observations)
        
        inference_time = (time.time() - start_time) * 1000  # ms 단위 계산
        
        # 4. 결과 발행
        action_msg = Float32MultiArray()
        action_msg.data = actions.tolist()
        self.action_pub.publish(action_msg)
        
        # 추론 성능 통계 업데이트
        self.inference_count += 1
        self.total_inference_time += inference_time
        
        # 5초마다 평균 추론 속도(Hz) 출력
        if time.time() - self.last_stats_print > 5.0:
            avg_time = self.total_inference_time / self.inference_count
            fps = 1000.0 / avg_time
            self.get_logger().info(
                f"Inference: {avg_time:.1f}ms avg ({fps:.1f} Hz), "
                f"Latest: {inference_time:.1f}ms"
            )
            self.last_stats_print = time.time()
    
    def _predict(self, observations: dict) -> np.ndarray:
        """실제 JAX 모델 추론을 수행하는 내부 함수."""
        # 모델 파라미터 구성
        variables = {'params': self.params['params']}
        if 'batch_stats' in self.params:
            variables['batch_stats'] = self.params['batch_stats']
        
        # 로봇 내부 상태(Proprioception) 준비
        with self.proprio_lock:
            if self.latest_proprio is not None:
                proprioception = self.latest_proprio[None, :]  # [1, 10]
            else:
                # 데이터가 없으면 0으로 채움
                proprioception = jnp.zeros((1, 10), dtype=jnp.float32)
        
        # 5. JAX 기반 순전파(Forward Pass) 실행
        actions = self.model.apply(
            variables,
            observations,
            proprioception,
            train=False,
            # 샘플링을 위한 랜덤 키 설정 (확률적 헤드인 경우 필요)
            rngs={'sample': jax.random.PRNGKey(0)}
        )
        
        # 결과를 numpy 배열로 변환 [1, 5] -> [5,]
        actions = np.array(actions[0])
        
        return actions


def main(args=None):
    """메인 엔트리 포인트."""
    rclpy.init(args=args)
    
    try:
        node = OctoInferenceNode()
        rclpy.spin(node) # 노드 실행 유지
    except KeyboardInterrupt:
        pass
    except Exception as e:
        # 심각한 에러 발생 시 로그 출력
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
