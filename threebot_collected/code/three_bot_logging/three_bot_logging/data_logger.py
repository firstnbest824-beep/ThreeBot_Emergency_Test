"""Episodic Data Logger for Three-Bot (π_0.6 Compatible).
# Three-Bot을 위한 에피소드 기반 데이터 로거 (π_0.6 호환)
# 학습용 데이터 수집을 위한 핵심 모듈입니다.

Features:
1. Ring Buffer (30s history)
   # 링 버퍼: 최근 30초간의 데이터를 메모리에 계속 기억하고 있습니다.
2. Episodic Recording (Triggered by user)
   # 에피소드 녹화: 사용자가 시작 명령을 내리면 그때부터 데이터를 파일로 저장합니다.
3. Async Labeling UI Integration
   # 비동기 라벨링 UI 통합: 녹화가 끝나면 성공/실패 여부를 묻는 창을 띄웁니다.
4. MCAP + JSON Metadata Storage
   # MCAP + JSON 메타데이터 저장: ROS 2 표준 포맷인 MCAP으로 센서 데이터를, JSON으로 라벨 정보를 저장합니다.
5. RLDS Schema Compliance
   # RLDS 스키마 준수: 구글의 강화학습 데이터셋 표준(RLDS) 구조를 따릅니다.

Author: Architect_Sonnet @ Nova Robotics
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from three_bot_core.msg import StampedAction, RobotState, EpisodeMetadata
import message_filters
# 여러 토픽 메시지의 시간을 동기화(Sync)하기 위한 라이브러리입니다.
from rclpy.serialization import serialize_message
# MCAP 저장을 위해 메시지를 바이트(byte) 형태로 직렬화하는 함수입니다.
import rosbag2_py
# ROS 2 가방(Bag) 파일 기록을 위한 공식 라이브러리입니다.
import json
import os
import shutil
from datetime import datetime
from collections import deque
# 링 버퍼 구현을 위한 데크(deque) 자료구조입니다.
import threading
from .labeling_ui import query_episode_label_async
# 별도 파일에 정의된 비동기 라벨링 함수를 가져옵니다.

class EpisodicDataLogger(Node):
    def __init__(self):
        super().__init__('episodic_data_logger')
        
        # Parameters
        # ROS 2 파라미터를 선언하고 기본값을 설정합니다.
        self.declare_parameter('output_dir', 'data/episodes')
        # 데이터가 저장될 기본 경로
        self.declare_parameter('buffer_duration', 30.0)
        # 녹화 시작 전 데이터를 얼마나 기억할지 (초 단위, 버퍼 길이)
        self.declare_parameter('slop', 0.1)
        # 시간 동기화 허용 오차 (0.1초 내외의 차이는 같은 시간으로 간주)
        
        self.base_dir = self.get_parameter('output_dir').value
        self.buffer_duration = self.get_parameter('buffer_duration').value
        self.slop = self.get_parameter('slop').value
        
        # State
        self.recording = False  # 현재 녹화 중인지 여부
        self.episode_id = None  # 현재 에피소드 ID (예: ep_20240201...)
        self.buffer = deque()   # 링 버퍼: (타임스탬프, 데이터패킷) 튜플을 저장
        self.writer = None      # MCAP 파일 작성기 객체
        
        # Ensure output directory exists
        # 저장 폴더가 없으면 미리 생성합니다.
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Subscribers
        # 5개의 핵심 데이터 소스를 구독합니다.
        # 일반 create_subscription 대신 message_filters를 사용하여 시간 동기화를 준비합니다.
        self.sub_left_gripper = message_filters.Subscriber(
            self, Image, '/three_bot/camera/left_gripper/image_raw')
        self.sub_right_gripper = message_filters.Subscriber(
            self, Image, '/three_bot/camera/right_gripper/image_raw')
        self.sub_global = message_filters.Subscriber(
            self, Image, '/three_bot/camera/global/image_raw')
        # 카메라 3개 (왼쪽, 오른쪽, 전체)
        
        self.sub_action = message_filters.Subscriber(
            self, StampedAction, '/three_bot/control/stamped_action')
        # 제어 명령 (Action)
        
        self.sub_state = message_filters.Subscriber(
            self, RobotState, '/three_bot/control/robot_state')
        # 로봇 상태 (State)
            
        # Synchronizer (5 inputs: 3 Cams + Action + State)
        # 5개의 토픽이 비슷한 시간대에 들어오면 하나의 묶음으로 처리하도록 동기화기를 설정합니다.
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_left_gripper, self.sub_right_gripper, self.sub_global, self.sub_action, self.sub_state],
            queue_size=10,
            slop=self.slop
        )
        self.ts.registerCallback(self.sync_callback)
        # 동기화가 성공할 때마다 sync_callback 함수를 호출합니다.
        
        self.get_logger().info("Episodic Logger Initialized. Press 'ENTER' in terminal to START/STOP episodes.")
        
        # Start input thread
        # 사용자 입력을 받기 위한 별도 스레드를 시작합니다 (메인 ROS 루프 차단 방지).
        self.input_thread = threading.Thread(target=self.input_loop, daemon=True)
        self.input_thread.start()

    def sync_callback(self, img_left, img_right, img_global, action, state):
        """Buffer data and write if recording."""
        # 모든 데이터가 동기화되어 도착했을 때 호출되는 함수입니다.
        
        now = self.get_clock().now().nanoseconds
        # 현재 시간을 나노초 단위로 가져옵니다.
        
        # 데이터 패킷을 딕셔너리 형태로 묶습니다.
        # 토픽 이름을 키로, (메시지 객체, 메시지 타입 문자열)을 값으로 저장합니다.
        data_packet = {
            '/three_bot/camera/left_gripper/image_raw': (img_left, 'sensor_msgs/msg/Image'),
            '/three_bot/camera/right_gripper/image_raw': (img_right, 'sensor_msgs/msg/Image'),
            '/three_bot/camera/global/image_raw': (img_global, 'sensor_msgs/msg/Image'),
            '/three_bot/control/stamped_action': (action, 'three_bot_core/msg/StampedAction'),
            '/three_bot/control/robot_state': (state, 'three_bot_core/msg/RobotState')
        }
        
        # Add to buffer
        # 링 버퍼에 데이터를 추가합니다.
        self.buffer.append((now, data_packet))
        
        # Prune buffer (approximate based on 20Hz)
        # 버퍼 관리: 30초(buffer_duration)보다 오래된 데이터는 버립니다.
        while self.buffer and (now - self.buffer[0][0]) > (self.buffer_duration * 1e9):
            self.buffer.popleft()
            
        # Write to bag if recording
        # 만약 현재 '녹화 중(recording)'이라면, 파일에도 즉시 씁니다.
        if self.recording and self.writer:
            for topic, (msg, _) in data_packet.items():
                self.writer.write(
                    topic,
                    serialize_message(msg), # 메시지를 바이트로 직렬화
                    now
                )

    def start_episode(self):
        """Start recording new episode."""
        # 새로운 에피소드 녹화를 시작하는 함수입니다.
        if self.recording:
            return # 이미 녹화 중이면 무시
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.episode_id = f"ep_{timestamp}"
        ep_dir = os.path.join(self.base_dir, self.episode_id)
        # 에피소드별로 별도의 폴더를 만듭니다 (예: data/episodes/ep_20240201_120000)
        
        # Initialize MCAP writer
        # MCAP 기록기(Writer)를 초기화합니다.
        self.writer = rosbag2_py.SequentialWriter()
        storage_options = rosbag2_py.StorageOptions(
            uri=ep_dir,
            storage_id='mcap' # 저장 포맷은 MCAP 사용
        )
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        self.writer.open(storage_options, converter_options)
        
        # Register topics
        # 파일에 저장될 토픽들의 정보를 등록합니다.
        topics = [
            ('/three_bot/camera/left_gripper/image_raw', 'sensor_msgs/msg/Image'),
            ('/three_bot/camera/right_gripper/image_raw', 'sensor_msgs/msg/Image'),
            ('/three_bot/camera/global/image_raw', 'sensor_msgs/msg/Image'),
            ('/three_bot/control/stamped_action', 'three_bot_core/msg/StampedAction'),
            ('/three_bot/control/robot_state', 'three_bot_core/msg/RobotState')
        ]
        
        for name, type_ in topics:
            self.writer.create_topic(rosbag2_py.TopicMetadata(
                name=name, type=type_, serialization_format='cdr'))
        
        # Dump buffer content first (Pre-roll)
        # 중요: 녹화 버튼을 누르기 *직전*의 30초 데이터(버퍼)를 먼저 파일에 씁니다.
        # (과거 데이터가 있어야 학습 시 문맥을 파악할 수 있기 때문입니다)
        self.get_logger().info(f"Dumping {len(self.buffer)} buffered frames...")
        for t, packet in self.buffer:
            for topic, (msg, _) in packet.items():
                self.writer.write(topic, serialize_message(msg), t)
                
        self.recording = True
        self.episode_start_dt = datetime.now() # 녹화 시작 시간 기록 (길이 측정용)
        self.get_logger().info(f"🔴 RECORDING Episode {self.episode_id}")

    def stop_episode(self):
        """Stop recording and trigger labeling."""
        # 녹화를 중지하고 라벨링 프로세스를 시작하는 함수입니다.
        if not self.recording:
            return
            
        self.recording = False
        self.writer.close() # 파일 닫기 (저장 완료)
        self.writer = None
        self.get_logger().info(f"Checking episode {self.episode_id}...")
        
        # Trigger Async Labeling
        # 사용자에게 성공/실패 여부를 묻는 라벨링 작업을 별도 스레드에서 실행합니다.
        # (녹화 루프를 방해하지 않기 위해 비동기로 처리)
        threading.Thread(target=self.process_metadata, args=(self.episode_id,), daemon=True).start()

    def process_metadata(self, episode_id):
        """Query label and save metadata."""
        # 라벨링 결과를 받아서 메타데이터 파일로 저장하는 함수입니다.
        self.get_logger().info("Waiting for user label...")
        
        # labeling_ui.py의 함수를 호출해 사용자 입력을 기다립니다 (최대 30초).
        metadata = query_episode_label_async(episode_id, timeout=30.0)
        
        ep_path = os.path.join(self.base_dir, episode_id)
        
        # 사용자가 'Discard(버림)'를 선택하거나 타임아웃 되면 해당 에피소드 폴더를 삭제합니다.
        if metadata is None:
            self.get_logger().warn(f"Discarding episode {episode_id}")
            shutil.rmtree(ep_path)
            return

        # Calculate metrics
        # 에피소드 길이(duration)를 계산합니다.
        duration = (datetime.now() - self.episode_start_dt).total_seconds()
        
        # Inject machine info
        # 최종 메타데이터 정보를 구성합니다.
        full_metadata = {
            'episode_id': episode_id,
            'timestamp': datetime.now().isoformat(),
            'task_name': 'grating_insertion',   # 태스크 이름 (학습용)
            'duration': duration,               # 수행 시간
            'robot_config': '5-DoF Shared-Z',   # 하드웨어 설정 정보
            **metadata                          # 사용자가 입력한 성공/실패 라벨
        }
        
        # Save JSON
        # metadata.json 파일로 저장합니다.
        json_path = os.path.join(ep_path, 'metadata.json')
        with open(json_path, 'w') as f:
            json.dump(full_metadata, f, indent=2)
            
        self.get_logger().info(f"✅ Saved metadata for {episode_id}")

    def input_loop(self):
        """Simple terminal control."""
        # 터미널 창에서 엔터키 입력을 감시하는 함수입니다.
        while rclpy.ok():
            try:
                cmd = input() # 사용자가 엔터를 칠 때까지 기다립니다 (블로킹).
                if cmd == '':
                    # 엔터키가 입력되었을 때:
                    if not self.recording:
                        self.start_episode() # 녹화 대기 중이면 -> 시작
                    else:
                        self.stop_episode()  # 녹화 중이면 -> 중지
            except EOFError:
                break

def main(args=None):
    rclpy.init(args=args)
    node = EpisodicDataLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
