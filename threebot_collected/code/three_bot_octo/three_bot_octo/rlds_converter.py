"""ROS 2 to RLDS Dataset Converter.
# ROS 2에서 RLDS 데이터셋으로 변환하는 모듈:
# Worker_Gemini의 데이터 로거가 생성한 ROS 2 bag 파일(.db3)을 
# Octo 모델 학습에 필요한 RLDS(Robot Learning Dataset) 형식으로 변환합니다.

Author: Architect_Sonnet @ Nova Robotics
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from pathlib import Path
from typing import List, Dict, Iterator, Tuple
import cv2

# ROS 2 관련 라이브러리 (오프라인 변환 시 없을 수도 있으므로 예외 처리)
try:
    import rclpy
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    from sensor_msgs.msg import Image
    from std_msgs.msg import Float32MultiArray
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    print("Warning: ROS 2 not available. Converter will not work.")
    ROS_AVAILABLE = False


class ThreeBotRLDS(tfds.core.GeneratorBasedBuilder):
    """RLDS dataset builder for Three-Bot teleoperation data.
    # Three-Bot 원격 조종 데이터를 위한 RLDS 데이터셋 빌더 클래스입니다.
    
    Dataset Format (데이터셋 구조):
    ---------------
    episodes/ (에피소드 단위)
        steps/ (타임스텝 단위)
            observation/ (관측값)
                gripper1_image: [H, W, 3] uint8 (왼쪽 그리퍼 영상)
                gripper2_image: [H, W, 3] uint8 (오른쪽 그리퍼 영상)
                global_image: [H, W, 3] uint8 (전체 시점 영상)
                robot_state: [6] float32 (이전 액션 값 - Proprioception 대용)
            action: [6] float32 (예측할 다음 액션: ΔX₁, ΔY₁, ΔZ₁, ΔX₂, ΔY₂, ΔZ₂)
            reward: float32 (학습용 보상값, BC에서는 보통 0)
            is_terminal: bool (에피소드 종료 여부)
            is_first: bool (에피소드 시작 여부)
            language_instruction: string (언어 명령: 예 "Insert grating")
    """
    
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release for Three-Bot dual 3-DoF grating insertion.',
    }
    
    def _info(self) -> tfds.core.DatasetInfo:
        """Define dataset metadata and features.
        # 데이터셋의 메타데이터와 특징(Features)을 정의합니다.
        """
        return tfds.core.DatasetInfo(
            builder=self,
            description="""
            Three-Bot teleoperation dataset for grating insertion task.
            Contains synchronized multi-camera observations and 6-DoF actions.
            """,
            features=tfds.features.FeaturesDict({
                'episode_id': tfds.features.Text(),
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'gripper1_image': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=tf.uint8,
                            encoding_format='png'
                        ),
                        'gripper2_image': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=tf.uint8,
                            encoding_format='png'
                        ),
                        'global_image': tfds.features.Image(
                            shape=(224, 224, 3),
                            dtype=tf.uint8,
                            encoding_format='png'
                        ),
                        'robot_state': tfds.features.Tensor(
                            shape=(6,),
                            dtype=tf.float32
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(6,),
                        dtype=tf.float32
                    ),
                    'reward': tf.float32,
                    'discount': tf.float32,
                    'is_terminal': tf.bool,
                    'is_first': tf.bool,
                    'language_instruction': tfds.features.Text(),
                })
            }),
            supervised_keys=None,
            homepage='https://github.com/novarobotics/three_bot',
            citation='Nova Robotics Three-Bot Dataset',
        )
    
    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define dataset splits.
        # 데이터셋을 Train(학습)과 Validation(검증) 세트로 분리합니다.
        """
        # 데이터 디렉토리에서 모든 bag 파일(.db3)을 찾습니다.
        data_dir = Path(self.data_dir) if hasattr(self, 'data_dir') else Path('data/bags')
        bag_paths = list(data_dir.glob('**/*.db3'))
        
        if not bag_paths:
            raise ValueError(f"No bag files found in {data_dir}")
        
        print(f"Found {len(bag_paths)} bag files")
        
        # 80%는 학습용, 20%는 검증용으로 나눕니다.
        num_train = int(len(bag_paths) * 0.8)
        train_bags = bag_paths[:num_train]
        val_bags = bag_paths[num_train:]
        
        return {
            'train': self._generate_examples(train_bags),
            'val': self._generate_examples(val_bags),
        }
    
    def _generate_examples(self, bag_paths: List[Path]) -> Iterator[Tuple[str, dict]]:
        """Convert ROS 2 bags to RLDS episodes.
        # 개별 ROS 2 bag 파일을 읽어서 RLDS 에피소드 형태로 변환합니다.
        
        Args:
            bag_paths: bag 파일 경로 리스트
        
        Yields:
            (에피소드 ID, 에피소드 데이터) 튜플
        """
        if not ROS_AVAILABLE:
            raise RuntimeError("ROS 2 is required for dataset conversion")
        
        bridge = CvBridge()
        
        for bag_idx, bag_path in enumerate(bag_paths):
            print(f"Processing bag {bag_idx + 1}/{len(bag_paths)}: {bag_path.name}")
            
            # ROS 2 bag 파일을 열고 순차적으로 읽을 준비를 합니다.
            reader = SequentialReader()
            storage_options = StorageOptions(uri=str(bag_path.parent), storage_id='sqlite3')
            converter_options = ConverterOptions(
                input_serialization_format='cdr',
                output_serialization_format='cdr'
            )
            
            try:
                reader.open(storage_options, converter_options)
            except Exception as e:
                print(f"Error opening bag {bag_path}: {e}")
                continue
            
            # 한 에피소드의 모든 단계(Step)를 담을 리스트
            episode_steps = []
            
            # 현재 시점의 관측 데이터 버퍼
            current_observation = {
                'gripper1': None,
                'gripper2': None,
                'global': None,
                'robot_state': np.zeros(6, dtype=np.float32)
            }
            
            # Bag 파일 내부의 메시지를 하나씩 읽습니다.
            while reader.has_next():
                topic, msg, timestamp = reader.read_next()
                
                # 토픽별로 메시지를 파싱하여 관측 데이터 버퍼에 저장합니다.
                if topic == '/three_bot/camera/gripper1/image_raw':
                    img = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                    img = cv2.resize(img, (224, 224)) # 모델 입력 크기로 조정
                    current_observation['gripper1'] = img
                
                elif topic == '/three_bot/camera/gripper2/image_raw':
                    img = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                    img = cv2.resize(img, (224, 224))
                    current_observation['gripper2'] = img
                
                elif topic == '/three_bot/camera/global/image_raw':
                    img = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                    img = cv2.resize(img, (224, 224))
                    current_observation['global'] = img
                
                elif topic == '/three_bot/teleop/action':
                    # 액션 메시지 수신 (사람의 조종 값)
                    action = np.array(msg.data, dtype=np.float32)
                    
                    # 모든 카메라 영상이 한 번씩은 들어왔는지 확인 후 Step 생성
                    if all(v is not None for k, v in current_observation.items() 
                           if k != 'robot_state'):
                        step = {
                            'observation': {
                                'gripper1_image': current_observation['gripper1'].copy(),
                                'gripper2_image': current_observation['gripper2'].copy(),
                                'global_image': current_observation['global'].copy(),
                                'robot_state': current_observation['robot_state'].copy(),
                            },
                            'action': action,
                            'reward': 0.0,  # 모방 학습용 보상 (기본 0)
                            'discount': 1.0,
                            'is_terminal': False,
                            'is_first': len(episode_steps) == 0,
                            'language_instruction': 'Insert grating into frame', # 고정 명령
                        }
                        episode_steps.append(step)
                        
                        # 중요: 현재 액션을 다음 타임스텝의 'robot_state'로 저장 (Proprioception 기능)
                        current_observation['robot_state'] = action.copy()
            
            # 마지막 Step에 종료 표시를 하고 에피소드를 발행(Yield)합니다.
            if episode_steps:
                episode_steps[-1]['is_terminal'] = True
                
                episode_id = f"episode_{bag_idx:06d}"
                episode_data = {
                    'episode_id': episode_id,
                    'steps': episode_steps,
                }
                
                yield episode_id, episode_data
                print(f"  → {len(episode_steps)} timesteps")
            else:
                print(f"  → Skipped (no valid timesteps)")


def convert_bags_to_rlds(
    bag_dir: str,
    output_dir: str,
    dataset_name: str = 'three_bot_rlds'
):
    """Convert directory of ROS 2 bags to RLDS dataset.
    # ROS 2 bag 파일들이 들어있는 디렉토리를 통째로 RLDS 데이터셋으로 변환합니다.
    """
    print("=" * 80)
    print("ROS 2 to RLDS Conversion")
    print("=" * 80)
    print(f"Input: {bag_dir}")
    print(f"Output: {output_dir}/{dataset_name}")
    
    # 1. TFDS 빌더 엔진 실행
    builder = ThreeBotRLDS(data_dir=bag_dir)
    builder.download_and_prepare(download_dir=output_dir)
    
    # 2. 결과 확인을 위해 데이터셋 로드 테스트
    ds = tfds.load(dataset_name, split='train', data_dir=output_dir)
    
    print("\n" + "=" * 80)
    print("Conversion completed!")
    print(f"Dataset info: {builder.info}")
    print("=" * 80)
    
    return ds


# 명령줄 인터페이스 (CLI)
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert ROS 2 bags to RLDS')
    parser.add_argument('--bag_dir', type=str, required=True,
                       help='ROS 2 bag 파일이 들어있는 디렉토리')
    parser.add_argument('--output_dir', type=str, default='data/rlds',
                       help='RLDS 데이터셋이 저장될 출력 디렉토리')
    parser.add_argument('--dataset_name', type=str, default='three_bot_rlds',
                       help='데이터셋 이름')
    
    args = parser.parse_args()
    
    convert_bags_to_rlds(
        bag_dir=args.bag_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name
    )
