"""STM32 Hardware Bridge for Three-Bot.

Converts ROS 2 action commands to Serial/UART packets for STM32 motor controllers.

Protocol:
- Header: 2 bytes (0xAA, 0x55)
- Payload: 6 floats (24 bytes) - [ΔX1, ΔY1, ΔZ_shared, ΔX2, ΔY2, ΔZ_shared]
- Checksum: 1 byte (XOR sum of payload)
- Tail: 2 bytes (0x0D, 0x0A)
Total Packet Size: 29 bytes

Note: Model outputs 5-DoF [X1, Y1, X2, Y2, Z], we expand Z to both units.

Author: Architect_Sonnet @ Nova Robotics
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import serial
import struct
import time
from threading import Lock

class MotorBridgeNode(Node):
    """ROS 2 to STM32 Bridge Node."""
    
    def __init__(self):
        super().__init__('motor_bridge_node')
        
        # Parameters
        self.declare_parameter('port', '/dev/ttyTHS1')  # Default for STM32 (Updated)
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('timeout', 0.1)
        
        port = self.get_parameter('port').value
        baud = self.get_parameter('baudrate').value
        timeout = self.get_parameter('timeout').value
        
        self.get_logger().info(f"Connecting to STM32 on {port} at {baud} baud...")
        
        # --- 시각적 강조 (Visual Emphasis) ---
        print("\n" + "="*30)
        print(f"   🚀  Attempting to Open Serial Port: \033[92m{port}\033[0m")
        print(f"       Baud: {baud} | Timeout: {timeout}")
        print("="*30 + "\n")
        # -------------------------------------

        # Serial connection
        try:
            self.serial = serial.Serial(port, baud, timeout=timeout)
            self.get_logger().info("Serial connection established.")
        except serial.SerialException as e:
            self.get_logger().error(f"Failed to connect to serial port: {e}")
            self.serial = None
        
        # Thread safety for serial port
        self.serial_lock = Lock()
        
        # Subscribe to SAFE actions
        self.create_subscription(
            Float32MultiArray,
            '/three_bot/safe/action',
            self._action_callback,
            10
        )
        
        # Publisher: Virtual Encoder Feedback (RobotState)
        self.state_pub = self.create_publisher(
            RobotState,
            '/three_bot/control/robot_state',
            10
        )
        
        # Virtual Encoder State
        # We track cumulative steps since the motors are open-loop.
        self.cumulative_steps = [0.0] * 5  # [X1, Y1, X2, Y2, Z]
        self.current_velocity = [0.0] * 5  # steps/sec
        self.last_action_time = self.get_clock().now()
        
        # Timer for publishing state (20Hz)
        self.create_timer(0.05, self._publish_state)
        
    def _action_callback(self, msg: Float32MultiArray):
        """Handle incoming action commands and update virtual state."""
        # 시리얼 포트가 연결되어 있지 않거나 열려있지 않으면 함수를 종료합니다.
        if self.serial is None or not self.serial.is_open:
            return
            
        # 수신된 메시지에서 데이터(액션 값들)를 추출합니다.
        actions = msg.data
        # 데이터의 길이가 5개가 아닐 경우(5-DoF가 아닐 경우) 에러 로그를 출력하고 종료합니다.
        if len(actions) != 5:
            self.get_logger().error(f"Invalid action size: {len(actions)}. Expected 5 [X1, Y1, X2, Y2, Z].")
            return
            
        # --- Virtual Encoder Logic (가상 인코더 로직) ---
        # 현재 시간을 가져옵니다.
        now = self.get_clock().now()
        # 이전 액션 시간과의 차이(dt)를 초 단위로 계산합니다.
        dt = (now - self.last_action_time).nanoseconds / 1e9
        
        # dt가 너무 작아서(0에 가까워서) 발생할 수 있는 0으로 나누기 오류나 속도 스파이크를 방지하기 위해 최소값을 0.05초로 설정합니다.
        if dt < 1e-4: 
            dt = 0.05 
        
        # 가상 상태 업데이트
        # actions 변수에는 이번 제어 주기 동안 이동할 델타 스텝(Δsteps) 값이 들어있습니다.
        for i in range(5):
            # i번째 축의 이동량(델타 스텝)을 가져옵니다.
            delta_steps = actions[i]
            
            # 위치(Position): 델타 스텝을 누적하여 현재 위치를 업데이트합니다. (적분)
            self.cumulative_steps[i] += delta_steps
            
            # 속도(Velocity): 이동량을 시간(dt)으로 나누어 현재 속도를 계산합니다.
            self.current_velocity[i] = delta_steps / dt
            
        # 마지막 액션 실행 시간을 현재 시간으로 갱신합니다.
        self.last_action_time = now
        # -----------------------------
        
        # 5-DoF 데이터 패킹: [X1, Y1, X2, Y2, Z]
        # 펌웨어(STM32)는 5개의 float 값 (20 바이트)을 기대합니다.
        
        # 데이터 패킹: 5개의 float 값을 리틀 엔디안('<') 형식으로 바이트로 변환합니다.
        payload = struct.pack('<5f', *actions)
        
        # 체크섬 계산 (페이로드의 모든 바이트를 XOR 연산)
        checksum = 0
        for b in payload:
            checksum ^= b
            
        # 패킷 생성: 헤더(2바이트) + 페이로드(20바이트) + 체크섬(1바이트) + 꼬리(2바이트) = 총 25바이트
        packet = b'\xAA\x55' + payload + bytes([checksum]) + b'\x0D\x0A'
        
        # 패킷 전송
        # 시리얼 포트 사용 시 다른 스레드와의 충돌을 방지하기 위해 락(Lock)을 겁니다.
        with self.serial_lock:
            try:
                # 생성된 패킷을 시리얼 포트로 전송합니다.
                self.serial.write(packet)
            except serial.SerialException as e:
                # 전송 중 에러가 발생하면 에러 로그를 출력합니다.
                self.get_logger().error(f"Serial write failed: {e}")

    def _publish_state(self):
        """Publish the estimated RobotState based on virtual encoder values."""
        msg = RobotState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.robot_id = "three_bot_virtual"
        msg.mode = "TELEOP" 
        msg.estop_active = False 
        msg.error_code = 0
        
        # Populate with Virtual Encoder data
        msg.position = [float(x) for x in self.cumulative_steps]
        msg.velocity = [float(v) for v in self.current_velocity]
        msg.effort = [0.0] * 5 # Open-loop, no effort data
        
        self.state_pub.publish(msg)
    
    def destroy_node(self):
        """Cleanup serial connection."""
        if self.serial and self.serial.is_open:
            self.serial.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    node = MotorBridgeNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
