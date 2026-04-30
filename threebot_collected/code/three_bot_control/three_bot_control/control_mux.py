"""제어 멀티플렉서 (Control MUX) 노드.

텔레오퍼레이션(수동)과 정책(AI) 동작 간의 제어 권한을 전환합니다.
한 번에 하나의 소스만 로봇을 제어하도록 보장합니다.

토픽 (Topics):
  Sub: /three_bot/teleop/action (전문가/수동)
  Sub: /three_bot/policy/action (에이전트/AI)
  Pub: /three_bot/mux/action    (하드웨어 브릿지 및 로거로 전송)

서비스 (Service):
  /three_bot/switch_mode (std_srvs/SetBool)
    - data: True (AI 모드), False (수동 모드)

작성자: Architect_Sonnet @ Nova Robotics
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
from std_srvs.srv import SetBool
from three_bot_core.msg import StampedAction

class ControlMuxNode(Node):
    def __init__(self):
        super().__init__('control_mux_node')
        
        # 상태 (State)
        self.ai_mode = False  # 기본값: 수동 모드 (Manual)
        
        # 구독자 (Subscribers)
        self.create_subscription(
            Float32MultiArray,
            '/three_bot/teleop/action',
            self._teleop_callback,
            10
        )
        self.create_subscription(
            Float32MultiArray,
            '/three_bot/policy/action',
            self._policy_callback,
            10
        )
        
        # 발행자: 최종 액션 -> 모터 브릿지 (Publisher: Final Action -> Motor Bridge)
        self.action_pub = self.create_publisher(
            Float32MultiArray,
            '/three_bot/mux/action',
            10
        )

        # 발행자: 타임스탬프 액션 -> 데이터 로거 (Publisher: Stamped Action -> Data Logger)
        self.stamped_pub = self.create_publisher(
            StampedAction,
            '/three_bot/control/stamped_action',
            10
        )
        
        # 모드 상태 발행자 (Mode Status Publisher)
        self.mode_pub = self.create_publisher(
            Bool,
            '/three_bot/mux/ai_mode',
            10,
        )
        
        # Homing Status Subscriber
        self.homing_complete = False
        self.create_subscription(
            Bool,
            '/three_bot/control/homing_status',
            self._homing_status_callback,
            10
        )
        
        # 모드 전환 서비스 (Service)
        self.create_service(
            SetBool,
            '/three_bot/switch_mode',
            self._switch_mode_callback
        )
        
        # 모드 상태 발행 타이머 (Timer to publish mode status)
        self.create_timer(1.0, self._publish_status)
        
        self.get_logger().info("Control MUX initialized. Default: MANUAL MODE")

    def _homing_status_callback(self, msg):
        self.homing_complete = msg.data

    def _switch_mode_callback(self, request, response):
        """모드 전환 서비스 콜백 함수.
        
        request.data가 True면 AI 모드, False면 수동 모드로 전환합니다.
        """
        self.ai_mode = request.data
        mode_str = "AI (POLICY)" if self.ai_mode else "MANUAL (TELEOP)"
        self.get_logger().warn(f"Switched control mode to: {mode_str}")
        
        response.success = True
        response.message = f"Mode switched to {mode_str}"
        return response

    def _publish_action(self, msg):
        """원본 액션(Raw)과 타임스탬프 액션(Stamped)을 모두 발행합니다."""
        # 1. 모터 브릿지로 전송 (Raw 데이터)
        self.action_pub.publish(msg)
        
        # 2. 데이터 로거로 전송 (Stamped 데이터)
        stamped = StampedAction()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.actions = msg.data
        self.stamped_pub.publish(stamped)

    def _teleop_callback(self, msg):
        """수동 모드일 때만 텔레오퍼레이션 액션을 통과시킵니다."""
        if not self.ai_mode:
            self._publish_action(msg)

    def _policy_callback(self, msg):
        """AI 모드일 때만 정책(Policy) 액션을 통과시킵니다."""
        if self.ai_mode:
            if self.homing_complete:
                self._publish_action(msg)
            else:
                # Optional: log throttling to avoid spam
                pass

    def _publish_status(self):
        """현재 제어 모드 상태를 주기적으로 발행합니다."""
        msg = Bool()
        msg.data = self.ai_mode
        self.mode_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ControlMuxNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
