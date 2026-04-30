import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from three_bot_core.msg import RobotState
from three_bot_core.srv import EmergencyStop
import time

class SafetyController(Node):
    def __init__(self):
        super().__init__('safety_controller')
        
        # Parameters
        self.declare_parameter('heartbeat_timeout', 0.5)
        self.timeout_threshold = self.get_parameter('heartbeat_timeout').value
        
        # State
        self.last_heartbeat_time = self.get_clock().now().nanoseconds / 1e9
        self.estop_active = False
        self.estop_reason = ""
        self.robot_id = "bot_1"
        
        # Publishers
        self.state_pub = self.create_publisher(RobotState, '/three_bot/control/robot_state', 10)
        
        # Subscribers
        self.heartbeat_sub = self.create_subscription(
            Header,
            'teleop_heartbeat',
            self.heartbeat_callback,
            10
        )
        
        # Services
        self.estop_srv = self.create_service(
            EmergencyStop,
            'emergency_stop',
            self.estop_callback
        )
        
        # Timer (100Hz Control Loop)
        self.create_timer(0.01, self.control_loop)
        
        self.get_logger().info("Safety Controller Initialized - Waiting for Heartbeat...")

    def heartbeat_callback(self, msg):
        self.last_heartbeat_time = self.get_clock().now().nanoseconds / 1e9
        # If we were in Heartbeat Lost errors, we might auto-reset or require manual reset?
        # For safety, manual reset is better, but for now let's just log.
        
    def estop_callback(self, request, response):
        self.trigger_estop(request.reason)
        response.success = True
        response.message = f"E-Stop Triggered: {request.reason}"
        return response

    def trigger_estop(self, reason):
        if not self.estop_active:
            self.estop_active = True
            self.estop_reason = reason
            self.get_logger().error(f"🚨 EMERGENCY STOP TRIGGERED: {reason} 🚨")
            # Here we would send 0 command to motors if we had the interface
            
    def control_loop(self):
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # Watchdog Check
        if not self.estop_active:
            if current_time - self.last_heartbeat_time > self.timeout_threshold:
                self.trigger_estop(f"Heartbeat Lost (> {self.timeout_threshold}s)")
        
        # Publish State
        state_msg = RobotState()
        state_msg.header.stamp = self.get_clock().now().to_msg()
        state_msg.robot_id = self.robot_id
        state_msg.estop_active = self.estop_active
        if self.estop_active:
            state_msg.mode = "ERROR"
            state_msg.error_code = 1
        else:
            state_msg.mode = "IDLE"
            state_msg.error_code = 0
            
        self.state_pub.publish(state_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SafetyController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
