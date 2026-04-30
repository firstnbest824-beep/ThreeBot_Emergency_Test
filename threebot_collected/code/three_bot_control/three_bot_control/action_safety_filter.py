"""Action Safety Filter for Three-Bot.

Applies safety constraints to action commands:
1. Position limits (workspace boundaries)
2. Z-axis clipping (critical for cable robots)
3. Velocity limits (jerk reduction)

Sits between MUX and Motor Bridge.

Author: Architect_Sonnet @ Nova Robotics
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool
import numpy as np

class ActionSafetyFilter(Node):
    def __init__(self):
        super().__init__('action_safety_filter')
        
        # Safety parameters
        self.declare_parameter('z_min', -0.3)
        self.declare_parameter('z_max', 0.3)
        self.declare_parameter('xy_min', -0.5)
        self.declare_parameter('xy_max', 0.5)
        self.declare_parameter('max_velocity', 0.15)  # m/s per timestep
        self.declare_parameter('enabled', True)
        
        self.z_min = self.get_parameter('z_min').value
        self.z_max = self.get_parameter('z_max').value
        self.xy_min = self.get_parameter('xy_min').value
        self.xy_max = self.get_parameter('xy_max').value
        self.max_vel = self.get_parameter('max_velocity').value
        self.enabled = self.get_parameter('enabled').value
        
        # State
        self.last_action = np.zeros(5)
        self.violation_count = 0
        
        # Subscribers
        self.create_subscription(
            Float32MultiArray,
            '/three_bot/mux/action',
            self._action_callback,
            10
        )
        
        # Publisher (safe actions → motor bridge)
        self.safe_pub = self.create_publisher(
            Float32MultiArray,
            '/three_bot/safe/action',
            10
        )
        
        # Status timer
        self.create_timer(10.0, self._print_stats)
        
        self.get_logger().info("🛡️ Action Safety Filter initialized")
        self.get_logger().info(f"  Z limits: [{self.z_min:.2f}, {self.z_max:.2f}] m")
        self.get_logger().info(f"  XY limits: [{self.xy_min:.2f}, {self.xy_max:.2f}] m")
        self.get_logger().info(f"  Max velocity: {self.max_vel:.2f} m/s")

    def _action_callback(self, msg: Float32MultiArray):
        """Apply safety checks."""
        if not self.enabled:
            # Bypass (debug mode)
            self.safe_pub.publish(msg)
            return
            
        actions = np.array(msg.data)
        if len(actions) != 5:
            self.get_logger().error(f"Invalid action size: expected 5, got {len(actions)}")
            return
        
        x1, y1, x2, y2, z = actions
        original = actions.copy()
        violated = False
        
        # 1. XY workspace limits
        x1_safe = np.clip(x1, self.xy_min, self.xy_max)
        y1_safe = np.clip(y1, self.xy_min, self.xy_max)
        x2_safe = np.clip(x2, self.xy_min, self.xy_max)
        y2_safe = np.clip(y2, self.xy_min, self.xy_max)
        
        if not (np.isclose(x1, x1_safe) and np.isclose(y1, y1_safe) and 
                np.isclose(x2, x2_safe) and np.isclose(y2, y2_safe)):
            violated = True
        
        # 2. Z-axis limits (CRITICAL)
        z_safe = np.clip(z, self.z_min, self.z_max)
        if not np.isclose(z, z_safe):
            violated = True
            if abs(z - z_safe) > 0.01:  # Log only significant violations
                self.get_logger().warn(
                    f"⚠️ Z-axis CLIPPED: {z:.3f} → {z_safe:.3f} m"
                )
        
        safe_actions = np.array([x1_safe, y1_safe, x2_safe, y2_safe, z_safe])
        
        # 3. Velocity limiting
        delta = safe_actions - self.last_action
        delta_norm = np.linalg.norm(delta)
        
        if delta_norm > self.max_vel:
            # Scale down
            safe_actions = self.last_action + delta * (self.max_vel / delta_norm)
            violated = True
        
        if violated:
            self.violation_count += 1
        
        # Update state
        self.last_action = safe_actions
        
        # Publish
        safe_msg = Float32MultiArray()
        safe_msg.data = safe_actions.tolist()
        self.safe_pub.publish(safe_msg)

    def _print_stats(self):
        """Log statistics."""
        if self.violation_count > 0:
            self.get_logger().info(
                f"Safety stats: {self.violation_count} violations in last 10s"
            )
            self.violation_count = 0

def main(args=None):
    rclpy.init(args=args)
    node = ActionSafetyFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
