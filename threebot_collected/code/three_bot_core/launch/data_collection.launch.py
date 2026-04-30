from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from datetime import datetime

def generate_launch_description():
    # Arguments
    output_dir_arg = DeclareLaunchArgument(
        'output_dir',
        default_value='data/bags',
        description='Directory to save ROS 2 bags'
    )
    
    port_arg = DeclareLaunchArgument(
        'port',
        default_value='/dev/ttyTHS1',
        description='Serial port for STM32'
    )

    # Nodes
    control_mux_node = Node(
        package='three_bot_control',
        executable='control_mux',
        name='control_mux',
        output='screen'
    )

    motor_bridge_node = Node(
        package='three_bot_control',
        executable='motor_bridge',
        name='motor_bridge',
        output='screen',
        parameters=[{
            'port': LaunchConfiguration('port')
        }]
    )

    data_logger_node = Node(
        package='three_bot_logging',
        executable='data_logger',
        name='data_logger',
        output='screen',
        parameters=[{
            'output_dir': LaunchConfiguration('output_dir'),
            'slop': 0.1
        }]
    )

    safety_controller_node = Node(
        package='three_bot_control',
        executable='safety_controller',
        name='safety_controller',
        output='screen',
        parameters=[{
            'heartbeat_timeout': 0.5
        }]
    )

    return LaunchDescription([
        output_dir_arg,
        port_arg,
        control_mux_node,
        motor_bridge_node,
        data_logger_node,
        safety_controller_node
    ])
