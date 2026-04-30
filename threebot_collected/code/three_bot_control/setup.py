from setuptools import setup

package_name = 'three_bot_control'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'pyserial'],
    zip_safe=True,
    maintainer='Worker_Gemini',
    maintainer_email='worker.gemini@novarobotics.com',
    description='Control logic for Nova Three-Bot',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'motor_bridge = three_bot_control.motor_bridge:main',
            'teleop_keyboard = three_bot_control.teleop_keyboard:main',
            'control_mux = three_bot_control.control_mux:main',
            'safety_controller = three_bot_control.safety_controller:main',
            'action_safety_filter = three_bot_control.action_safety_filter:main',
        ],
    },
)
