from setuptools import setup
import os
from glob import glob

package_name = 'three_bot_perception'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Worker_Gemini',
    maintainer_email='worker.gemini@novarobotics.com',
    description='Camera perception and pipe-lining for Nova Three-Bot',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_sync = three_bot_perception.camera_sync_node:main',
        ],
    },
)
