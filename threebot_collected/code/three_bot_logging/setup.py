from setuptools import setup

package_name = 'three_bot_logging'

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
    description='Synchronized data logging for Three-Bot RL',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'data_logger = three_bot_logging.data_logger:main',
        ],
    },
)
