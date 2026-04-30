from setuptools import setup

package_name = 'three_bot_octo'

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
    maintainer='Architect_Sonnet',
    maintainer_email='architect.sonnet@novarobotics.com',
    description='Modified Octo model for Three-Bot dual 3-DoF winch control',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'octo_inference_node = three_bot_octo.octo_inference_node:main',
            'octo_trainer = three_bot_octo.training:main',
            'octo_replay_episode = three_bot_octo.replay_episode:main',
        ],
    },
)
