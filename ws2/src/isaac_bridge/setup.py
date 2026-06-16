from setuptools import find_packages, setup

package_name = 'isaac_bridge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Relay joint states to Isaac Sim',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'joint_state_relay = isaac_bridge.joint_state_relay:main',
        ],
    },
)