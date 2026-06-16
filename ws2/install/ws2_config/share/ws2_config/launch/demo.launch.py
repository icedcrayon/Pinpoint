from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_demo_launch

from launch_ros.actions import Node


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("SO101", package_name="ws2_config").to_moveit_configs()

    ld = generate_demo_launch(moveit_config)

    isaac_bridge_node = Node(
        package="isaac_bridge",
        executable="joint_state_relay",
        name="joint_state_relay_to_isaac",
        output="screen",
        parameters=[
            {"input_topic": "/joint_states"},
            {"output_topic": "/isaac_joint_command"},
            {"joint_order": [
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            ]},
        ],
    )

    ld.add_action(isaac_bridge_node)
    return ld