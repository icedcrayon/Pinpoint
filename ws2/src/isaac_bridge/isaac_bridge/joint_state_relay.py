#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointStateRelay(Node):
    def __init__(self):
        super().__init__('joint_state_relay_to_isaac')

        self.declare_parameter('input_topic', '/joint_states')
        self.declare_parameter('output_topic', '/isaac_joint_command')
        self.declare_parameter(
            'joint_order',
            [
                'shoulder_pan',
                'shoulder_lift',
                'elbow_flex',
                'wrist_flex',
                'wrist_roll',
                'gripper',
            ],
        )

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.joint_order = list(self.get_parameter('joint_order').value)

        self.sub = self.create_subscription(
            JointState,
            input_topic,
            self.joint_state_callback,
            10
        )

        self.pub = self.create_publisher(JointState, output_topic, 10)

        self.get_logger().info(f'Listening: {input_topic}')
        self.get_logger().info(f'Publishing: {output_topic}')
        self.get_logger().info(f'Joint order: {self.joint_order}')

    def joint_state_callback(self, msg: JointState):
        if not msg.name or not msg.position:
            return

        name_to_pos = {
            name: msg.position[i]
            for i, name in enumerate(msg.name)
            if i < len(msg.position)
        }

        missing = [j for j in self.joint_order if j not in name_to_pos]
        if missing:
            self.get_logger().debug(f'Missing joints: {missing}')
            return

        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = list(self.joint_order)
        out.position = [name_to_pos[j] for j in self.joint_order]
        out.velocity = []
        out.effort = []

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = JointStateRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()