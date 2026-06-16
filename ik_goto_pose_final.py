
import threading
import csv
import os
import time
import math
import argparse
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import String

from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import (
    RobotState,
    MotionPlanRequest,
    Constraints,
    JointConstraint,
    PlanningOptions,
)
from moveit_msgs.action import MoveGroup
from rcl_interfaces.srv import GetParameters



IK_LINK_NAME = "tcp_link"

# ====================================================================
# default pose
# ====================================================================
STANDBY_POSE_NAME = "standby"
PLANNING_GROUP = "arm"

# ====================================================================
# name of topics
# ====================================================================
OBJECT_POSE_TOPIC = "/object_pose"
TASK_RESULT_TOPIC = "/task_result"
START_RECORDING_TOPIC = "/start_recording"
STOP_RECORDING_TOPIC = "/stop_recording"

SHUTDOWN_GRACE_SEC = 2.0
RECORDING_MATCH_TIMEOUT = 5.0
TRIAL_RECORDING_PRE_DELAY = 0.5
TRIAL_RECORDING_POST_DELAY = 0.3


class IKGotoWithPoseSub(Node):
    def __init__(self, max_trials=100, auto_mode=True):
        super().__init__("ik_goto_pose")

        self._node_start_time = time.time()

        self.joint_names = [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ]

        self.current_joint_state = None
        self.latest_pose = None

        self._recording = False
        self._trial_index = 0

        self.auto_mode = auto_mode
        self.max_trials = max_trials
        self.auto_started = False
        self._busy = False
        self._last_processed_pose = None
        self._auto_done_event = threading.Event()

        self._success_count = 0
        self._failed_count = 0

        self._standby_joint_targets = None

        # exort directory
        self.dataset_dir = os.path.expanduser("~/Desktop/datasets")
        self.trajectory_dir = os.path.join(self.dataset_dir, "Trajectory")
        self.joints_dir = os.path.join(self.dataset_dir, "joints")
        os.makedirs(self.trajectory_dir, exist_ok=True)
        os.makedirs(self.joints_dir, exist_ok=True)
        self.log_dir = self.dataset_dir

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # trajectory_goals CSV
        self.traj_log_path = os.path.join(
            self.trajectory_dir, f"trajectory_goals_{timestamp}.csv"
        )
        self.traj_log_file = open(self.traj_log_path, "w", newline="")
        self.traj_writer = csv.writer(self.traj_log_file)
        self.traj_writer.writerow([
            "trial", "time", "unix_timestamp",
            "motion_start_ts", "motion_end_ts",
            "input_x", "input_y", "input_z",
            "target_x", "target_y", "target_z",
            "ik_shoulder_pan", "ik_shoulder_lift", "ik_elbow_flex",
            "ik_wrist_flex", "ik_wrist_roll",
            "result",
        ])

        # joint_states — saved seperately for each trial
        # save example: joints/joint_states_TIMESTAMP/trial_XXXX.csv
        self.joints_session_dir = os.path.join(
            self.joints_dir, f"joint_states_{timestamp}"
        )
        os.makedirs(self.joints_session_dir, exist_ok=True)

        # saves current trial's joint states
        self._current_joint_rows = []

        # trajectory per trial
        self.traj_dir = os.path.join(
            self.trajectory_dir, f"trajectories_{timestamp}"
        )
        os.makedirs(self.traj_dir, exist_ok=True)

        self.get_logger().info(f"데이터셋 경로: {self.dataset_dir}")
        self.get_logger().info(f"Trajectory 폴더: {self.trajectory_dir}")
        self.get_logger().info(f"joints 폴더: {self.joints_dir}")
        self.get_logger().info(f"trial별 trajectory 폴더: {self.traj_dir}")
        self.get_logger().info(
            f"trial별 joint_states 폴더: {self.joints_session_dir}"
        )
        self.get_logger().info(
            f"[INIT] IK link: {IK_LINK_NAME} (URDF tcp_link, MoveIt 자동 처리)"
        )
        self.get_logger().info(
            f"[INIT] 가는 길 + 복귀 모두 MoveGroup 으로 plan & execute "
            f"(multi-waypoint trajectory 저장)"
        )
        self.get_logger().info(
            f"[INIT] 자동 모드={self.auto_mode}, max_trials={self.max_trials}, "
            f"복귀 pose='{STANDBY_POSE_NAME}'"
        )
        self.get_logger().info(
            f"[INIT] 연속 녹화 모드: 세션 전체를 1개 영상 파일로 녹화"
        )

        # ROS Interface
        self.create_subscription(
            JointState, "/joint_states", self.joint_state_callback, 10,
        )
        self.create_subscription(
            Pose, OBJECT_POSE_TOPIC, self.object_pose_callback, 10,
        )

        self.ik_client = self.create_client(GetPositionIK, "/compute_ik")
        self.move_group_client = ActionClient(
            self,
            MoveGroup,
            "/move_action",
        )

        self.result_pub = self.create_publisher(String, TASK_RESULT_TOPIC, 10)

        recording_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.start_recording_pub = self.create_publisher(
            String, START_RECORDING_TOPIC, recording_qos,
        )
        self.stop_recording_pub = self.create_publisher(
            String, STOP_RECORDING_TOPIC, recording_qos,
        )
        self._recording_matched = False
        self._trial_recording_active = False

        if self.auto_mode:
            self.get_logger().info(
                f"[INIT] {OBJECT_POSE_TOPIC} 수신을 기다립니다. "
                f"Isaac Sim Timeline이 Play 상태인지 확인하세요."
            )

    def _now_elapsed(self):
        return time.time() - self._node_start_time

    # ================================================================
    # callbacks
    # ================================================================
    def joint_state_callback(self, msg):
        if len(msg.name) > 0:
            self.current_joint_state = msg
            if self._recording:
                joint_map = dict(zip(msg.name, msg.position))
                elapsed = round(self._now_elapsed(), 4)
                unix_ts = round(time.time(), 4)
                self._current_joint_rows.append([
                    self._trial_index, elapsed, unix_ts,
                    round(joint_map.get("shoulder_pan",  0.0), 6),
                    round(joint_map.get("shoulder_lift", 0.0), 6),
                    round(joint_map.get("elbow_flex",    0.0), 6),
                    round(joint_map.get("wrist_flex",    0.0), 6),
                    round(joint_map.get("wrist_roll",    0.0), 6),
                ])

    def object_pose_callback(self, msg):
        self.latest_pose = msg

        if not self.auto_mode:
            return
        if self._busy:
            return
        if self._trial_index >= self.max_trials:
            return

        if self._last_processed_pose is not None:
            p_new = msg.position
            p_old = self._last_processed_pose
            dx = p_new.x - p_old.x
            dy = p_new.y - p_old.y
            dz = p_new.z - p_old.z
            if (dx * dx + dy * dy + dz * dz) < 1e-8:
                return

        if not self.auto_started:
            self.auto_started = True
            self.get_logger().info(
                f"[AUTO] 자동 모드 시작. 목표 trial 수: {self.max_trials}"
            )

        self._last_processed_pose = msg.position
        self._busy = True
        threading.Thread(target=self._run_trial_safely, daemon=True).start()

    def _run_trial_safely(self):
        try:
            self.on_button_pressed()
        except Exception as e:
            self.get_logger().error(f"[AUTO] trial 실행 중 예외: {e}")
        finally:
            self._busy = False
            if self._trial_index >= self.max_trials:
                self.get_logger().info(
                    f"[AUTO] 목표 trial 수 {self.max_trials}회 완료. 종료합니다."
                )
                self._auto_done_event.set()

    # ================================================================
    # utilities
    # ================================================================
    def wait_future(self, future, timeout_sec=10.0):
        start = time.time()
        while not future.done():
            if time.time() - start > timeout_sec:
                self.get_logger().error("Future timeout!")
                return False
            time.sleep(0.05)
        return True

    def wait_for_joint_state(self):
        self.get_logger().info("Waiting for joint states...")
        while rclpy.ok() and self.current_joint_state is None:
            time.sleep(0.1)

    def convert_input_to_robot_base(self, x, y, z):
        """Isaac world → robot base (Z=180° 회전)."""
        return -x, -y, z

    def publish_result(self, result_text):
        msg = String()
        msg.data = result_text
        self.result_pub.publish(msg)
        self.get_logger().info(f"결과 토픽 발행: {result_text}")

    # ================================================================
    # topic for video recording
    # ================================================================
    def _wait_for_subscriber(self, publisher, topic_name,
                             timeout=RECORDING_MATCH_TIMEOUT):
        start = time.time()
        while time.time() - start < timeout:
            count = publisher.get_subscription_count()
            if count > 0:
                return True
            time.sleep(0.1)
        self.get_logger().warn(
            f"[recording] {topic_name} subscriber 매칭 timeout ({timeout}s)"
        )
        return False

    def publish_recording_start(self):
        if self._recording_matched is False:
            self._wait_for_subscriber(
                self.start_recording_pub, START_RECORDING_TOPIC,
            )
            self._recording_matched = True
        msg = String()
        msg.data = "start"
        self.start_recording_pub.publish(msg)
        self._trial_recording_active = True
        self.get_logger().info(f"[recording] 세션 녹화 시작 (data='{msg.data}')")
        time.sleep(TRIAL_RECORDING_PRE_DELAY)

    def publish_recording_stop(self):
        if not self._trial_recording_active:
            return
        msg = String()
        msg.data = "stop"
        self.stop_recording_pub.publish(msg)
        self._trial_recording_active = False
        self.get_logger().info(f"[recording] 세션 녹화 정지 (data='{msg.data}')")
        time.sleep(TRIAL_RECORDING_POST_DELAY)

    # ================================================================
    # IK call
    # ================================================================
    def compute_ik(self, x, y, z):
        self.get_logger().info(f"IK 입력값: x={x:.4f}  y={y:.4f}  z={z:.4f}")

        if not self.ik_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("IK service not available")
            return None

        req = GetPositionIK.Request()
        req.ik_request.group_name = PLANNING_GROUP
        req.ik_request.ik_link_name = IK_LINK_NAME
        req.ik_request.timeout.sec = 1

        pose = PoseStamped()
        pose.header.frame_id = "base"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        pose.pose.orientation.w = 1.0
        req.ik_request.pose_stamped = pose

        seed = RobotState()
        seed.joint_state = self.current_joint_state
        req.ik_request.robot_state = seed

        future = self.ik_client.call_async(req)
        if not self.wait_future(future):
            return None
        if future.result() is None:
            self.get_logger().error("IK call failed")
            return None

        res = future.result()
        if res.error_code.val != 1:
            self.get_logger().error(f"IK failed: {res.error_code.val}")
            return None

        joint_map = dict(zip(
            res.solution.joint_state.name,
            res.solution.joint_state.position,
        ))
        return [joint_map.get(name, 0.0) for name in self.joint_names]

    # ================================================================
    # search SRDF named pose
    # ================================================================
    def get_named_pose_joints(self, pose_name):
        if self._standby_joint_targets is not None and pose_name == STANDBY_POSE_NAME:
            return self._standby_joint_targets

        client = self.create_client(GetParameters, "/move_group/get_parameters")
        if not client.wait_for_service(timeout_sec=3.0):
            self.get_logger().error("move_group parameter service 없음")
            return None

        req = GetParameters.Request()
        req.names = ["robot_description_semantic"]
        future = client.call_async(req)
        if not self.wait_future(future, timeout_sec=5.0):
            return None

        res = future.result()
        if not res.values:
            self.get_logger().error("SRDF 파라미터 조회 실패")
            return None

        srdf_xml = res.values[0].string_value
        try:
            root = ET.fromstring(srdf_xml)
            for gs in root.findall("group_state"):
                if gs.get("name") == pose_name and gs.get("group") == PLANNING_GROUP:
                    jmap = {}
                    for j in gs.findall("joint"):
                        jmap[j.get("name")] = float(j.get("value"))
                    targets = [jmap.get(n, 0.0) for n in self.joint_names]
                    if pose_name == STANDBY_POSE_NAME:
                        self._standby_joint_targets = targets
                    self.get_logger().info(
                        f"[SRDF] '{pose_name}' pose 로드: "
                        f"{[round(v, 4) for v in targets]}"
                    )
                    return targets
        except ET.ParseError as e:
            self.get_logger().error(f"SRDF 파싱 실패: {e}")
            return None

        self.get_logger().error(
            f"'{pose_name}' pose 를 SRDF 에서 찾을 수 없음 (group='{PLANNING_GROUP}')"
        )
        return None

    # ================================================================
    # MoveGroup plan & execute
    # ================================================================
    def _plan_and_execute_joints(self, targets, suffix="", log_label="모션"):
        if not self.move_group_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f"[{log_label}] move_group action 서버 없음")
            return False, None, None

        goal = MoveGroup.Goal()
        goal.request = MotionPlanRequest()
        goal.request.group_name = PLANNING_GROUP
        goal.request.num_planning_attempts = 5
        goal.request.allowed_planning_time = 5.0
        goal.request.max_velocity_scaling_factor = 0.5
        goal.request.max_acceleration_scaling_factor = 0.5

        start_state = RobotState()
        start_state.joint_state = self.current_joint_state
        goal.request.start_state = start_state

        constraints = Constraints()
        for name, val in zip(self.joint_names, targets):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(val)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        goal.request.goal_constraints.append(constraints)

        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = False
        goal.planning_options.replan = False

        self.get_logger().info(f"[{log_label}] MoveGroup plan & execute 시작...")

        motion_start_ts = time.time()
        start_elapsed = self._now_elapsed()

        future = self.move_group_client.send_goal_async(goal)
        if not self.wait_future(future, timeout_sec=5.0):
            return False, motion_start_ts, time.time()

        gh = future.result()
        if gh is None or not gh.accepted:
            self.get_logger().error(f"[{log_label}] MoveGroup goal 거부됨")
            return False, motion_start_ts, time.time()

        rf = gh.get_result_async()
        if not self.wait_future(rf, timeout_sec=20.0):
            self.get_logger().error(f"[{log_label}] MoveGroup 결과 timeout")
            return False, motion_start_ts, time.time()

        motion_end_ts = time.time()

        result = rf.result()
        if result is None or result.result.error_code.val != 1:
            err = result.result.error_code.val if result else "no result"
            self.get_logger().error(f"[{log_label}] MoveGroup 실패: {err}")
            return False, motion_start_ts, motion_end_ts

        # planned trajectory saved as csv (multi-waypoint pos+vel+acc)
        try:
            planned_traj = result.result.planned_trajectory.joint_trajectory
            if planned_traj.points:
                self._save_trajectory_csv(
                    planned_traj, suffix=suffix,
                    start_elapsed=start_elapsed,
                )
        except Exception as e:
            self.get_logger().warn(f"[{log_label}] trajectory 저장 실패: {e}")

        self.get_logger().info(
            f"[{log_label}] 완료. (구간: {motion_start_ts:.4f} ~ {motion_end_ts:.4f}, "
            f"길이 {motion_end_ts - motion_start_ts:.2f}s)"
        )
        return True, motion_start_ts, motion_end_ts

    def go_to_named_pose(self, pose_name=STANDBY_POSE_NAME):
        targets = self.get_named_pose_joints(pose_name)
        if targets is None:
            self.get_logger().error(f"'{pose_name}' pose 정보 없음 → 복귀 생략")
            return False

        success, _, _ = self._plan_and_execute_joints(
            targets, suffix="_return", log_label=f"복귀-{pose_name}",
        )
        return success

    def _save_trajectory_csv(self, trajectory, suffix="", start_elapsed=None):
        fname = f"trial_{self._trial_index:04d}{suffix}.csv"
        fpath = os.path.join(self.traj_dir, fname)

        if start_elapsed is None:
            start_elapsed = self._now_elapsed()

        joint_names = list(trajectory.joint_names)
        n = len(joint_names)

        header = ["point_idx", "time_from_start_sec", "elapsed_sec"]
        header += [f"pos_{j}" for j in joint_names]
        header += [f"vel_{j}" for j in joint_names]
        header += [f"acc_{j}" for j in joint_names]

        try:
            with open(fpath, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                for idx, pt in enumerate(trajectory.points):
                    t_rel = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
                    elapsed = start_elapsed + t_rel
                    pos = list(pt.positions) if pt.positions else [""] * n
                    vel = list(pt.velocities) if pt.velocities else [""] * n
                    acc = list(pt.accelerations) if pt.accelerations else [""] * n
                    pos = (pos + [""] * n)[:n]
                    vel = (vel + [""] * n)[:n]
                    acc = (acc + [""] * n)[:n]
                    row = [idx, round(t_rel, 6), round(elapsed, 6)]
                    row += [round(v, 6) if isinstance(v, float) else v for v in pos]
                    row += [round(v, 6) if isinstance(v, float) else v for v in vel]
                    row += [round(v, 6) if isinstance(v, float) else v for v in acc]
                    w.writerow(row)
            self.get_logger().info(
                f"[traj] trial {self._trial_index}{suffix} → {fname} "
                f"({len(trajectory.points)} points)"
            )
        except Exception as e:
            self.get_logger().error(f"trajectory CSV 저장 실패: {e}")

    def _flush_joint_states(self):
        """현재 trial 의 joint state buffer 를 trial_XXXX.csv 로 저장 후 비우기.

        standby 복귀 완료 후 호출되어 다음 구조의 파일을 생성:
            joints/joint_states_TIMESTAMP/trial_0001.csv
        """
        if not self._current_joint_rows:
            return

        fname = f"trial_{self._trial_index:04d}.csv"
        fpath = os.path.join(self.joints_session_dir, fname)

        try:
            with open(fpath, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "trial", "elapsed_sec", "unix_timestamp",
                    "shoulder_pan", "shoulder_lift", "elbow_flex",
                    "wrist_flex", "wrist_roll",
                ])
                w.writerows(self._current_joint_rows)
            self.get_logger().info(
                f"[joints] trial {self._trial_index} → {fname} "
                f"({len(self._current_joint_rows)} samples)"
            )
        except Exception as e:
            self.get_logger().error(f"joint states CSV 저장 실패: {e}")
        finally:
            self._current_joint_rows = []
    # ================================================================
    # run main trial
    # --------------------------------------------------------------
    # cube_isaac → base transformation → IK → MoveGroup plan & execute (outbound)
    # → standby pose return→ joint state buffer save
    # ================================================================
    def on_button_pressed(self):
        if self.latest_pose is None:
            self.get_logger().warn(f"아직 {OBJECT_POSE_TOPIC} 수신 전입니다.")
            self.publish_result("failed")
            return

        self._trial_index += 1
        p = self.latest_pose.position

        self.get_logger().info(
            f"[trial {self._trial_index}/{self.max_trials}] 시작! "
            f"cube_isaac=({p.x:.4f},{p.y:.4f},{p.z:.4f})"
        )

        self.wait_for_joint_state()

        
        self._current_joint_rows = []
        self._recording = True

        # Isaac → base transformation
        ik_x, ik_y, ik_z = self.convert_input_to_robot_base(p.x, p.y, p.z)

        self.get_logger().info(
            f"IK 목표 (base): ({ik_x:.4f},{ik_y:.4f},{ik_z:.4f})  "
            f"[ik_link={IK_LINK_NAME}]"
        )

        # IK call
        joints = self.compute_ik(ik_x, ik_y, ik_z)

        if joints is None:
            self.get_logger().error("IK 실패 → 중단")
            self._recording = False
            self._current_joint_rows = []   # when ik fails buffer is not saved
            self._log_traj(p, ik_x, ik_y, ik_z, None, "IK_FAILED")
            self.publish_result("failed")
            self._failed_count += 1
            self._print_progress()
            return

        self.get_logger().info(f"IK 결과: {joints}")

        # Outbound — MoveGroup plan & execute (trial_XXXX.csv)
        success, motion_start_ts, motion_end_ts = self._plan_and_execute_joints(
            joints, suffix="", log_label="가는길",
        )
        result_str = "SUCCESS" if success else "FAILED"

        self.publish_result("success" if success else "failed")

        if success:
            self._success_count += 1
        else:
            self._failed_count += 1

        # log
        self._log_traj(
            p, ik_x, ik_y, ik_z, joints, result_str,
            motion_start_ts=motion_start_ts, motion_end_ts=motion_end_ts,
        )
        self._print_summary(p, ik_x, ik_y, ik_z, joints, result_str)
        self._print_progress()

        # standby return
        if success:
            self.go_to_named_pose(STANDBY_POSE_NAME)

        # trial complete — joint state buffer 저장 후 초기화
        self._recording = False
        self._flush_joint_states()

    # ================================================================
    # logging / export
    # ================================================================
    def _log_traj(self, p, jx, jy, jz, joints, result_str,
                  motion_start_ts=None, motion_end_ts=None):
        now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        unix_ts = round(time.time(), 4)
        j = joints if joints else [None] * 5

        def fmt(v):
            return round(v, 6) if v is not None else ""

        def fmt_ts(v):
            return round(v, 4) if v is not None else ""

        self.traj_writer.writerow([
            self._trial_index, now, unix_ts,
            fmt_ts(motion_start_ts), fmt_ts(motion_end_ts),
            round(p.x, 6), round(p.y, 6), round(p.z, 6),
            round(jx, 6), round(jy, 6), round(jz, 6),
            *[fmt(v) for v in j],
            result_str,
        ])
        self.traj_log_file.flush()

    def _print_summary(self, p, jx, jy, jz, joints, result_str):
        print("\n" + "=" * 62)
        print(f"  [trial {self._trial_index}/{self.max_trials}] 결과: {result_str}")
        print(f"  Isaac Sim    x={p.x:.4f}  y={p.y:.4f}  z={p.z:.4f}")
        print(f"  IK 목표      x={jx:.4f}  y={jy:.4f}  z={jz:.4f}")
        if joints is not None:
            print(f"  {'조인트':<16} {'IK 목표(rad)':>12}  {'(deg)':>8}")
            print(f"  {'-' * 42}")
            for name, val in zip(self.joint_names, joints):
                print(f"  {name:<16} {val:>12.4f}  {math.degrees(val):>7.1f}°")
        print(f"  로그: {self.log_dir}")
        print("=" * 62 + "\n")

    def _print_progress(self):
        total = self._success_count + self._failed_count
        success_rate = (self._success_count / total * 100) if total > 0 else 0.0
        self.get_logger().info(
            f"[누적] 성공: {self._success_count} / 실패: {self._failed_count} "
            f"/ 총: {total} (성공률: {success_rate:.1f}%)"
        )

    def close_logs(self):
        self._recording = False

        #  remaining buffer at end saved as final trial
        if self._current_joint_rows:
            self._flush_joint_states()

        try:
            self.traj_log_file.close()
        except Exception:
            pass

        total = self._success_count + self._failed_count
        success_rate = (self._success_count / total * 100) if total > 0 else 0.0
        print("\n" + "=" * 62)
        print(f"  [최종 결과]")
        print(f"  총 trial 수    : {total}")
        print(f"  성공          : {self._success_count}")
        print(f"  실패          : {self._failed_count}")
        print(f"  성공률        : {success_rate:.1f}%")
        print("=" * 62)
        print(f"\n로그 저장 완료: {self.dataset_dir}")
        print(f"  - Trajectory/{os.path.basename(self.traj_log_path)}")
        print(f"  - joints/{os.path.basename(self.joints_session_dir)}/")
        print(f"  - Trajectory/{os.path.basename(self.traj_dir)}/")


def main():
    parser = argparse.ArgumentParser(
        description="IK Goto with Pose Subscriber (자동 데이터 수집)"
    )
    parser.add_argument(
        "-n", "--num-trials", type=int, default=100,
        help="자동 모드에서 실행할 trial 수 (기본값: 100)",
    )
    parser.add_argument(
        "--manual", action="store_true",
        help="수동 모드로 실행 (엔터 입력으로 trial 진행)",
    )
    args, _ = parser.parse_known_args()

    rclpy.init()
    node = IKGotoWithPoseSub(
        max_trials=args.num_trials,
        auto_mode=not args.manual,
    )

    from rclpy.executors import SingleThreadedExecutor
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    def spin_with_executor():
        try:
            executor.spin()
        except Exception as e:
            node.get_logger().warn(f"spin 종료: {e}")

    t = threading.Thread(target=spin_with_executor, daemon=True)
    t.start()

    if node.auto_mode:
        print(f"\n[AUTO MODE] 자동 시작합니다.")
        print(f"  목표: {node.max_trials} trials")
        print(f"  보정: URDF tcp_link 만 사용 (닫힌 루프 보정 없음)")
        print(f"  실행: 가는 길 + 복귀 모두 MoveGroup plan & execute")
        print(f"  녹화 방식: 세션 전체를 연속 녹화 (1개 파일)")
        print(f"  녹화 시작 토픽: {START_RECORDING_TOPIC}")
        print(f"  녹화 정지 토픽: {STOP_RECORDING_TOPIC}")
        print(f"  (Ctrl+C 로 중단 가능)\n")

        STARTUP_DELAY = 2.0
        print(f"[AUTO] DDS discovery 대기 ({STARTUP_DELAY}s)...")
        time.sleep(STARTUP_DELAY)

        node.publish_recording_start()

        print(f"\n[AUTO MODE] {OBJECT_POSE_TOPIC} 수신 시 자동 실행 시작. "
              f"(Ctrl+C 로 중단)\n")
        try:
            while rclpy.ok() and not node._auto_done_event.is_set():
                node._auto_done_event.wait(timeout=0.5)
        except KeyboardInterrupt:
            print("\n[AUTO] 사용자 중단")
    else:
        print("준비 완료. 엔터를 누르면 현재 물체 위치로 로봇이 이동합니다. "
              "(q + 엔터: 종료)")
        try:
            while rclpy.ok():
                key = input()
                if key.strip().lower() == "q":
                    break
                node.on_button_pressed()
        except KeyboardInterrupt:
            pass

    if node._trial_recording_active:
        print("\n[AUTO] 세션 녹화 종료 중...")
        node.publish_recording_stop()
        time.sleep(SHUTDOWN_GRACE_SEC)

    node.close_logs()

    executor.shutdown()
    t.join(timeout=2.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()