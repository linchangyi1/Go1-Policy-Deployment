import time
from mocap.natnet_client import NatNetClient
from mocap.config import MocapCfg
import queue

import numpy as np
np.set_printoptions(precision=3, suppress=True)
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from tf.transformations import quaternion_from_matrix, quaternion_multiply, quaternion_inverse, quaternion_matrix, rotation_matrix, euler_from_quaternion


class OptiTrack:
    def __init__(self, cfg: MocapCfg):
        self.cfg = cfg
        self.frequency = cfg.frequency
        self.positions = {}
        self.rotations = {}
        self.optitrack_queue = None
        self.time_interval = 1 / self.frequency

        self.is_running = False
        self.streaming_client = NatNetClient()
        self.streaming_client.set_client_address(cfg.client_address)
        self.streaming_client.set_server_address(cfg.server_address)
        self.streaming_client.set_use_multicast(cfg.use_multicast)
        self.streaming_client.rigid_body_listener = self.receive_rigid_body_frame
        self.streaming_client.set_print_level(cfg.print_level)

        self.robot_id, self.object_id = cfg.robot_id, cfg.object_id
        self.robot_position_world, self.object_position_world = np.zeros(3), np.zeros(3)
        self.robot_quaternion_world, self.object_quaternion_world = np.zeros(4), np.zeros(4)
        self.robot_frame, self.object_frame = 0, 0

        self.object_position, self.object_position_prev = np.zeros(3), np.zeros(3)
        self.object_quaternion, self.object_quaternion_prev = np.zeros(4), np.zeros(4)
        self.object_lin_vel, self.object_ang_vel = np.zeros(3), np.zeros(3)
        self.last_update_time = time.time()
        self.first_update = True

        self.object_pose_pub = rospy.Publisher(cfg.object_pose_topic, Odometry, queue_size=1)
        self.object_odom_msg = Odometry()
        self.local_x_90_deg_matrix = rotation_matrix(np.pi/2, [1, 0, 0])

        # add robot publisher
        self.robot_position_prev, self.robot_quaternion_prev = np.zeros(3), np.zeros(4)
        self.robot_lin_vel, self.robot_ang_vel = np.zeros(3), np.zeros(3)
        self.robot_odom_msg = Odometry()
        self.robot_pose_pub = rospy.Publisher(cfg.robot_pose_topic, Odometry, queue_size=1)



    def receive_rigid_body_frame(self, id, position, quaternion, frame_number):
        position = np.array(position)
        quaternion = np.array(quaternion)
        if id == self.robot_id:
            self.robot_frame = frame_number
            self.robot_position_world[:] = position
            self.robot_quaternion_world[:] = quaternion
        elif id == self.object_id:
            self.object_frame = frame_number
            self.object_position_world[:] = position
            self.object_quaternion_world[:] = quaternion
        if id in (self.robot_id, self.object_id) and self.robot_frame == self.object_frame:
            robot_matrix = quaternion_matrix(self.robot_quaternion_world) @ self.local_x_90_deg_matrix
            object_matrix = quaternion_matrix(self.object_quaternion_world) @ self.local_x_90_deg_matrix
            # print("robot_quaternion_world: ", self.robot_quaternion_world)
            # print("robot_matrix: \n", robot_matrix)
            # print("object_matrix: \n", object_matrix)
            self.object_position[:] = robot_matrix[:3, :3].T @ (self.object_position_world - self.robot_position_world)
            self.object_quaternion[:] = quaternion_from_matrix(robot_matrix.T @ object_matrix)
            if self.first_update:
                self.first_update = False
                self.last_update_time = time.time()
            else:
                dt = time.time() - self.last_update_time
                self.last_update_time = time.time()
                # compute linear velocity
                self.object_lin_vel[:] = (self.object_position - self.object_position_prev) / dt
                # compute angular velocity
                q_relative = quaternion_multiply(self.object_quaternion, quaternion_inverse(self.object_quaternion_prev))
                angle = 2 * np.arccos(np.clip(q_relative[3], -1.0, 1.0))  # Ensure numerical stability
                axis = q_relative[:3] / np.linalg.norm(q_relative[:3]) if np.linalg.norm(q_relative[:3]) > 1e-6 else np.zeros(3)
                self.object_ang_vel[:] = (angle / dt) * axis  # Angular velocity
                # for robot
                self.robot_lin_vel[:] = (self.robot_position_world - self.robot_position_prev) / dt
                q_relative_robot = quaternion_multiply(self.robot_quaternion_world, quaternion_inverse(self.robot_quaternion_prev))
                angle_robot = 2 * np.arccos(np.clip(q_relative_robot[3], -1.0, 1.0))
                axis_robot = q_relative_robot[:3] / np.linalg.norm(q_relative_robot[:3]) if np.linalg.norm(q_relative_robot[:3]) > 1e-6 else np.zeros(3)
                self.robot_ang_vel[:] = (angle_robot / dt) * axis_robot

            if (frame_number-4) % 10 == 0:
                print("-------------------")
                # print('robot_position_world: ', self.robot_position_world)
                # print('object_position_world: ', self.object_position_world)
                # print('robot_quaternion_world: ', self.robot_quaternion_world)
                # print('object_quaternion_world: ', self.object_quaternion_world)
                # print("robot_rotation_matrix: \n", quaternion_matrix(self.robot_quaternion_world))
                # print("object_rotation_matrix: \n", quaternion_matrix(self.object_quaternion_world))
                print('object_position:\n', self.object_position)
                print('object_lin_vel:\n', self.object_lin_vel)
                print('object_quaternion:\n', self.object_quaternion)
                # print("object_rotation_matrix: \n", quaternion_matrix(self.object_quaternion))
                print("object euler angles:\n", euler_from_quaternion(self.object_quaternion))
                # print("axis angel:\n", angle)
                print('object_ang_vel:\n', self.object_ang_vel)
            
            self.publish_robot_odometry()
            self.robot_position_prev[:] = self.robot_position_world.copy()
            self.robot_quaternion_prev[:] = self.robot_quaternion_world.copy()

            self.publish_object_odometry()
            self.object_position_prev[:] = self.object_position.copy()
            self.object_quaternion_prev[:] = self.object_quaternion.copy()

    def publish_robot_odometry(self):
        # Header
        self.robot_odom_msg.header.stamp = rospy.Time.now()
        self.robot_odom_msg.header.frame_id = "world_frame"
        # Position
        self.robot_odom_msg.pose.pose.position.x = self.robot_position_world[0]
        self.robot_odom_msg.pose.pose.position.y = self.robot_position_world[1] 
        self.robot_odom_msg.pose.pose.position.z = self.robot_position_world[2]
        # Orientation
        self.robot_odom_msg.pose.pose.orientation = Quaternion(*self.robot_quaternion_world)
        # Linear velocity
        self.robot_odom_msg.twist.twist.linear.x = self.robot_lin_vel[0]
        self.robot_odom_msg.twist.twist.linear.y = self.robot_lin_vel[1]
        self.robot_odom_msg.twist.twist.linear.z = self.robot_lin_vel[2]
        # Angular velocity
        self.robot_odom_msg.twist.twist.angular.x = self.robot_ang_vel[0]
        self.robot_odom_msg.twist.twist.angular.y = self.robot_ang_vel[1]
        self.robot_odom_msg.twist.twist.angular.z = self.robot_ang_vel[2]
        # Publish
        self.robot_pose_pub.publish(self.robot_odom_msg)

    def publish_object_odometry(self):
        # Header
        self.object_odom_msg.header.stamp = rospy.Time.now()
        self.object_odom_msg.header.frame_id = "robot_frame"  # Object is in robot frame
        # Position
        self.object_odom_msg.pose.pose.position.x = self.object_position[0]
        self.object_odom_msg.pose.pose.position.y = self.object_position[1]
        self.object_odom_msg.pose.pose.position.z = self.object_position[2]
        # Orientation
        self.object_odom_msg.pose.pose.orientation = Quaternion(*self.object_quaternion)
        # Linear velocity
        self.object_odom_msg.twist.twist.linear.x = self.object_lin_vel[0]
        self.object_odom_msg.twist.twist.linear.y = self.object_lin_vel[1]
        self.object_odom_msg.twist.twist.linear.z = self.object_lin_vel[2]
        # Angular velocity
        self.object_odom_msg.twist.twist.angular.x = self.object_ang_vel[0]
        self.object_odom_msg.twist.twist.angular.y = self.object_ang_vel[1]
        self.object_odom_msg.twist.twist.angular.z = self.object_ang_vel[2]
        # Publish
        self.object_pose_pub.publish(self.object_odom_msg)

    def start_streaming(self, robot_id = 1):
        is_running = self.streaming_client.run()
        while is_running:
            if robot_id in self.positions:
                print('robot_id', robot_id, 'Last position', self.positions[robot_id], 'rotation', self.rotations[robot_id])
            time.sleep(1 / self.frequency)
    
    def stop_streaming(self):
        self.is_running = False
        self.streaming_client.shutdown()


    def start_streaming_attitude(self, robot_ids = [1], show = False):  
        self.is_running = self.streaming_client.run()
        current_time = time.perf_counter()
        sleep_time = 0
        freq = 0
        sleep_interval = (1/self.frequency)
        last_time = time.perf_counter() - sleep_interval
        while self.is_running and (not rospy.is_shutdown()):
            begin_time = time.perf_counter()
            for robot_id in robot_ids:
                if robot_id in self.positions:
                    robot_position = self.positions[robot_id]
                    robot_rotation = self.rotations[robot_id]
                    current_time = time.perf_counter()
                    begin_delay = current_time - begin_time
                    data = [robot_id, robot_position, robot_rotation, current_time]
                    if self.optitrack_queue != None:
                        self.optitrack_queue.put(data)
                    if show == True:
                        freq = 1/(current_time - last_time)
                        print('robot_id', robot_id, 'Last position', self.positions[robot_id], 'rotation', self.rotations[robot_id], 'time', current_time, 'freq', freq)
                    sleep_time = max(0, (current_time - time.perf_counter() + sleep_interval - begin_delay))
                    
            last_time = current_time

            time.sleep(sleep_time)

    #streaming by multiprocessing
    def start_streaming_attitude_multi(self, robot_ids = [1], show = False):  
        self.is_running = self.streaming_client.run()
        current_time = time.perf_counter()
        sleep_time = 0
        freq = 0
        sleep_interval = (1/self.frequency)
        last_time = time.perf_counter() - sleep_interval
        # begin_delay = 0
        while self.is_running:
            begin_time = time.perf_counter()
            for robot_id in robot_ids:
                if robot_id in self.positions:
                    robot_position = self.positions[robot_id]
                    robot_rotation = self.rotations[robot_id]
                    current_time = time.perf_counter()
                    currenit_data_time = time.time()
                    begin_delay = current_time - begin_time
                    data = [robot_id, robot_position, robot_rotation, currenit_data_time]
                    
                    if self.optitrack_queue != None:
                        try:
                            self.optitrack_queue.put_nowait(data)
                        except queue.Full:
                            try:
                                old_data = self.optitrack_queue.get_nowait()
                                self.optitrack_queue.put_nowait(data)
                            except queue.Empty:
                                continue

                    if show == True:
                        freq = 1/(current_time - last_time)
                        print('robot_id', robot_id, 'Last position', self.positions[robot_id], 'rotation', self.rotations[robot_id], 'time', currenit_data_time, 'freq', freq)

            last_time = current_time
            sleep_time = max(0,sleep_interval + current_time - time.perf_counter())
            time.sleep(sleep_interval)


if __name__ == "__main__":
    rospy.init_node(MocapCfg.node_name)
    robo_pos = OptiTrack(MocapCfg)
    try:
        robo_pos.start_streaming_attitude([3, 4], show=False)
    except (KeyboardInterrupt, rospy.ROSInterruptException):
        print("\nKeyboardInterrupt detected. Stopping streaming...")
    finally:
        robo_pos.stop_streaming()
        rospy.signal_shutdown("Shutting down node due to KeyboardInterrupt")
        time.sleep(1)  # Allow time for the shutdown process to complete

