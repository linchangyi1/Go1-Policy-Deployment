from config.locomotion_cfg import LocoCfg
import platform
system_arch = platform.machine()
if system_arch == 'x86_64':
    import unitree_legged_sdk.lib.python.amd64.robot_interface as sdk
elif system_arch == 'aarch64':
    import unitree_legged_sdk.lib.python.arm64.robot_interface as sdk
else:
    raise ImportError("Unsupported architecture: {}".format(system_arch))

import torch
import rospy
from utils.math_utils import quat_rotate_inverse
from robots.motor import MotorCommand
import time

class Robot:
    def __init__(self, cfg: LocoCfg.robot, device='cpu', dt=1/60.0):
        self._cfg = cfg
        self._device = device
        self._dt = dt

        # init sdk
        self._udp = sdk.UDP(self._cfg.sdk.LowLevel, self._cfg.sdk.LocalPort, self._cfg.sdk.IP, self._cfg.sdk.RemotePort)
        self._raw_state = sdk.LowState()
        self._cmd = sdk.LowCmd()
        self._udp.InitCmdData(self._cmd)
        self._safe = sdk.Safety(sdk.LeggedType.Go1)
        self._PowerProtectLevel = self._cfg.sdk.PowerProtectLevel

        # init torso states
        self._base_ang_vel = torch.zeros(3, dtype=torch.float, device=self._device, requires_grad=False)

        self._quaternion = torch.zeros(4, dtype=torch.float, device=self._device, requires_grad=False)
        self._gravity_dirt_vector = torch.tensor([0, 0, -1], dtype=torch.float, device=self._device, requires_grad=False)
        self._projected_gravity = torch.zeros_like(self._gravity_dirt_vector)

        # init joint states
        self._num_joint = self._cfg.num_joint
        self._joint_pos = torch.zeros(self._num_joint, dtype=torch.float, device=self._device, requires_grad=False)
        self._joint_vel = torch.zeros(self._num_joint, dtype=torch.float, device=self._device, requires_grad=False)

        # init foot states
        self._num_leg = self._cfg.num_leg
        self._contact_force_threshold = torch.zeros(self._num_leg, dtype=torch.float, device=self._device, requires_grad=False)
        self._raw_contact_force = torch.zeros(self._num_leg, dtype=torch.float, device=self._device, requires_grad=False)
        self._foot_contact = torch.ones(self._num_leg, dtype=torch.bool, device=self._device, requires_grad=False)
        self._sdk_to_customize_foot_mapping = torch.tensor([i for i in range(self._num_leg)], dtype=torch.int, device=self._device, requires_grad=False)

        # init joint buffer
        self._default_joint_pos = torch.tensor(self._cfg.default_joint_pos, dtype=torch.float, device=self._device, requires_grad=False)
        self._default_joint_vel = torch.tensor(self._cfg.default_joint_vel, dtype=torch.float, device=self._device, requires_grad=False)
        self._joint_pos_limit_min = torch.tensor(self._cfg.joint_pos_limit_min, dtype=torch.float, device=self._device, requires_grad=False)
        self._joint_pos_limit_max = torch.tensor(self._cfg.joint_pos_limit_max, dtype=torch.float, device=self._device, requires_grad=False)
        # self._target_joint_pos = torch.zeroslike(self._joint_pos)
        # self._target_joint_vel = torch.zeroslike(self._joint_vel)
        self._sdk_to_customize_joint_mapping = torch.tensor([i for i in range(self._num_joint)], dtype=torch.int, device=self._device, requires_grad=False)


        # init standing buffer
        self._standing_time = self._cfg.standing_time
        self._standing_kp = self._cfg.standing_kp
        self._standing_kd = self._cfg.standing_kd
        self._collecting_foot_force_time = self._cfg.collecting_foot_force_time
        self._foot_contact_force_ratio = self._cfg._foot_contact_force_ratio
        self._lying_joint_pos = torch.tensor(self._cfg.lying_down_joint_pos, dtype=torch.float, device=self._device, requires_grad=False)


    def stand_up(self):
        # test the communication
        zero_action = MotorCommand(desired_position=self._default_joint_pos,
                                    desired_velocity=torch.zeros_like(self._default_joint_pos),
                                    kp=torch.zeros_like(self._default_joint_pos),
                                    kd=torch.zeros_like(self._default_joint_pos),
                                    desired_extra_torque=torch.zeros_like(self._default_joint_pos),
                                    )
        for _ in range(10):
            self.step(zero_action)
        print("Ready to reset the robot!")
        
        initial_joint_pos = self.joint_pos.clone()
        stand_up_action = MotorCommand(desired_position=initial_joint_pos,
                                       desired_velocity=torch.zeros_like(initial_joint_pos),
                                       kp=torch.ones_like(initial_joint_pos) * self._standing_kp,
                                       kd=torch.ones_like(initial_joint_pos) * self._standing_kd,
                                       desired_extra_torque=torch.zeros_like(initial_joint_pos),
                                       )
        # Stand up, and collect the foot contact forces afterwords
        stand_foot_forces = []
        last_step_time = time.time()
        for t in torch.arange(0, self._standing_time + self._collecting_foot_force_time, self._dt):
            blend_ratio = min(t / self._standing_time, 1)
            desired_joint_pos = blend_ratio * self._default_joint_pos + (1 - blend_ratio) * initial_joint_pos
            stand_up_action.desired_position[:] = desired_joint_pos
            if not rospy.is_shutdown():
                self.step(stand_up_action)
                time.sleep(max(self._dt - (time.time() - last_step_time), 0))
                last_step_time = time.time()
            if t > self._standing_time:
                stand_foot_forces.append(self._raw_contact_force)
        # Calibrate foot force sensors
        if stand_foot_forces:
            stand_foot_forces_tensor = torch.stack(stand_foot_forces)
            mean_foot_forces = torch.mean(stand_foot_forces_tensor, dim=0)
        else:
            mean_foot_forces = torch.zeros_like(self._contact_force_threshold)
        self._contact_force_threshold[:] = mean_foot_forces * self._foot_contact_force_ratio
        self.update_sensors()
        print("Robot has already stood up!")

    def standing(self):
        standing_action = MotorCommand(desired_position=self._default_joint_pos,
                                       desired_velocity=torch.zeros_like(self._default_joint_pos),
                                       kp=torch.ones_like(self._default_joint_pos) * self._standing_kp,
                                       kd=torch.ones_like(self._default_joint_pos) * self._standing_kd,
                                       desired_extra_torque=torch.zeros_like(self._default_joint_pos),
                                       )
        self.step(standing_action)


    def lie_down(self):
        self.update_sensors()
        current_joint_pos = self.joint_pos.clone()
        lying_action = MotorCommand(desired_position=current_joint_pos,
                                       desired_velocity=torch.zeros_like(current_joint_pos),
                                       kp=torch.ones_like(current_joint_pos) * self._standing_kp,
                                       kd=torch.ones_like(current_joint_pos) * self._standing_kd,
                                       desired_extra_torque=torch.zeros_like(current_joint_pos),
                                       )
        lying_time = self._standing_time + 0.0
        last_step_time = time.time()
        for t in torch.arange(0, lying_time, self._dt):
            blend_ratio = min(t / lying_time, 1)
            desired_joint_pos = blend_ratio * self._lying_joint_pos + (1 - blend_ratio) * current_joint_pos
            lying_action.desired_position[:] = desired_joint_pos
            if not rospy.is_shutdown():
                self.step(lying_action)
                time.sleep(max(self._dt - (time.time() - last_step_time), 0))
                last_step_time = time.time()
        print("Robot has already lied down!")

    def lying(self):
        lying_action = MotorCommand(desired_position=self._lying_joint_pos,
                                       desired_velocity=torch.zeros_like(self._lying_joint_pos),
                                       kp=torch.zeros_like(self._lying_joint_pos),
                                       kd=torch.zeros_like(self._lying_joint_pos),
                                       desired_extra_torque=torch.zeros_like(self._lying_joint_pos),
                                       )
        self.step(lying_action)

    def step(self, action: MotorCommand):
        self.apaply_action(action)
        self.update_sensors()

    def apaply_action(self, action: MotorCommand):
        action.desired_position = torch.clamp(action.desired_position, self._joint_pos_limit_min, self._joint_pos_limit_max)
        desired_joint_pos = action.desired_position.cpu().tolist()
        desired_joint_vel = action.desired_velocity.cpu().tolist()
        kp = action.kp.cpu().tolist()
        kd = action.kd.cpu().tolist()
        desired_joint_torque = action.desired_extra_torque.cpu().tolist()
        for motor, q, dq, kp_val, kd_val, tau in zip(self._cmd.motorCmd, desired_joint_pos, desired_joint_vel, kp, kd, desired_joint_torque):
            motor.q = q
            motor.dq = dq
            motor.Kp = kp_val
            motor.Kd = kd_val
            motor.tau = tau
        self._safe.PowerProtect(self._cmd, self._raw_state, self._PowerProtectLevel)
        self._udp.SetSend(self._cmd)
        self._udp.Send()

    def update_sensors(self):
        self._udp.Recv()
        self._udp.GetRecv(self._raw_state)

        self._base_ang_vel[:] = torch.tensor(self._raw_state.imu.gyroscope, dtype=torch.float, device=self._device, requires_grad=False)
        
        self._quaternion[:] = torch.tensor(self._raw_state.imu.quaternion, dtype=torch.float, device=self._device, requires_grad=False)
        self._projected_gravity[:] = quat_rotate_inverse(self._quaternion, self._gravity_dirt_vector)

        self._joint_pos[:] = torch.tensor([self._raw_state.motorState[i].q for i in range(self._num_joint)], dtype=torch.float, device=self._device, requires_grad=False)
        self._joint_vel[:] = torch.tensor([self._raw_state.motorState[i].dq for i in range(self._num_joint)], dtype=torch.float, device=self._device, requires_grad=False)

        self._raw_contact_force[:] = torch.tensor(self._raw_state.footForce, dtype=torch.float, device=self._device, requires_grad=False)
        self._foot_contact[:] = self._raw_contact_force > self._contact_force_threshold

    def set_sdk_to_customize_joint_mapping(self, mapping):
        if len(mapping) != self._num_joint:
            raise ValueError("The length of mapping should be equal to the number of joints!")
        if isinstance(mapping, list):
            mapping = torch.tensor(mapping, dtype=torch.int, device=self._device, requires_grad=False)
        self._sdk_to_customize_joint_mapping[:] = mapping

    def set_sdk_to_customize_foot_mapping(self, mapping):
        if len(mapping) != self._num_leg:
            raise ValueError("The length of mapping should be equal to the number of legs!")
        if isinstance(mapping, list):
            mapping = torch.tensor(mapping, dtype=torch.int, device=self._device, requires_grad=False)
        self._sdk_to_customize_foot_mapping[:] = mapping

    @property
    def base_ang_vel(self):
        return self._base_ang_vel

    @property
    def projected_gravity(self):
        return self._projected_gravity

    @property
    def joint_pos(self):
        return self._joint_pos

    @property
    def joint_vel(self):
        return self._joint_vel

    @property
    def foot_contact(self):
        return self._foot_contact

    def get_base_ang_vel(self):
        return self._base_ang_vel

    def get_projected_gravity(self):
        return self._projected_gravity

    def get_joint_pos(self):
        return self._joint_pos[self._sdk_to_customize_joint_mapping]

    def get_joint_pos_rel(self):
        return (self._joint_pos - self._default_joint_pos)[self._sdk_to_customize_joint_mapping]

    def get_joint_vel(self):
        return self._joint_vel[self._sdk_to_customize_joint_mapping]

    def get_foot_contact(self):
        return self._foot_contact[self._sdk_to_customize_foot_mapping]


if __name__ == "__main__":
    robot = Robot(LocoCfg.robot, LocoCfg.device, LocoCfg.dt)
    zero_action = MotorCommand()
    for _ in range(10):
        robot.step(zero_action)
    print("Ready to reset the robot!")
    robot.stand_up()
    time.sleep(0.2)
    robot.lie_down()
    while not rospy.is_shutdown():
        rospy.sleep(1)

