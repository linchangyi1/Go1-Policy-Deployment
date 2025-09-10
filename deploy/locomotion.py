import torch
import time
from config.locomotion_cfg import LocoCfg
from robots.motor import MotorCommand
from robots.robot import Robot
from robots.fsm import OperationMode, OperationModeFunc, FSMCommand, fsm_cmd_to_operation_mode_mapping
from policy.policy import Policy
torch.set_printoptions(sci_mode=False, precision=4)

import rospy
from std_msgs.msg import Float32MultiArray, Int32
import numpy as np


class Locomotion:
    def __init__(self, cfg: LocoCfg):
        rospy.init_node('deploy')
        self._cfg = cfg
        self._robot = Robot(cfg.robot, cfg.device, cfg.dt)
        self._policy = Policy(cfg.policy, cfg.device)
        self._init_buffer()
        self._set_robot_policy_mapping()
        # self._robot.stand_up()

    def _init_buffer(self):
        self._dt = self._cfg.dt
        self._device = self._cfg.device
        self._num_step = 0
        self._last_step_time = time.time()

        policy_cfg = self._cfg.policy

        # velocity commands
        self._velocity_commands = torch.zeros(policy_cfg.command.num_command, device=self._device, requires_grad=False)
        self._velocity_ranges = torch.tensor(np.array(policy_cfg.command.ranges)[:, 1], device=self._device, requires_grad=False)
        self._vel_cmd_sub = rospy.Subscriber(policy_cfg.command.vel_cmd_topic, Float32MultiArray, self._update_vel_cmd_callback, queue_size=1)

        # operation mode
        self._operation_mode = OperationMode.Lying
        self._static_transition_modes = [(OperationMode.Standing, OperationMode.LieDown), (OperationMode.Lying, OperationMode.StandUp)]
        self._dynamic_transition_modes = [OperationMode.Standing, OperationMode.Policy, OperationMode.ZeroCmdPolicy]
        self._policies_modes = [OperationMode.Policy, OperationMode.ZeroCmdPolicy]
        self._fsm_cmd_sub = rospy.Subscriber(policy_cfg.command.fsm_cmd_topic, Int32, self._update_fsm_cmd_callback, queue_size=1)
        self._mode_step_funcs = {}
        for operation_mode in OperationMode:
            fun_name = f"step_{OperationModeFunc[operation_mode.name].value}"
            if not hasattr(self, fun_name) or not callable(getattr(self, fun_name)):
                raise AttributeError(f"Method '{fun_name}' not found in Deploy class.")
            self._mode_step_funcs[operation_mode] = getattr(self, fun_name)

        # observation buffer
        self._term_dims = policy_cfg.obs.dims
        self._term_scales = policy_cfg.obs.scales
        self._term_histories = policy_cfg.obs.history_length
        self._hist_concate_type_ABAB = policy_cfg.obs.history_concatenate_type == 'ABAB'
        self.reset_history_buffer = True
        if self._hist_concate_type_ABAB:
            self._num_single_obs = sum(self._term_dims)
            self._obs_scales = torch.ones(self._num_single_obs, device=self._device, requires_grad=False)
            start_idx = 0
            for dim, scale in zip(self._term_dims, self._term_scales):
                self._obs_scales[start_idx:start_idx + dim] = scale
                start_idx += dim
        else:
            self._term_obs = []
            for dim, hist_len in zip(self._term_dims, self._term_histories):
                self._term_obs.append(torch.zeros(dim*hist_len, device=self._device, requires_grad=False))

        self._num_actor_obs = policy_cfg.obs.num_actor_obs
        self._obs = torch.zeros(self._num_actor_obs, device=self._device, requires_grad=False)
        # observation functions
        self._obs_functions = []
        for term in policy_cfg.obs.terms:
            fun_name = f"get_{term}"
            if term in policy_cfg.obs.deploy_terms:
                if not hasattr(self, fun_name) or not callable(getattr(self, fun_name)):
                    raise AttributeError(f"Method '{fun_name}' not found in Deploy class.")
                self._obs_functions.append(getattr(self, fun_name))
            elif hasattr(self._robot, fun_name):
                self._obs_functions.append(getattr(self._robot, fun_name))
            else:
                raise AttributeError(f"Method '{fun_name}' not found in Robot class.")

        # action
        self._action_clip_range = policy_cfg.action.clip_range
        self._action_scale = policy_cfg.action.action_scale
        self._using_delta_pos_action = policy_cfg.action.using_delta_pos_action
        self._policy_to_sdk_joint_mapping = torch.tensor(policy_cfg.action.policy_to_sdk_joint_mapping, dtype=torch.int, device=self._device)
        self._motor_command = MotorCommand()
        self._policy_kp = policy_cfg.action.kp
        self._policy_kd = policy_cfg.action.kd
        self._last_action = torch.zeros(policy_cfg.action.num_action, device=self._device, requires_grad=False)
        self._delta_action_limit = policy_cfg.action.delta_action_limit

        # debug
        self.log_count = 0
        self.max_log_count = 200
        self._max_delta_action = torch.zeros_like(self._last_action)
        self._min_delta_action = torch.zeros_like(self._last_action)

    def _update_vel_cmd_callback(self, msg):
        command_np = np.array(msg.data)
        # self._velocity_commands[:] = torch.tensor(command_np, device=self._device, requires_grad=False)
        self._velocity_commands[:] = torch.tensor(command_np, device=self._device, requires_grad=False) * self._velocity_ranges

    def _update_fsm_cmd_callback(self, msg):
        new_fsm_cmd = FSMCommand(msg.data)
        new_operation_mode = fsm_cmd_to_operation_mode_mapping(new_fsm_cmd)
        # print("new fsm command: ", new_fsm_cmd)
        # print("new operation mode: ", new_operation_mode)
        new_operation_mode = OperationMode.Standing if new_operation_mode==OperationMode.StandUp and self._operation_mode!=OperationMode.Lying else new_operation_mode
        if new_operation_mode != self._operation_mode and \
            ((self._operation_mode, new_operation_mode) in self._static_transition_modes or \
            (self._operation_mode in self._dynamic_transition_modes and new_operation_mode in self._dynamic_transition_modes)):
            if self._operation_mode not in self._policies_modes and new_operation_mode in self._policies_modes:
                self.reset_history_buffer = True
            self._operation_mode = new_operation_mode
            # print(f"***** FSM Command: {new_fsm_cmd}, Operation Mode: {self._operation_mode} *****")

    def _set_robot_policy_mapping(self):
        # set joint mapping
        sdk_to_policy_joint_mapping = self._policy_to_sdk_joint_mapping.clone()
        for i, j in enumerate(self._policy_to_sdk_joint_mapping):
            sdk_to_policy_joint_mapping[j] = i
        self._robot.set_sdk_to_customize_joint_mapping(sdk_to_policy_joint_mapping)
        # set foot mapping
        sdk_to_policy_foot_mapping = torch.tensor(self._cfg.policy.action.policy_to_sdk_foot_mapping, dtype=torch.int, device=self._device)
        for i, j in enumerate(self._cfg.policy.action.policy_to_sdk_foot_mapping):
            sdk_to_policy_foot_mapping[j] = i
        self._robot.set_sdk_to_customize_foot_mapping(sdk_to_policy_foot_mapping)

    def run(self):
        print("-------- Start Running --------")
        with torch.inference_mode():
            while not rospy.is_shutdown():
                self.step()

    def run_debug(self):
        while (not rospy.is_shutdown()) and self.log_count < self.max_log_count:
            self.step()
            self.log_count += 1

    def step(self):
        self._last_step_time = time.time()
        self._mode_step_funcs[self._operation_mode]()

    def step_stand_up(self):
        self._robot.stand_up()
        self._operation_mode = OperationMode.Standing

    def step_standing(self):
        self._robot.standing()

    def step_lie_down(self):
        self._robot.lie_down()
        self._operation_mode = OperationMode.Lying
    
    def step_lying(self):
        self._robot.lying()

    def step_zero_cmd_policy(self):
        self._velocity_commands[:] = 0.0
        self.step_policy()

    def step_policy(self):
        obs = self._get_observation()

        actions = self._policy(obs).flatten()
        if actions.device != self._device:
            actions = torch.clip(actions, -self._action_clip_range, self._action_clip_range).to(self._device).detach()
        actions *= self._action_scale
        self._last_action[:] = actions
        actions = actions[self._policy_to_sdk_joint_mapping]
        if not self.check_safety(delta_actions=actions):
            return

        # for gathering the max and min delta actions
        # self._max_delta_action = torch.max(self._max_delta_action, actions)
        # self._min_delta_action = torch.min(self._min_delta_action, actions)
        # print("*"*50)
        # # print("Max Delta Pos:\n", self._max_delta_action)
        # # print("Min Delta Pos:\n", self._min_delta_action)
        # print("Desired Delta Pos:\n", actions)

        if self._using_delta_pos_action:
            actions = actions + self._robot._default_joint_pos
        self._motor_command.desired_position[:] = actions
        self._motor_command.desired_velocity[:] = 0.0
        self._motor_command.kp[:] = self._policy_kp
        self._motor_command.kd[:] = self._policy_kd
        self._motor_command.desired_extra_torque[:] = 0.0

        # # standing for debugging
        # self._motor_command.desired_position[:] = self._robot._default_joint_pos
        # self._motor_command.kp[:] = self._cfg.robot.standing_kp
        # self._motor_command.kd[:] = self._cfg.robot.standing_kd
        # print("Desired Joint Pos:\n", actions)

        self._robot.apaply_action(self._motor_command)
        self._num_step += 1
        sleep_time = max(self._dt - (time.time() - self._last_step_time), 0)
        if sleep_time > 0:
            time.sleep(sleep_time)
            # print("sleep time: ", sleep_time)
        else:
            print("* " * 40)
            print(f"Warning !!! : step time {time.time() - self._last_step_time} exceeds dt {self._dt}.")
        # self.log_count += 1

    def _get_observation(self):
        self._robot.update_sensors()
        if self._hist_concate_type_ABAB:
            obs = torch.cat([func() for func in self._obs_functions], dim=-1)
            obs = obs * self._obs_scales
            if not self.reset_history_buffer:
                self._obs[0:-self._num_single_obs] = self._obs[self._num_single_obs:].clone()
                self._obs[-self._num_single_obs:] = obs
            else:
                self._obs[:] = obs.repeat(self._term_histories[0])
                self.reset_history_buffer = False
        else:
            for i, term_obs in enumerate(self._term_obs):
                if not self.reset_history_buffer:
                    term_obs[:-self._term_dims[i]] = term_obs[self._term_dims[i]:].clone()
                    term_obs[-self._term_dims[i]:] = self._obs_functions[i]() * self._term_scales[i]
                else:
                    term_obs[:] = (self._obs_functions[i]() * self._term_scales[i]).repeat(self._term_histories[i])
            if self.reset_history_buffer:
                self.reset_history_buffer = False
            self._obs = torch.cat(self._term_obs, dim=-1)
        
        return self._obs

        # print("*"*50)
        # print("*"*20, self.log_count, "*"*20)
        # get_term_idx = lambda i: sum(term_dim*hist_len for term_dim, hist_len in zip(self._term_dims[:i], self._term_histories[:i]))
        # print("obs: \n", self._obs)
        # print("Velocity commands: \n", self._obs[get_term_idx(0):get_term_idx(1)])
        # print("Base angular velocity: \n", self._obs[get_term_idx(1):get_term_idx(2)])
        # print("Projected gravity: \n", self._obs[get_term_idx(2):get_term_idx(3)])
        # print("Joint position: \n", self._obs[get_term_idx(3):get_term_idx(4)])
        # print("Joint velocity: \n", self._obs[get_term_idx(4):get_term_idx(5)])
        # print("Last action: \n", self._obs[get_term_idx(5):])

    def get_velocity_commands(self):
        return self._velocity_commands
    
    def get_last_action(self):
        if self.reset_history_buffer:
            self._last_action[:] = 0.0
        return self._last_action

    def check_safety(self, delta_actions):
        if torch.any(torch.abs(delta_actions) > self._delta_action_limit):
            self._operation_mode = OperationMode.Standing
            print(f"Warning !!! : delta actions {delta_actions} exceeds limit {self._delta_action_limit}.")
            return False
        return True





if __name__ == "__main__":
    locomotion = Locomotion(LocoCfg)
    locomotion.run()



