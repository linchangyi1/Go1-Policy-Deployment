from params_proto import PrefixProto
import math


class LocoCfg(PrefixProto, cli=False):
    device = 'cpu'
    dt = 1 / 50.0

    class robot(PrefixProto, cli=False):
        num_joint = 12
        num_leg = 4

        standing_time = 2.0
        standing_kp = 200
        standing_kd = 2.0
        collecting_foot_force_time = 1.0
        _foot_contact_force_ratio = 0.8
        joint_pos_limit_min = [-0.853, -0.676, -2.808, -0.853, -0.676, -2.808, -0.853, -0.676, -2.808, -0.853, -0.676, -2.808]
        joint_pos_limit_max = [0.853, 4.491, -0.878, 0.853, 4.491, -0.878, 0.853, 4.491, -0.878, 0.853, 4.491, -0.878]

        # Modify them based on the training configuration
        default_joint_pos = [0.0, 0.9, -1.8] * num_leg
        default_joint_vel = [0.0] * num_joint

        default_joint_pos[0] = -0.1
        default_joint_pos[3] = 0.1
        default_joint_pos[6] = -0.1
        default_joint_pos[9] = 0.1

        lying_down_joint_pos = [0.35, 1.20, -2.79] * num_leg
        lying_down_joint_pos[0] *= -1.0
        lying_down_joint_pos[6] *= -1.0

        class sdk(PrefixProto, cli=False):
            LowLevel  = 0xff
            HighLevel = 0xee
            IP = '192.168.123.10'
            LocalPort = 8080
            RemotePort = 8007
            PowerProtectLevel = 10

    class policy(PrefixProto, cli=False):
        class model(PrefixProto, cli=False):
            class_name = "ActorCritic"
            actor_hidden_dims = [512, 256, 128]
            critic_hidden_dims = [512, 256, 128]
            activation = 'elu'
            init_noise_std = 1.0
            model_path = 'checkpoint/locomotion/2025-04-07_05-30-35-model_9350.pt'

        class command(PrefixProto, cli=False):
            num_command = 3
            ranges = [[-1.0, 1.0], [-0.5, 0.5], [-math.pi / 2, math.pi / 2]]
            vel_cmd_topic = '/velocity_command'
            fsm_cmd_topic = '/fsm_command'

        class obs(PrefixProto, cli=False):
            terms = ['velocity_commands', 'base_ang_vel', 'projected_gravity', 'joint_pos_rel', 'joint_vel', 'last_action']
            deploy_terms = ["velocity_commands", "last_action"]
            dims = [3, 3, 3, 12, 12, 12]
            scales = [1.0, 0.25, 1.0, 1.0, 0.05, 1.0]

            # history_length = [6] * len(obs_dims)
            history_length = 6
            history_length = [history_length] * len(dims) if isinstance(history_length, int) else history_length
            num_actor_obs = sum([dim * history_length for dim, history_length in zip(dims, history_length)])
            num_critic_obs = num_actor_obs
            history_concatenate_type = 'AAAABB'  # repeat an obs_term for its history_length times, then concatenate the next obs_term
            # history_concatenate_type = 'ABAB'  # concatenate the history of all obs_terms one by one

        class action(PrefixProto, cli=False):
            num_action = 12
            clip_range = 100.0
            action_scale = 0.25
            using_delta_pos_action = True
            kp = 25.0
            kd = 0.5
            policy_to_sdk_joint_mapping = [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
            policy_to_sdk_foot_mapping = [0, 1, 2, 3]
            delta_action_limit = 1.8  # for safety, please modify it based on the estimated max action of your policy


    class teleop(PrefixProto, cli=False):
        joystick_teleop = True  # True for joystick teleop, False for keyboard teleop
        rate = 30
        min_switch_interval = 0.5  # to avoid continuous switch

        class joystick(PrefixProto, cli=False):
            node_name = 'joystick_teleop'
            feature_names = ['DualSense', 'Xbox 360']
            # button
            button_order = ['action_down', 'action_up', 'action_left', 'action_right', 'l1', 'r1', 'l3', 'r3', 'ps_button']
            button_name_order_mapping = {name: i for i, name in enumerate(button_order)}
            button_ids = [[0, 2, 3, 1, 4, 5, 11, 12, 10], 
                          [0, 3, 2, 1, 4, 5, 9, 10, 8]]
            # hat
            hat_order = ['dpad_down', 'dpad_up', 'dpad_left', 'dpad_right']
            hat_name_order_mapping = {name: i for i, name in enumerate(hat_order)}
            hat_values = [[(0, -1), (0, 1), (-1, 0), (1, 0)],
                          [(0, -1), (0, 1), (-1, 0), (1, 0)]]
            # axis
            axis_order = ['left_up_down', 'left_left_right', 'right_up_down', 'right_left_right', 'l2', 'r2']
            axis_name_order_mapping = {name: i for i, name in enumerate(axis_order)}
            axis_ids = [[1, 0, 4, 3, 2, 5],
                        [1, 0, 4, 3, 2, 5]]



