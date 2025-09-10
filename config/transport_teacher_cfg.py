from params_proto import PrefixProto
import math
from config.locomotion_cfg import LocoCfg


class TransportTeacherCfg(LocoCfg):
    class policy(LocoCfg.policy, cli=False):
        class model(LocoCfg.policy.model, cli=False):
            model_path = 'checkpoint/transport_teacher/2025-09-01_21-03-58_model_15000.pt'

        class command(LocoCfg.policy.command, cli=False):
            ranges = [[-0.5, 0.5], [-0.25, 0.25], [-math.pi / 4, math.pi / 4]]

        class obs(LocoCfg.policy.obs, cli=False):
            object_terms = ['object_state']
            terms = LocoCfg.policy.obs.terms + object_terms
            deploy_terms = LocoCfg.policy.obs.deploy_terms + object_terms
            dims = LocoCfg.policy.obs.dims + [13]
            scales = LocoCfg.policy.obs.scales + [1.0]
            history_length = 6
            history_length = [history_length] * len(dims) if isinstance(history_length, int) else history_length
            num_actor_obs = sum([dim * history_length for dim, history_length in zip(dims, history_length)])
            num_critic_obs = num_actor_obs

            # object observation scales
            object_position_scale = 1.0
            object_lin_vel_scale = 0.5
            object_orientation_scale = 1.0
            object_ang_vel_scale = 0.25

            # default object observations
            object_position = [0.0, 0.0, 0.143]
            object_orientation = [1.0, 0.0, 0.0, 0.0]
            object_lin_vel = [0.0, 0.0, 0.0]
            object_ang_vel = [0.0, 0.0, 0.0]
        
        class action(LocoCfg.policy.action, cli=False):
            delta_action_limit = 2.5


    class mocap(PrefixProto, cli=False):
        node_name = 'mocap_robot_object_pose'
        server_address = "192.168.50.2"
        client_address = "192.168.50.7"
        use_multicast = False
        print_level = 0
        frequency = 120
        robot_id = 4
        robot_pose_topic = '/robot_pose'
        object_id = 5  # cylinder
        object_position_shift = [0.0, 0.0, 0.002]  # cylinder
        object_pose_topic = '/object_pose'

