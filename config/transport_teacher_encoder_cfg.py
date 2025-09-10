from params_proto import PrefixProto
import math
from config.locomotion_cfg import LocoCfg


class TransportTeacherEncoderCfg(LocoCfg):

    class policy(LocoCfg.policy, cli=False):
        class model(LocoCfg.policy.model, cli=True):
            # model_path = '/home/linchangyi/robotics_projects/tactile_robots/IsaacLabExtensionTemplate/logs/rsl_rl/locotouch_cylinder_transport_teacher/2025-03-22_22-11-20/model_20000.pt'
            model_path = '/home/linchangyi/robotics_projects/tactile_robots/IsaacLabExtensionTemplate/logs/rsl_rl/locotouch_cylinder_transport_teacher/2025-03-24_01-29-12/model_15700.pt'  # adding gait
            model_path = '/home/linchangyi/robotics_projects/tactile_robots/IsaacLabExtensionTemplate/logs/rsl_rl/locotouch_cylinder_transport_teacher/2025-03-25_01-36-58/model_10300.pt'  # encoder_adding_gait_ar_075_cl_015_fs_05
            # model_path = '/home/linchangyi/robotics_projects/tactile_robots/IsaacLabExtensionTemplate/logs/rsl_rl/locotouch_cylinder_transport_teacher/2025-03-25_05-34-43/model_8400.pt'  # encoder_adding_gait_df
            
            class_name = "ActorCriticEncoder"
            actor_flatten_obs_end_idx = -13*6
            actor_encoder_obs_start_idx = -13*6
            actor_encoder_hidden_dims = [256, 128, 64]
            actor_encoder_embedding_dim = 64
            critic_flatten_obs_end_idx = None
            critic_encoder_obs_start_idx = None
            critic_encoder_hidden_dims = None
            critic_encoder_embedding_dim = None
            encoder_activation = "elu"

        class command(LocoCfg.policy.command, cli=False):
            ranges = [[-0.5, 0.5], [-0.25, 0.25], [-math.pi / 4, math.pi / 4]]
            # ranges = [[-0.4, 0.4], [-0.2, 0.2], [-math.pi / 4, math.pi / 4]]

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

            # default object observations
            object_position = [0.0, 0.0, 0.143]
            object_lin_vel = [0.0, 0.0, 0.0]
            object_orientation = [1.0, 0.0, 0.0, 0.0]
            object_ang_vel = [0.0, 0.0, 0.0]

            object_position_scale = 1.0
            object_lin_vel_scale = 0.5
            object_orientation_scale = 1.0
            object_ang_vel_scale = 0.25

        class action(LocoCfg.policy.action, cli=False):
            delta_action_limit = 1.8


    class mocap(PrefixProto, cli=False):
        node_name = 'mocap_robot_object_pose'
        server_address = "192.168.50.2"
        client_address = "192.168.50.7"
        use_multicast = False
        print_level = 0
        frequency = 120
        robot_id = 4
        # object_id = 3  # cuboid
        # object_position_shift = [0.0, 0.0, 0.0]  # cuboid
        object_id = 5  # cylinder
        object_position_shift = [0.0, 0.0, 0.005]  # cylinder
        object_pose_topic = '/object_pose'



