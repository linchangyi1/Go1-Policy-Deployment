from params_proto import PrefixProto
import math
from config.locomotion_cfg import LocoCfg


class TransportStudentCfg(LocoCfg):
    class policy(LocoCfg.policy, cli=False):
        class model(LocoCfg.policy.model, cli=True):
            model_path = 'checkpoint/transport_student/2025-09-02_23-27-14_model_7.pt'

            class_name = "Student"
            proprioception_dim = 270
            tactile_signal_dim = 17*13*2

            class pre_encoder:
                model_type = "MLP"
                hidden_dims = [128, 128, 64]
                embedding_dim = 64
                activation = "elu"
                final_layer_activation = None
                
                model_type = "CNN2dHead"
                hidden_dims = None
                cnn_channels = (24, 24, 24)
                cnn_kernel_size = (4, 3, 2)
                cnn_stride = (2, 1, 1)
                cnn_nonlinearity = "relu"
                cnn_padding = None
                cnn_use_maxpool = True
                cnn_normlayer = None
                img_shape = (2, 17, 13)

            class tactile_encoder:
                model_type = "RNN"
                rnn_num_layers = 1
                rnn_hidden_size = 512
                hidden_dims = [256, 128, 64]
                embedding_dim = 64
                activation = "elu"
                final_layer_activation = None
                rnn_type = "gru"

            class student_policy:
                model_type = "MLP"
                hidden_dims = [512, 256, 128]
                embedding_dim = 64
                activation = "elu"
                final_layer_activation = None

        class command(LocoCfg.policy.command, cli=False):
            ranges = [[-0.5, 0.5], [-0.25, 0.25], [-math.pi / 4, math.pi / 4]]

        class action(LocoCfg.policy.action, cli=False):
            delta_action_limit = 1.8

    class tactile(PrefixProto, cli=False):
        policy_tactile_topic = '/policy_tactile_signal'

