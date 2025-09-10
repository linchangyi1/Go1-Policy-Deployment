import torch.nn as nn
from model import MLP, RNN, CNN2d, CNN2dHead
from params_proto import PrefixProto


class ModelCfg(PrefixProto, cli=False):
    # MLP specific
    model_type: str = "MLP"
    hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"
    final_layer_activation = None

    # RNN specific
    rnn_type = "gru"
    rnn_hidden_size = 256
    rnn_num_layers = 1

    # CNN specific
    cnn_channels = (24, 24, 24)
    cnn_kernel_size = (4, 3, 2)
    cnn_stride = (2, 1, 1)
    cnn_nonlinearity = "relu"
    cnn_padding = None
    cnn_use_maxpool = True
    cnn_normlayer = None


def generate_model(
    input_dim:int,
    output_dim:int,
    cfg:ModelCfg):
    model_type = cfg.model_type
    if model_type == "MLP":
        return MLP(input_dim, cfg.hidden_dims, output_dim, cfg.activation, cfg.final_layer_activation)
    elif model_type == "RNN":
        return RNN(input_dim, cfg.hidden_dims, output_dim, cfg.activation,
                    cfg.rnn_type, cfg.rnn_hidden_size, cfg.rnn_num_layers)
    elif model_type ==  "CNN2d":
        return CNN2d(input_dim, cfg.cnn_channels, cfg.cnn_kernel_size, cfg.cnn_stride, cfg.cnn_padding,
                     cfg.cnn_nonlinearity, cfg.cnn_use_maxpool, cfg.cnn_normlayer)
    elif model_type == "CNN2dHead":
        return CNN2dHead(cfg.img_shape, cfg.cnn_channels, cfg.cnn_kernel_size, cfg.cnn_stride, cfg.cnn_padding,
                         cfg.hidden_dims, output_dim, cfg.cnn_nonlinearity, cfg.cnn_use_maxpool, cfg.cnn_normlayer)
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")



