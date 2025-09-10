import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation="elu", final_layer_activation=None):
        super().__init__()
        layers = []
        activation_func = resolve_nn_activation(activation)
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation_func)
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        if final_layer_activation is not None:
            layers.append(resolve_nn_activation(final_layer_activation))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

    def reset(self, dones=None):
        pass


def generate_mlp_layers(input_dim, hidden_dims, output_dim, activation="elu", final_layer_activation=None):
    layers = []
    activation_func = resolve_nn_activation(activation)
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation_func)
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    if final_layer_activation is not None:
        layers.append(resolve_nn_activation(final_layer_activation))
    
    return nn.Sequential(*layers)


def resolve_nn_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None