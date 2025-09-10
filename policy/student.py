import torch
import torch.nn as nn
from config.transport_student_cfg import TransportStudentCfg
from model.model_generation import generate_model
from model import MLP, RNN, CNN2d, CNN2dHead

class Student(nn.Module):
    def __init__(
        self,
        cfg: TransportStudentCfg.policy.model,
        device: str,
        action_dim: int,
        ):
        super().__init__()
        # parameters
        self.cfg = cfg
        self.device = device
        self.proprioception_dim = self.cfg.proprioception_dim
        self.tactile_signal_dim = self.cfg.tactile_signal_dim
        self.tactile_embedding_dim = self.cfg.tactile_encoder.embedding_dim
        self.action_dim = action_dim

        # pre encoder
        pre_encoder_type = cfg.pre_encoder.model_type
        self.use_pre_encoder = True if "CNN" in pre_encoder_type else (cfg.pre_encoder.hidden_dims is not None)
        if self.use_pre_encoder:
            self.pre_encoder: MLP|RNN|CNN2d|CNN2dHead = generate_model(
                self.tactile_signal_dim,
                cfg.pre_encoder.embedding_dim,
                cfg.pre_encoder).to(self.device)
            print(f"Pre Encoder: {self.pre_encoder}")

        # student encoder
        self.student_encoder: MLP|RNN|CNN2d = generate_model(
            self.tactile_signal_dim if not self.use_pre_encoder else cfg.pre_encoder.embedding_dim,
            self.tactile_embedding_dim,
            cfg.tactile_encoder).to(self.device)
        print(f"Student Encoder: {self.student_encoder}")

        # student backbone
        self.student_backbone: MLP|RNN = generate_model(
            self.proprioception_dim + self.tactile_embedding_dim,
            self.action_dim,
            cfg.student_policy).to(self.device)
        print(f"Student Backbone: {self.student_backbone}")

    def act_inference(self, obs):
        proprioception = obs[:self.proprioception_dim]
        tactile_signal = obs[self.proprioception_dim:]
        tactile_embedding = self.encoder_forward(tactile_signal)
        policy_input = torch.cat((proprioception, tactile_embedding), dim=-1)
        return self.student_backbone(policy_input)

    def encoder_forward(self, tactile_signal, hidden_states=None):
        if self.use_pre_encoder:
            if isinstance(self.pre_encoder, CNN2d) or isinstance(self.pre_encoder, CNN2dHead):
                tactile_signal = tactile_signal.reshape(-1, 2, 17, 13)
                tactile_signal = self.pre_encoder(tactile_signal).reshape(self.tactile_embedding_dim)
            else:
                tactile_signal = self.pre_encoder(tactile_signal)

        return self.student_encoder(tactile_signal, hidden_states)
    
    def load_checkpoint(self, model_path):
        self.load_state_dict(torch.load(model_path, map_location=self.device))

    def reset(self, dones=None):
        if self.use_pre_encoder: self.pre_encoder.reset(dones)
        self.student_encoder.reset(dones)
        self.student_backbone.reset(dones)

