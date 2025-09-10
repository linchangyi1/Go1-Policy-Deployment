from dataclasses import dataclass
import torch

@dataclass
class MotorCommand:
    desired_position: torch.Tensor = torch.zeros(12)
    kp: torch.Tensor = torch.zeros(12)
    desired_velocity: torch.Tensor = torch.zeros(12)
    kd: torch.Tensor = torch.zeros(12)
    desired_extra_torque: torch.Tensor = torch.zeros(12)

