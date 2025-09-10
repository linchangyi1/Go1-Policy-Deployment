import torch
from config.locomotion_cfg import LocoCfg
from policy.actor_critic import ActorCritic
from policy.actor_critic_encoder import ActorCriticEncoder

class Policy:
    def __init__(self, cfg: LocoCfg.policy, device='cpu'):
        self._cfg = cfg
        self._device = device
        model_name = self._cfg.model.class_name
        if "ActorCritic" in model_name:
            model_class = eval(model_name)
            self._model: ActorCritic | ActorCriticEncoder = model_class(
                self._cfg.obs.num_actor_obs,
                self._cfg.obs.num_critic_obs,
                self._cfg.action.num_action,
                **vars(self._cfg.model)
                )
            loaded_dict = torch.load(self._cfg.model.model_path, map_location=torch.device(self._device))
            print(self._cfg.model.model_path)
            print(self._cfg.model.actor_hidden_dims)
            self._model.load_state_dict(loaded_dict['model_state_dict'])
        elif "Student" in model_name:
            from policy.student import Student
            model_class = eval(model_name)
            self._model: Student = model_class(
                self._cfg.model,
                self._device,
                self._cfg.action.num_action,
                )
            self._model.load_checkpoint(self._cfg.model.model_path)
        
        self._model.eval()


    def __call__(self, observations: torch.Tensor):
        return self._model.act_inference(observations)


