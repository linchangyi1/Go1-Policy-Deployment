import torch
torch.set_printoptions(sci_mode=False, precision=4)
from deploy.locomotion import Locomotion
from config.transport_student_cfg import TransportStudentCfg
import rospy
from std_msgs.msg import Float32MultiArray


class TransportStudent(Locomotion):
    def __init__(self, cfg: TransportStudentCfg):
        super().__init__(cfg)
        self._cfg = cfg
        
    def _init_buffer(self):
        super()._init_buffer()

        # tactile signal subscription
        self._tactile_signal = torch.zeros(self._cfg.policy.model.tactile_signal_dim, device=self._device, requires_grad=False)
        # self._tactile_scale = self._cfg.policy.model.tactile_scale
        self._tactile_signal_sub = rospy.Subscriber(self._cfg.tactile.policy_tactile_topic, Float32MultiArray, self._update_tactile_signal_callback, queue_size=1)

    def _update_tactile_signal_callback(self, msg:Float32MultiArray):
        self._tactile_signal[:] = torch.tensor(msg.data, device=self._device, requires_grad=False)
        # print("tactile_signal:\n", self._tactile_signal.cpu().numpy())

    def _get_observation(self):
        proprioception = super()._get_observation()
        obs = torch.cat((proprioception, self._tactile_signal), dim=0)
        return obs


    # def run(self):
    #     while not rospy.is_shutdown():
    #         self.step()
    #         self.get_object_state()

    def run_debug(self):
        self._robot.stand_up()
        while (not rospy.is_shutdown()) and self.log_count < self.max_log_count:
            self.step()
            self.log_count += 1


if __name__ == "__main__":
    transport_student = TransportStudent(TransportStudentCfg)
    transport_student.run()
    # transport_teacher.run_debug()
    # i = 0
    # while not rospy.is_shutdown():
    #     if i % 10 == 0:
    #         transport_teacher.get_object_state()
    #     i += 1
    #     rospy.sleep(0.02)










