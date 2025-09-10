import torch
torch.set_printoptions(sci_mode=False, precision=4)
from deploy.locomotion import Locomotion
from config.transport_teacher_encoder_cfg import TransportTeacherEncoderCfg
from mocap.config import MocapCfg
import rospy
from nav_msgs.msg import Odometry


class TransportTeacherEncoder(Locomotion):
    def __init__(self, cfg: TransportTeacherEncoderCfg):
        super().__init__(cfg)
        self._cfg = cfg
        self._object_position_shift = torch.tensor(MocapCfg.object_position_shift, device=self._device, requires_grad=False)
        
    def _init_buffer(self):
        super()._init_buffer()

        # object state subscription
        self._object_position = torch.zeros(3, device=self._device, requires_grad=False)
        self._object_position[:] = torch.tensor(self._cfg.policy.obs.object_position, device=self._device, requires_grad=False)
        self._object_position_scale = self._cfg.policy.obs.object_position_scale
        self._object_quat_wxyz = torch.zeros(4, device=self._device, requires_grad=False)
        self._object_quat_wxyz[:] = torch.tensor(self._cfg.policy.obs.object_orientation, device=self._device, requires_grad=False)
        self._object_orientation_scale = self._cfg.policy.obs.object_orientation_scale
        self._object_lin_velocity = torch.zeros(3, device=self._device, requires_grad=False)
        self._object_lin_velocity[:] = torch.tensor(self._cfg.policy.obs.object_lin_vel, device=self._device, requires_grad=False)
        self._object_lin_vel_scale = self._cfg.policy.obs.object_lin_vel_scale
        self._object_ang_velocity = torch.zeros(3, device=self._device, requires_grad=False)
        self._object_ang_velocity[:] = torch.tensor(self._cfg.policy.obs.object_ang_vel, device=self._device, requires_grad=False)
        self._object_ang_vel_scale = self._cfg.policy.obs.object_ang_vel_scale
        self._object_state_scaled = torch.zeros(13, device=self._device, requires_grad=False)
        self.get_object_state()
        self._object_state_sub = rospy.Subscriber(MocapCfg.object_pose_topic, Odometry, self._update_object_state_callback, queue_size=1)

    def _update_object_state_callback(self, msg:Odometry):
        self._object_position[:] = torch.tensor([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z], device=self._device, requires_grad=False) - self._object_position_shift
        self._object_quat_wxyz[:] = torch.tensor([msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z], device=self._device, requires_grad=False)
        self._object_lin_velocity[:] = torch.tensor([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z], device=self._device, requires_grad=False)
        self._object_ang_velocity[:] = torch.tensor([msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z], device=self._device, requires_grad=False)

    def get_object_position(self):
        return self._object_position
    
    def get_object_lin_vel(self):
        return self._object_lin_velocity

    def get_object_orientation(self):
        return self._object_quat_wxyz

    def get_object_ang_vel(self):
        return self._object_ang_velocity
    
    def get_object_state(self):
        self._object_state_scaled[0:3] = self._object_position * self._object_position_scale
        self._object_state_scaled[3:6] = self._object_lin_velocity * self._object_lin_vel_scale
        self._object_state_scaled[6:10] = self._object_quat_wxyz * self._object_orientation_scale
        self._object_state_scaled[10:13] = self._object_ang_velocity * self._object_ang_vel_scale
        return self._object_state_scaled

    def run_debug(self):
        self._robot.stand_up()
        while (not rospy.is_shutdown()) and self.log_count < self.max_log_count:
            self.step()
            self.log_count += 1

if __name__ == "__main__":
    transport_teacher_encoder = TransportTeacherEncoder(TransportTeacherEncoderCfg)
    transport_teacher_encoder.run()

