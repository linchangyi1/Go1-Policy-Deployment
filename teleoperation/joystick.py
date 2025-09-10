import rospy
from std_msgs.msg import Float32MultiArray, Int32
import pygame
import numpy as np
from config.locomotion_cfg import LocoCfg
import time
import os


class JoystickTeleoperator():
    def __init__(self, cfg: LocoCfg):
        rospy.init_node(cfg.teleop.joystick.node_name)
        self._cfg = cfg
        self._rate = rospy.Rate(self._cfg.teleop.rate)
        self._init_joystick()

        self.fsm_cmd_publisher = rospy.Publisher(self._cfg.policy.command.fsm_cmd_topic, Int32, queue_size=1)
        self.fsm_cmd_msg = Int32()
        self.fsm_cmd_msg.data = 0
        self.last_fsm_updated_time = time.time()
        self.min_switch_interval = self._cfg.teleop.min_switch_interval

        self.vel_cmd_publisher = rospy.Publisher(self._cfg.policy.command.vel_cmd_topic, Float32MultiArray, queue_size=1)
        self.vel_cmd = np.zeros(self._cfg.policy.command.num_command)
        self.vel_cmd_msg = Float32MultiArray()


    def _init_joystick(self):
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.display.init()
        pygame.joystick.init()
        detected_joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
        if len(detected_joysticks) == 0:
            print('No joystick is detected!')
            exit()
        
        candidate_joysticks = self._cfg.teleop.joystick.feature_names
        joystick_initializd = False
        for joystick in detected_joysticks:
            for i, candidate_joystick in enumerate(candidate_joysticks):
                if candidate_joystick in joystick.get_name():
                    print('The {} joystick is detected.'.format(joystick.get_name()))
                    self.joystick = joystick
                    self.joystick.init()
                    self.button_ids = self._cfg.teleop.joystick.button_ids[i]
                    self.axis_ids = self._cfg.teleop.joystick.axis_ids[i]
                    self.hat_values = self._cfg.teleop.joystick.hat_values[i]
                    joystick_initializd = True
                    break
            if joystick_initializd:
                break
        if not joystick_initializd:
            print('The detected joystick is not in the joystick candidate list! Please update the new joystick candidate in the config file!')
            exit()

    def run(self):
        while not rospy.is_shutdown():
            self.construct_commands()
            self._rate.sleep()

    def construct_commands(self):
        pygame.event.pump()

        # Detect FSM command
        fsm_buttons = [self.joystick.get_button(self.button_ids[i]) for i in range(4)] + [self.joystick.get_hat(0) == self.hat_values[i] for i in range(len(self.hat_values))]
        fsm_buttons = list(map(bool, fsm_buttons))
        if True in fsm_buttons:
            new_fsm = fsm_buttons.index(True)
            if time.time() - self.last_fsm_updated_time > self.min_switch_interval:
                self.fsm_cmd_msg.data = new_fsm
                self.last_fsm_updated_time = time.time()
                self.fsm_cmd_publisher.publish(self.fsm_cmd_msg)
                # print('fsm_cmd: ', self.fsm_cmd_msg.data)

        # Construct the velocity commands
        x = -self.joystick.get_axis(self.axis_ids[0])
        y = -self.joystick.get_axis(self.axis_ids[1])
        yaw = -self.joystick.get_axis(self.axis_ids[3])
        self.vel_cmd[:] = np.array([x, y, yaw])
        self.vel_cmd[abs(self.vel_cmd) < 0.1] = 0
        self.vel_cmd_msg.data = self.vel_cmd.tolist()
        self.vel_cmd_publisher.publish(self.vel_cmd_msg)
        # print('vel_cmd: ', self.vel_cmd_msg.data)

if __name__ == '__main__':
    joystick_controller = JoystickTeleoperator(LocoCfg)
    try:
        joystick_controller.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        pygame.quit()
