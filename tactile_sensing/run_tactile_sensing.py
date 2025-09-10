import time
import numpy as np
import serial
import cv2
import argparse
import rospy
from std_msgs.msg import Float32MultiArray

try:
    from tactile_sensing.config import Cfg
except Exception:
    from .config import Cfg


class TactileSensor:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg

        # init serial port
        self.port = self.cfg.pcb.port
        self.baudrate = self.cfg.pcb.baudrate
        self.sensor_serial = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=1.0)
        assert self.sensor_serial.is_open, 'Failed to open COM port!'
        self.cable_num_row = self.cfg.pcb.cable_num_row
        self.cable_num_col = self.cfg.pcb.cable_num_col
        self.received_data_length = 2 * self.cable_num_row * self.cable_num_col

        # init sensor buffer
        self.taxel_num_row = self.cfg.sensor.taxel_num_row
        self.taxel_num_col = self.cfg.sensor.taxel_num_col
        self.raw_signal = np.zeros((self.taxel_num_row, self.taxel_num_col))
        self.delta_raw_signal = np.zeros((self.taxel_num_row, self.taxel_num_col))
        self.pseudo_force_signal = np.zeros((self.taxel_num_row, self.taxel_num_col))
        self.binary_signal = np.zeros((self.taxel_num_row, self.taxel_num_col))
        self.signal_base = self.cfg.sensor.signal_base
        self.signal_min = self.cfg.sensor.signal_min
        self.signal_max = self.cfg.sensor.signal_max
        self.use_log = self.cfg.sensor.use_log
        self.use_reference = self.cfg.sensor.use_reference
        if self.use_reference:
            self.reference = np.zeros((self.taxel_num_row, self.taxel_num_col))
            self.reference_frame_num = self.cfg.sensor.reference_frame_num
            self.reference_colleted_frame_num = 0
        self.contact_signal_threshold = self.cfg.sensor.contact_signal_threshold
        self.contact_force_threshold = self.cfg.sensor.contact_force_threshold

        # ros publisher
        self.discrete_policy = self.cfg.policy.discrete_policy
        if self.discrete_policy:
            self.normalized_signal = np.zeros((self.taxel_num_row, self.taxel_num_col))
            self.discretized_signal = np.zeros((self.taxel_num_row, self.taxel_num_col))
            self.max_delat = np.zeros_like(self.raw_signal)
            self.total_levels = 5
            self.discrete_bin = 1.0 / self.total_levels

        self.pub = self.cfg.ros.publish_signal
        if self.pub:
            self.policy_tactile_pub = rospy.Publisher(self.cfg.ros.policy_tactile_topic, Float32MultiArray, queue_size=1)
            self.policy_tactile_msg = Float32MultiArray()
            # self.processed_tactile_pub = rospy.Publisher(self.cfg.ros.processed_tactile_topic, Float32MultiArray, queue_size=1)
            # self.processed_tactile_msg = Float32MultiArray()

        # init gui buffer
        self.viz = self.cfg.gui.viz
        if self.viz:
            self.window_name = self.cfg.gui.window_name
            self.pixel_per_row = self.cfg.gui.pixel_per_row
            self.pixel_per_col = self.cfg.gui.pixel_per_col
            self.resolution = [self.pixel_per_col * self.taxel_num_col, self.pixel_per_row * self.taxel_num_row] # w, h
            self.contact_scale = self.cfg.gui.contact_scale
            self.viz_data = self.cfg.gui.viz_data
            if self.viz_data:
                self.viz_delta_data = True if self.cfg.gui.viz_data_type == 'delta' else False
                self.font_scale = self.cfg.gui.font_scale
                self.font_thickness = self.cfg.gui.font_thickness
                self.font_color = self.cfg.gui.font_color
        self.detected_key = None
        self.log_frequency = self.cfg.gui.log_frequency
        print("Successfully initialize the tactile sensor!")

    def run(self):
        self.collect_reference()

        # begin main loop
        frame_count = 0
        start_time = time.time()
        while not rospy.is_shutdown():
            if self.read_signal():
                self.process_signal()
            else:
                time.sleep(0.001)
                continue
            if self.pub: self.publish_signal()
            if self.viz: self.visualize_signal()
            if self.log_frequency:
                frame_count += 1
                if frame_count == 30:
                    elapsed = time.time() - start_time
                    avg_fps = 30 / elapsed
                    print(f"[TactileSensor] Avg FPS: {avg_fps:.2f}")
                    frame_count = 0
                    start_time = time.time()

        # release resources
        if self.viz:
            cv2.destroyAllWindows()
        self.sensor_serial.close()

    def collect_reference(self):
        if self.use_reference:
            while not rospy.is_shutdown() and self.reference_colleted_frame_num < self.reference_frame_num:
                if not self.read_signal():
                    time.sleep(0.001)
                    continue
                ratio = 1 / (self.reference_colleted_frame_num + 1)
                # print("raw signal shape: ", self.raw_signal.shape)
                # print("reference shape: ", self.reference.shape)
                self.reference = ratio * self.raw_signal + (1 - ratio) * self.reference
                self.reference_colleted_frame_num += 1

    def read_signal(self):
        # Request readout
        self.sensor_serial.reset_input_buffer() # Remove the confirmation 'w' sent by the sensor
        self.sensor_serial.write('a'.encode('utf-8')) # Request data from the sensor
        # Receive data
        received_string = self.sensor_serial.read(self.received_data_length)
        received_valid_signal = len(received_string) == self.received_data_length
        if received_valid_signal:
            data = np.frombuffer(received_string, dtype=np.uint8).astype(np.uint16)
            # print("[" + ", ".join(map(str, data[1::2])) + "]")
            data = data[0::2] * self.signal_base + data[1::2]
            # data = data.reshape(self.cable_num_row, self.cable_num_col)  # normal order
            data = data.reshape(self.cable_num_col, self.cable_num_row).transpose(1, 0) # inverted order
            self.raw_signal = data[0:self.taxel_num_row, 0:self.taxel_num_col]
        # else:
        #     print("Only got %d values => Drop frame." % len(received_string))
        return received_valid_signal


    def process_signal(self):
        # data processing
        self.delta_raw_signal = self.raw_signal.astype(float) - (self.reference if self.use_reference else self.raw_signal)
        self.delta_raw_signal = np.where(self.delta_raw_signal < 0.0, 0.0, self.delta_raw_signal)
        self.binary_signal = np.where(self.delta_raw_signal >= self.contact_signal_threshold, 1.0, 0.0)

        # useful for calibration/debug
        # self.max_delat[self.delta_raw_signal > self.max_delat] = self.delta_raw_signal[self.delta_raw_signal > self.max_delat]
        # print("\n max_delat:\n", self.max_delat)
        # self.pseudo_force_signal = np.log(self.delta_raw_signal + 1) / np.log(2.0) if self.use_log else self.delta_raw_signal
        # print("pseudo_force_signal: ", self.pseudo_force_signal)

        # only for discrete policy (have not been used in practice)
        if self.discrete_policy:
            # Normalize the signal
            normalized_signal = np.clip(self.delta_raw_signal / (self.signal_max-self.signal_min), 0.0, 1.0)
            valid_normalized_signal = np.where(self.binary_signal, normalized_signal, 0.0) # zero out invalid "taxels"
            if np.any(self.binary_signal):
                valid_values = valid_normalized_signal[self.binary_signal>0.0]
                min_force = valid_values.min()
                max_force = valid_values.max()
                force_range = np.where((max_force-min_force) > 0.0, max_force-min_force, 1) # avoid divide-by-zero
                valid_normalized_signal = (valid_normalized_signal - min_force) / force_range
                valid_normalized_signal = np.clip(valid_normalized_signal, 0.0, 1.0)
            discretized_signal = np.round(valid_normalized_signal / self.discrete_bin) * self.discrete_bin
            self.discretized_signal = np.clip(discretized_signal, 0.0, 1.0)

    def publish_signal(self):
        # for single channel
        # self.policy_tactile_msg.data = data.tolist()

        # for double channel
        if self.discrete_policy:
            self.policy_tactile_msg.data = np.concatenate((self.binary_signal, self.discretized_signal), axis=0).flatten().astype(float).tolist()
        else:
            self.policy_tactile_msg.data = np.concatenate((self.binary_signal, self.binary_signal), axis=0).flatten().astype(float).tolist()
        self.policy_tactile_pub.publish(self.policy_tactile_msg)

        # # discretized signal
        # binary_discretized_signal = np.concatenate((self.binary_signal, self.discretized_signal), axis=0)
        # self.processed_tactile_msg.data = binary_discretized_signal.flatten().astype(float).tolist()
        # self.processed_tactile_pub.publish(self.processed_tactile_msg)

    def visualize_signal(self):
        data = np.where(self.delta_raw_signal >= self.contact_signal_threshold, np.clip((self.delta_raw_signal - self.contact_signal_threshold)/(self.signal_max - self.signal_min), self.contact_scale, 1), 0.0)
        if self.use_log:
            data = np.log(data + 1) / np.log(2.0)
        im = cv2.applyColorMap((data * 255).astype('uint8'), cv2.COLORMAP_DEEPGREEN)
        im = cv2.resize(im, self.resolution, interpolation=cv2.INTER_NEAREST)

        if self.viz_data:
            rows, cols = self.taxel_num_row, self.taxel_num_col
            cell_h = self.resolution[1] // rows
            cell_w = self.resolution[0] // cols

            for r in range(rows):
                for c in range(cols):
                    value = int(self.delta_raw_signal[r, c]) if self.viz_delta_data else self.raw_signal[r, c]
                    text = f"{value}"  # Convert to string
                    pos_x = c * cell_w + cell_w // 4  # Center text horizontally
                    pos_y = r * cell_h + cell_h // 2  # Center text vertically
                    
                    cv2.putText(im, text, (pos_x, pos_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_color, thickness=self.font_thickness, lineType=cv2.LINE_AA)

        cv2.imshow(self.window_name, im)
        self.detected_key = cv2.waitKey(1) & 0xFF
        if self.detected_key == ord('q'):
            print('Detected user termination command.')
            rospy.signal_shutdown("User termination command.")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise TypeError('Boolean value expected.')

def parse_args():
    rospy.init_node('tactile_sensor')
    parser = argparse.ArgumentParser(prog="Tactile")
    parser.add_argument("--pub", type=str2bool, default=True)
    parser.add_argument("--viz", type=str2bool, default=False)
    return parser.parse_args()

def debug_cfg(cfg: Cfg):
    cfg.pcb.cable_num_row = 32
    cfg.pcb.cable_num_col = 32
    cfg.sensor.taxel_num_row = 32
    cfg.sensor.taxel_num_col = 32
    cfg.gui.pixel_per_row = 30
    cfg.gui.pixel_per_col = 30


if __name__ == "__main__":
    args = parse_args()
    Cfg.ros.publish_signal = args.pub
    Cfg.gui.viz = args.viz
    # debug_cfg(Cfg)
    tactile_sensor = TactileSensor(Cfg)
    tactile_sensor.run()
