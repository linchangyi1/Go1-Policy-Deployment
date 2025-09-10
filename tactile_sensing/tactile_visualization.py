import rospy
import numpy as np
import cv2
from std_msgs.msg import Float32MultiArray
import os


class TactileVisualizer:
    def __init__(
        self,
        policy_tactile_topic="/policy_tactile_signal",
        original_tactile_topic="/original_tactile_signal",
        processed_tactile_topic="/processed_tactile_signal",
        row=17,
        column=13):
        self.row = row
        self.column = column
        self.data_policy = None
        self.data_original = None
        self.data_processed = None
        self.image_policy = None
        self.image_original = None
        self.image_processed = None
        rospy.Subscriber(policy_tactile_topic, Float32MultiArray, self.tactile_signal_callback)
        rospy.Subscriber(original_tactile_topic, Float32MultiArray, self.original_tactile_signal_callback)
        rospy.Subscriber(processed_tactile_topic, Float32MultiArray, self.proccessed_tactile_signal_callback)

        # saveing
        self.saved_folder = 'experiment/tactile_image_sim2real/'
        experiment_name = 'yaw_30_degree'
        self.saved_folder = os.path.join(self.saved_folder, experiment_name)

        self.category = 'real'
        # category = 'compact'
        # category = 'overlap'
        # category = 'real_horizon'
        # category = 'compact_horizon'
        # category = 'overlap_horizon'
        self.saved_folder = os.path.join(self.saved_folder, self.category)

        # create folder if not exist
        if not os.path.exists(self.saved_folder):
            os.makedirs(self.saved_folder)
        self.save_id = 1

    def tactile_signal_callback(self, msg):
        self.data_policy = np.array(msg.data, dtype=np.float32).reshape(-1, self.row, self.column)
        self.image_policy = self._generate_heatmap(self.data_policy)

    def original_tactile_signal_callback(self, msg):
        self.data_original = np.array(msg.data, dtype=np.float32).reshape(-1, self.row, self.column)
        self.image_original = self._generate_heatmap(self.data_original)
    
    def proccessed_tactile_signal_callback(self, msg):
        self.data_processed = np.array(msg.data, dtype=np.float32).reshape(-1, self.row, self.column)
        self.image_processed = self._generate_heatmap(self.data_processed)

    def _generate_heatmap(self, data):
        # arr = np.array(msg.data, dtype=np.float32)
        # reshaped = arr.reshape(-1, self.row, self.column)  # (N, H, W)
        map_num = data.shape[0]

        # Normalize and convert to uint8
        norm = (data * 255).clip(0, 255).astype(np.uint8)

        # Apply colormap per frame (OpenCV doesn't support batch LUT for 3-channel outputs)
        heatmaps = [cv2.applyColorMap(norm[i], cv2.COLORMAP_DEEPGREEN) for i in range(map_num)]

        # Resize each map
        heatmaps = [cv2.resize(hm, (hm.shape[1]*30, hm.shape[0]*30), interpolation=cv2.INTER_NEAREST) for hm in heatmaps]

        # Create a white spacer
        spacer = 255 * np.ones((heatmaps[0].shape[0], 5, 3), dtype=np.uint8)

        # Stack heatmaps with spacers
        combined = np.hstack([hm if i == 0 else np.hstack([spacer, hm]) for i, hm in enumerate(heatmaps)])

        return combined


    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            if self.image_policy is not None:
                cv2.imshow("Policy Tactile Signal", self.image_policy)
            if self.image_original is not None:
                cv2.imshow("Original Tactile Signal", self.image_original)
            if self.image_processed is not None:
                cv2.imshow("Processed Tactile Signal", self.image_processed)
            key = cv2.waitKey(1) & 0xFF  # Call once per loop
            if key == ord('q'):
                break
            elif key == ord('s'):
                # save all data and images
                if self.image_policy is not None:
                    cv2.imwrite(os.path.join(self.saved_folder, f'{self.category}_policy_tactile_signal_{self.save_id}.png'), self.image_policy)
                    np.savez(os.path.join(self.saved_folder, f'{self.category}_policy_tactile_signal_{self.save_id}.npz'), self.data_policy)
                if self.image_original is not None:
                    cv2.imwrite(os.path.join(self.saved_folder, f'{self.category}_original_tactile_signal_{self.save_id}.png'), self.image_original)
                    np.savez(os.path.join(self.saved_folder, f'{self.category}_original_tactile_signal_{self.save_id}.npz'), self.data_original)
                if self.image_processed is not None:
                    cv2.imwrite(os.path.join(self.saved_folder, f'{self.category}_processed_tactile_signal_{self.save_id}.png'), self.image_processed)
                    np.savez(os.path.join(self.saved_folder, f'{self.category}_processed_tactile_signal_{self.save_id}.npz'), self.data_processed)
                print(f"Saved tactile signal images and data with ID {self.save_id}")
                self.save_id += 1
            rate.sleep()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    rospy.init_node("tactile_visualizer", anonymous=True)
    visualizer = TactileVisualizer()
    visualizer.run()