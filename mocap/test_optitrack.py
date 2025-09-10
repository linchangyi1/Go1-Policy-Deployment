import time
from mocap.natnet_client import NatNetClient
from mocap.util import quaternion_to_euler
import sched
import queue


class OptiTrack:
    def __init__(self, server_address = "192.168.50.2", client_address = "192.168.50.7", use_multicast = False, frequency = 120, optitrack_queue=None):
        self.frequency = frequency
        self.positions = {}
        self.rotations = {}
        self.optitrack_queue = optitrack_queue
        self.time_interval = 1 / self.frequency

        self.is_running = False
        self.streaming_client = NatNetClient()
        self.streaming_client.set_client_address(client_address)
        self.streaming_client.set_server_address(server_address)
        self.streaming_client.set_use_multicast(use_multicast)
        self.streaming_client.rigid_body_listener = self.receive_rigid_body_frame

        self.stream_sched = sched.scheduler(time.perf_counter, time.sleep)
        
    def receive_rigid_body_frame(self, id, position, rotation_quaternion):
        # Position and rotation received
        self.positions[id] = position
        # The rotation is in quaternion. We need to convert it to euler angles
        rotx, roty, rotz = quaternion_to_euler(rotation_quaternion)
        # Store the roll pitch and yaw angles
        self.rotations[id] = (rotx, roty, rotz)
        print('id', id, 'position', position, 'rotation', (rotx, roty, rotz))
        
    def start_streaming(self, robot_id = 1):
        is_running = self.streaming_client.run()
        while is_running:
            if robot_id in self.positions:
                print('robot_id', robot_id, 'Last position', self.positions[robot_id], 'rotation', self.rotations[robot_id])
            time.sleep(1 / self.frequency)
    
    def stop_streaming(self):
        self.is_running = False
        self.streaming_client.shutdown()
            
    def start_streaming_attitude(self, robot_ids = [1], show = False):  
        self.is_running = self.streaming_client.run()
        current_time = time.perf_counter()
        sleep_time = 0
        freq = 0
        sleep_interval = (1/self.frequency)
        last_time = time.perf_counter() - sleep_interval
        while self.is_running:
            # print('robot_ids', robot_ids)
            # print("command_call_count: ", self.streaming_client._command_call_count)
            # print("data_call_count: ", self.streaming_client._data_call_count)
            # print("stop_threads: ", self.streaming_client.stop_threads)
            begin_time = time.perf_counter()
            for robot_id in robot_ids:
                # print("positions", self.positions)
                if robot_id in self.positions:
                    robot_position = self.positions[robot_id]
                    robot_rotation = self.rotations[robot_id]
                    # print('robot_id', robot_id, 'Last position', self.positions[robot_id], 'rotation', self.rotations[robot_id])
                    current_time = time.perf_counter()
                    begin_delay = current_time - begin_time
                    data = [robot_id, robot_position, robot_rotation, current_time]
                    if self.optitrack_queue != None:
                        self.optitrack_queue.put(data)
                    if show == True:
                        freq = 1/(current_time - last_time)
                        print('robot_id', robot_id, 'Last position', self.positions[robot_id], 'rotation', self.rotations[robot_id], 'time', current_time, 'freq', freq)
                    
                    sleep_time = max(0, (current_time - time.perf_counter() + sleep_interval - begin_delay))
                    
            last_time = current_time

            time.sleep(sleep_time)




    #streaming by multiprocessing
    def start_streaming_attitude_multi(self, robot_ids = [1], show = False):  
        self.is_running = self.streaming_client.run()
        current_time = time.perf_counter()
        sleep_time = 0
        freq = 0
        sleep_interval = (1/self.frequency)
        last_time = time.perf_counter() - sleep_interval
        # begin_delay = 0
        while self.is_running:
            begin_time = time.perf_counter()
            for robot_id in robot_ids:
                if robot_id in self.positions:
                    robot_position = self.positions[robot_id]
                    robot_rotation = self.rotations[robot_id]
                    current_time = time.perf_counter()
                    currenit_data_time = time.time()
                    begin_delay = current_time - begin_time
                    data = [robot_id, robot_position, robot_rotation, currenit_data_time]
                    
                    if self.optitrack_queue != None:
                        try:
                            self.optitrack_queue.put_nowait(data)
                        except queue.Full:
                            try:
                                old_data = self.optitrack_queue.get_nowait()
                                self.optitrack_queue.put_nowait(data)
                            except queue.Empty:
                                continue

                    if show == True:
                        freq = 1/(current_time - last_time)
                        print('robot_id', robot_id, 'Last position', self.positions[robot_id], 'rotation', self.rotations[robot_id], 'time', currenit_data_time, 'freq', freq)
   
            last_time = current_time
            sleep_time = max(0,sleep_interval + current_time - time.perf_counter())
            time.sleep(sleep_interval)    
   
            
    # def start_streaming_attitude_all(self, robot_id = 1, show = False):
    #     self.is_running = self.streaming_client.run()
    #     while self.is_running:
    #         if robot_id in self.positions:
    #             robot_position = self.positions[robot_id]
    #             robot_rotation = self.rotations[robot_id]
    #             if show == True:
    #                 print('robot_id', robot_id, 'Last position', self.positions[robot_id], 'rotation', self.rotations[robot_id])
    #                 print(time.time())
            
    #         time.sleep(1 / self.frequency)
            
    #     print(robot_id, robot_position, robot_rotation)    
    
    
    #streaming by threading
    def start_streaming_attitude_thread(self, robot_id=1, show=False):
        self.is_running = self.streaming_client.run()
        try:
            while self.is_running:
                if robot_id in self.positions:
                    robot_position = self.positions[robot_id]
                    robot_rotation = self.rotations[robot_id]
                    if show:
                        print('robot_id', robot_id, 'Last position', self.positions[robot_id], 'rotation', self.rotations[robot_id])
                        print(time.time())
                time.sleep(1 / self.frequency)
        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Stopping streaming.")
            self.stop_streaming()
            raise  # Re-raise the KeyboardInterrupt to propagate it to the main script 

    

if __name__ == "__main__" :
    server_address = "192.168.50.2"
    client_address = "192.168.50.7"
    use_multicast = False
    robot_ids = [2]
    frequency = 60
    robo_pos = OptiTrack(server_address, client_address, use_multicast, frequency)
    try:
        robo_pos.start_streaming_attitude(robot_ids, show = False)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Stopping streaming.")
    finally:
        robo_pos.stop_streaming()


