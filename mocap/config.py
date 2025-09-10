class MocapCfg:
    node_name = 'mocap_robot_object_pose'
    server_address = "192.168.50.2"
    client_address = "192.168.50.7"
    use_multicast = False
    print_level = 0
    frequency = 120
    robot_id = 4
    robot_pose_topic = '/robot_pose'
    object_id = 5  # cylinder
    object_position_shift = [0.0, 0.0, 0.002]  # cylinder
    object_pose_topic = '/object_pose'


