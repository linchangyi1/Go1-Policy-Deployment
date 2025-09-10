from params_proto import PrefixProto


class Cfg(PrefixProto, cli=False):
    class thread(PrefixProto, cli=False):
        sp = False
        sp = True

    class pcb(PrefixProto, cli=False):
        port = '/dev/ttyUSB0'
        baudrate = 2000000
        cable_num_row = 17
        cable_num_col = 13

    class sensor(PrefixProto, cli=False):
        taxel_num_row = 17
        taxel_num_col = 13
        signal_base = 32
        signal_min = 5*signal_base
        signal_max = 25*signal_base
        use_log = True
        use_reference = True
        reference_frame_num = 120
        contact_signal_threshold = 40
        contact_force_threshold = 0.05

    class gui(PrefixProto, cli=False):
        viz = False
        viz = True
        window_name = 'Tactile Sensor'
        pixel_per_row = 65
        pixel_per_col = 65
        viz_data = True
        viz_data_type = 'delta'
        # viz_data_type = 'raw'
        font_scale = 0.6
        font_thickness = 1
        font_color = (255, 255, 255)
        contact_scale = 0.3
        # contact_scale = 0.
        log_frequency = True

    class ros(PrefixProto, cli=False):
        publish_signal = True
        binary_policy = False
        # binary_policy = True
        policy_tactile_topic = '/policy_tactile_signal'
        processed_tactile_topic="/processed_tactile_signal"

    class policy(PrefixProto, cli=False):
        discrete_policy = False
        # discrete_policy = True






