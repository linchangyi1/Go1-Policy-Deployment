import enum

class OperationMode(enum.Enum):
    LieDown = 1
    Lying = 2
    StandUp = 3
    Standing = 4
    Policy = 5
    ZeroCmdPolicy = 6

class OperationModeFunc(enum.Enum):
    LieDown = 'lie_down'
    Lying = 'lying'
    StandUp = 'stand_up'
    Standing = 'standing'
    Policy = 'policy'
    ZeroCmdPolicy = 'zero_cmd_policy'

class FSMCommand(enum.Enum):
    ActionDown = 0
    ActionUp = 1
    ActionLeft = 2
    ActionRight = 3

class FSMCmdtoOperationMode(enum.Enum):
    ActionDown = OperationMode.LieDown
    ActionUp = OperationMode.StandUp
    ActionLeft = OperationMode.Policy
    ActionRight = OperationMode.ZeroCmdPolicy

def fsm_cmd_to_operation_mode_mapping(fsm_cmd):
    return FSMCmdtoOperationMode[fsm_cmd.name].value


if __name__ == "__main__":
    for item in FSMCmdtoOperationMode:
        print(item, item.value)
    func = {operation_mode:'step_'+OperationModeFunc[operation_mode.name].value for operation_mode in OperationMode}
    print(func[OperationMode.LieDown])
