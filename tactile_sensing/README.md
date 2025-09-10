### Validated System
Ubuntu 20.04 with ROS Noetic

### Indepedent Installation
1. Create a conda environment with python3.10:
   ```bash
   conda create -n touch python=3.10
   ```
2. Install the dependencies:
   ```bash
   conda activate touch
   pip install -r requirements.txt
   ```

### Usage
Run the script:
```bash
python run_tactile_sensing.py
```
or
```bash
python run_tactile_sensing.py --viz=True
```

