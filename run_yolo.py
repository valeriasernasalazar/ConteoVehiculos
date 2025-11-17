
"""
YOLO Wrapper Script for PyTorch 2.6+ Compatibility

This script modifies torch.load behavior to handle the serialization restrictions
introduced in PyTorch 2.6, allowing YOLO model weights to load correctly.
"""

import sys
import os
import torch

# Add the necessary classes to the safe globals list for PyTorch 2.6+
# This is based on the specific error message we received
torch.serialization.add_safe_globals([
    'ultralytics.nn.tasks.DetectionModel',
    'ultralytics.nn.modules.Conv',
    'ultralytics.nn.modules.Bottleneck',
    'ultralytics.nn.modules.C3',
    'ultralytics.nn.modules.SPPF',
    'ultralytics.nn.modules.Detect'
])

# Also monkey patch torch.load to handle the case where weights_only=True
# causes issues with the YOLO model
original_torch_load = torch.load

def patched_torch_load(f, map_location=None, pickle_module=None, **kwargs):
    """
    Patched version of torch.load that attempts both weights_only=False and True
    """
    try:
        # First try with weights_only=False (pre-2.6 behavior)
        return original_torch_load(f, map_location=map_location, 
                               pickle_module=pickle_module, 
                               weights_only=False, **kwargs)
    except Exception as e:
        print(f"Warning: Error loading with weights_only=False: {e}")
        print("Retrying with weights_only=True and safe globals...")
        # If that fails, try with weights_only=True and our safe globals
        return original_torch_load(f, map_location=map_location, 
                               pickle_module=pickle_module, 
                               weights_only=True, **kwargs)

# Replace torch.load with our patched version
torch.load = patched_torch_load

# Get the command line arguments for the YOLO script
yolo_script = 'yolov8-object-tracking/yolo/v8/detect/detect_follow.py'
args = sys.argv[1:]  # Skip the script name (run_yolo.py)

# Build the command to run the YOLO script
cmd_args = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in args)
full_cmd = f'python {yolo_script} {cmd_args}'

print(f"Running command with PyTorch 2.6+ compatibility: {full_cmd}")

# Run the YOLO script
exit_code = os.system(full_cmd)
sys.exit(exit_code // 256)  # Convert os.system exit code to regular exit code
