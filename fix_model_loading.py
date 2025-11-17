import torch
from ultralytics.nn.tasks import DetectionModel

# Add the DetectionModel class to PyTorch's safe globals list
torch.serialization.add_safe_globals([DetectionModel])
