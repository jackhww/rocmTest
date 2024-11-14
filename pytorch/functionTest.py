import torch
from torch.utils.collect_env import main

print(f"Device name [0]:", torch.cuda.get_device_name(0))
main()
