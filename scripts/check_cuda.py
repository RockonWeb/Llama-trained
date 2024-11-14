import torch

print("CUDA доступна:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Текущее устройство CUDA:", torch.cuda.current_device())
    print("Название GPU:", torch.cuda.get_device_name(0))
