import torch

class CONFIG:
    BATCH_SIZE = 16
    
    # limit the data to prototype faster
    DATA_LIMIT = 100
    
    CLASSES = 3
    ROUND_NUMBER = 3
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    TASK_NAME = "DeepLense2024_task1"