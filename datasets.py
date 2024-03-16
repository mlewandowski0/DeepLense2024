import os 
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class DeepLenseClassificationDataset(Dataset):

    def __init__(self, folder_path : str, randomize_dataset : bool = True,
                 preprocess_dataset : bool = True, data_limit=0, mean=None, std=None) -> None:
        self.folder_path = folder_path
        self.paths = []
        
        self.class_folders = []     
        self.class_names = [] 
        for class_folder in os.listdir(folder_path):
            full_path = os.path.join(folder_path, class_folder)
            if os.path.isdir(full_path):
                self.class_names.append(class_folder)
                self.class_folders.append(full_path)
        
        self.filepaths = []
        self.classes = []
        self.dataset = []
        
        # craete the classes and filepath arrays
        for class_, class_folder in enumerate(self.class_folders):
            if data_limit <= 0:
                elements = os.listdir(class_folder)
            else :
                elements = os.listdir(class_folder)[:data_limit]        

            for el in elements:
                full_path = os.path.join(class_folder, el)
                self.filepaths.append(full_path)
                self.classes.append(class_)
                                               
        self.filepaths = np.array(self.filepaths)
        self.classes = np.array(self.classes)
        
        # load the dataset
        for filepath in tqdm(self.filepaths, desc="loading numpy"):
            datapoint = np.load(filepath)
            self.dataset.append(datapoint)            
            
        if randomize_dataset:
            self.randomize_dataset()
            
        if mean is None:
            self.mean = np.mean(self.dataset)
        else:
            self.mean = mean

        if std is None:
            self.std = np.std(self.dataset)
        else:
            self.std = std

        if preprocess_dataset:
            for i in tqdm(range(len(self.dataset)), desc="preprocessing"):
                self.dataset[i] = self.preprocess_single(self.dataset[i])

        self.dataset = torch.stack(self.dataset)


    def randomize_dataset(self):
        idxes = np.arange(len(self.filepaths))
        np.random.shuffle(idxes)

        self.filepaths = self.filepaths[idxes]
        self.classes = self.classes[idxes]
        self.dataset = self.dataset[idxes]
    
    def preprocess_single(self, x : np.ndarray) -> torch.Tensor:
        return torch.tensor( (x - self.mean) / self.std).float()

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        return self.dataset[idx], self.classes[idx] 