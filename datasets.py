import os 
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

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
    
    
class DeepLenseSuperresolutionDataset(Dataset):

    def __init__(self, folder_path : str,
                 randomize_dataset : bool = True,
                 preprocess_LR : bool = True, 
                 preprocess_HR : bool = True,
                 call_preprocess : bool = True,
                 data_limit=0, 
                 mean_LR = None, std_LR=None, 
                 mean_HR=None, std_HR=None) -> None:
        
        self.folder_path = folder_path
        self.class_folders = []
        self.preprocess_LR = preprocess_LR
        self.preprocess_HR = preprocess_HR

        folders = [os.path.join(self.folder_path, v) for v in os.listdir(folder_path)]

        self.LR = [v for v in folders if v.endswith("LR")][0]
        self.HR = [v for v in folders if v.endswith("HR")][0]

        self.class_folders = [self.LR, self.HR]
        
        print(self.LR, self.HR)
        assert os.listdir(self.LR) == os.listdir(self.HR), "the number of samples in Low Resolution has to be the same as High Resolution"

        # get the samples 
        self.samples = os.listdir(self.LR)
        
        # limit the data (for faster prototyping )
        if data_limit > 0:
            self.samples = self.samples[:data_limit]
                
        # Datapoints
        self.LR_data = []
        self.HR_data = []
            
        pbar = tqdm(self.samples)
        for path in pbar:
            # load from the low resolution
            img1 = np.load(os.path.join(self.LR, path))
            self.LR_data.append(torch.Tensor(img1))            
            
            # load from the high resolution
            img2 = np.load(os.path.join(self.HR, path))
            self.HR_data.append(torch.Tensor(img2))
            
            pbar.set_description("Loading dataset : ")
        
        self.samples = np.array(self.samples)
        self.LR_data = torch.stack(self.LR_data)
        self.HR_data = torch.stack(self.HR_data)
        
        # calculate statistical values about the dataset 
        self.mean_HR = mean_HR
        if self.mean_HR is None:
            self.mean_HR = torch.mean(self.HR_data.reshape(-1))
            
        self.mean_LR = mean_LR
        if self.mean_LR is None:
            self.mean_LR = torch.mean(self.LR_data.reshape(-1))

        self.std_HR = std_HR
        if self.std_HR is None:
            self.std_HR = torch.std(self.HR_data.reshape(-1))

        self.std_LR = std_LR
        if self.std_LR is None:
            self.std_LR = torch.std(self.LR_data.reshape(-1))
            
        if call_preprocess:
            self.preprocess()            
        
        
        if randomize_dataset:
            self.randomize_dataset()

    def preprocess(self):
        
        pbar = tqdm(self.samples)
        
        for i, _ in enumerate(pbar):
            
            if self.preprocess_LR:
                self.LR_data[i] = self.preprocess_LR_func(self.LR_data[i])

            if self.preprocess_HR:
                self.HR_data[i] = self.preprocess_HR_func(self.HR_data[i])
            
            pbar.set_description("preprocessing :")
        
        
    # To override later (if any preprocessing is required)
    def preprocess_LR_func(self, x : torch.Tensor) -> torch.Tensor:
        return x
        
    def preprocess_HR_func(self, x : torch.Tensor) -> torch.Tensor:
        return x
        

    def randomize_dataset(self):
        idxes = np.arange(len(self.LR_data))
        random.shuffle(idxes)

        self.samples = self.samples[idxes]
        self.LR_data = self.LR_data[idxes]
        self.HR_data = self.HR_data[idxes]
    
    def preprocess_input(self, x : np.ndarray) -> torch.Tensor:
        return torch.tensor( (x - self.mean) / self.std).float()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.LR_data[idx], self.HR_data[idx]
    