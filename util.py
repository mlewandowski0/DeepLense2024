from torch.utils.data import DataLoader
import torch.nn as nn 
import numpy as np 
from tqdm import tqdm 
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
import wandb
from typing import Dict
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from CONFIG import CONFIG
from torchinfo import summary
import os

def format(val : int) -> str:
    if val <= 1024:
        return f"{val}B"
    elif val <= 1024**2:
        return f"{round(val/1024, 2)}KB" 
    elif val <= 1024**3:
        return f"{round(val/(1024**2), 2)}MB" 
    return f"{round(val / (1024**3), 2)}GB"
    
def test(model : nn.Module, val_dataset : DataLoader, cfg : CONFIG,   run = None):
    
    # change the model to evaluation
    model.eval()
    
    # get the number of datapoints
    number_of_datapoints = len(val_dataset.dataset)    

    # allocate the memory for these datapoints (no need to keep appending the data, which will make it slower)
    predictions_prob = np.zeros((number_of_datapoints, cfg.CLASSES))
    predictions = np.zeros(number_of_datapoints)
    true_values = np.zeros(number_of_datapoints) 
    

    # get the number of batches
    dataset_len = len(val_dataset)

    # create the progreess bar 
    pbar = tqdm(val_dataset)

    # variable that will track where we are in terms of all data (after iteration add batch size to it)
    c = 0
    for i, (x,y) in enumerate(pbar): 
        # get the predictions
        pred = model(x.to(cfg.DEVICE))
 
        # get the batch size
        bs = x.shape[0]

        true_values[c : (c + bs)] = y.detach().numpy()
        predictions_prob[c : (c + bs)] = torch.softmax(pred.cpu().detach(), dim=1).numpy()
        predictions[c : (c + bs)] = torch.argmax(pred, 1).cpu().detach().numpy()
        c += bs 
           
        if i % (dataset_len//10) == 0 or i == dataset_len -1:
            acc = accuracy_score(predictions[:c], true_values[:c])
            try:
                roc_auc = roc_auc_score(true_values[:c], predictions_prob[:c, :], multi_class='ovr')            
            
            # It can happen at the beginning
            except Exception as e:
                roc_auc = 0

            pbar.set_description(f"examples seen so far : {c}, accuracy = {round(acc, cfg.ROUND_NUMBER)}, AUC ROC = {round(roc_auc, CONFIG.ROUND_NUMBER)}")
    
    return {"predition_prob" : predictions_prob, "predictions" : predictions, "true" : true_values}

def report_metrics(results : Dict, epoch : int, WANDB_ON : bool = True, prefix="val", run=None) -> Dict:
    predictions = results["predictions"]
    true_values = results["true"]
    predictions_prob = results["predition_prob"]
    
    acc = accuracy_score(predictions, true_values)
    roc_auc_ovr = roc_auc_score(true_values, predictions_prob, multi_class='ovr')            
    roc_auc_ovo = roc_auc_score(true_values, predictions_prob, multi_class='ovo')  
    
    if WANDB_ON:
        wandb.log({f"{prefix}_acc": acc, f"{prefix}_ROC_AUC_ovr": roc_auc_ovr, f"{prefix}_ROC_AUC_ovo" : roc_auc_ovo})
        wandb.log({f"{prefix}_ROC_epoch={epoch}" : wandb.plot.roc_curve(true_values, predictions_prob, labels=val_data.class_names)})
    
    return {"accuracy" : acc, "ROC_AUC_OVR" :  roc_auc_ovr, "ROC_AUC_OVO" : roc_auc_ovo}


def save_model(model : nn.Module, metrics_results : Dict, metric_keyword : str, best_metric : float, savepath : str):
    
    if metrics_results[metric_keyword] > best_metric:
        torch.save(model.state_dict(), savepath)
        
    return max(metrics_results[metric_keyword], best_metric)

def train(train_dataloader : torch.utils.data.DataLoader, 
          model : nn.Module, 
          optimizer : optim.Optimizer, 
          scheduler : lr_sched.LRScheduler, 
          criterion, 
          epoch : int, 
          cfg : CONFIG,
          WANDB_ON : bool=True):
    model.train()
    running_loss = 0.0
    i = 1
    train_len = len(train_dataloader)
    
    pb = tqdm(train_dataloader)
    for inputs, labels in pb:
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs.to(cfg.DEVICE))
        loss = criterion(outputs, labels.to(cfg.DEVICE))
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()  # Update learning rate
        
        running_loss += loss.item()
        
        if (i-1) % (train_len//10) == 0 or i == train_len:      
            pb.set_description(f"EPOCH : {epoch}, average loss : {running_loss / i}")
        i += 1
    
    if WANDB_ON:
        wandb.log({"loss" : running_loss/len(train_dataloader)})

    

def run_experiment(train_dataloader : torch.utils.data.DataLoader,
                   val_dataloader : torch.utils.data.DataLoader,
                   Model : nn.Module, 
                   run_name : str, 
                   model_parameters : dict, 
                   epochs : int, 
                   learning_rate : float, 
                   optimizer : str, 
                   savepath : str,
                   cfg : CONFIG,
                   base_lr:float=1e-4, 
                   max_lr:float=1e-3, 
                   scheduler_en : bool = True,
                   metric_keyword : str = "acc",
                   lr_steps : int = 1000,
                   WANDB_ON : bool = True):

    try:
        os.mkdir("models") 
    except FileExistsError:
        pass
    
    model = Model(**model_parameters).to(cfg.DEVICE)
    
    config = {"model name" : model.__class__,
              "run name" : run_name,
              "epochs" : epochs,
              "learning rate" : learning_rate,
              "optimizer" : optimizer, 
              "uses scheduler" : scheduler_en,
              "base_lr" : base_lr,
              "max_lr" : max_lr,
              "lr_steps" : lr_steps}
    
    config.update(model_parameters)    
    
    model_summary_str = str(summary(model, input_size=(cfg.BATCH_SIZE, 1, 150, 150)))
        
    if WANDB_ON:
        run = wandb.init(project=cfg.TASK_NAME,
                     name=f"experiment_{run_name}",
                     notes="Model summary : \n" + model_summary_str,
                     config=config)

    
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    if optimizer.lower() == "adam":
        optimizer_ = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer.lower() == "adamw":
        optimizer_ = optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        raise Exception("specify correctly the optimizer !")

    # Set up CyclicLR scheduler
    scheduler = None
    if scheduler_en:
        scheduler = lr_sched.CyclicLR(optimizer_, base_lr=base_lr, max_lr=max_lr, step_size_up=lr_steps, mode='triangular')

    best_metric = 0 
    
    for epoch in range(epochs):
        train(train_dataloader, model, optimizer_, scheduler, criterion, epoch=epoch, WANDB_ON=WANDB_ON, cfg=cfg)

        test_res = test(model, train_dataloader, cfg=cfg)
        evaluation = report_metrics(test_res, epoch=epoch, prefix="train", WANDB_ON=WANDB_ON)
        
        test_res = test(model, val_dataloader, cfg=cfg)
        evaluation = report_metrics(test_res, epoch=epoch, prefix="val", WANDB_ON=WANDB_ON)

        best_metric = save_model(model, evaluation, metric_keyword, best_metric, savepath)
    
    if WANDB_ON:
        wandb.finish()