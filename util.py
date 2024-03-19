import cv2
import os
import warnings
import torch.nn as nn 
import numpy as np 
from tqdm import tqdm 
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
import wandb
from typing import Dict, List
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from CONFIG import CONFIG
from torchinfo import summary
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

###############################################################################33
# Metrics
class Metric: 
    def __init__(self) -> None:
        self.name = "metric_name"
        self.average = False
    
    def eval(y_pred : torch.Tensor, y : torch.Tensor) -> float:
        raise NotImplemented("")

class MSE_Metric(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Mean Squared Error (MSE)"
        self.average = True
        
    def eval(self, y_pred : torch.Tensor, y : torch.Tensor) -> float:
        return torch.mean((y_pred.cpu() - y.cpu())**2).item()
    
class PSNR_Metric(Metric):
    def __init__(self, crop_border=0) -> None:
        super().__init__()
        self.crop_border = crop_border
        self.name = "PSNR"
        self.average = True
        
    def eval(self, y_pred : torch.Tensor, y : torch.Tensor) -> float:   
        return psnr(255*y_pred.detach().cpu().numpy(), 255*y.detach().cpu().numpy())
    
class SSIM_Metric(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.name = "SSIM"
        self.average = True
    
    def eval(self, y_pred : torch.Tensor, y : torch.Tensor) -> float:
        return ssim(y_pred.detach().cpu(), y.detach().cpu(), data_range=1)

###############################################################################33
# utilities
def format(val : int) -> str:
    if val <= 1024:
        return f"{val}B"
    elif val <= 1024**2:
        return f"{round(val/1024, 2)}KB" 
    elif val <= 1024**3:
        return f"{round(val/(1024**2), 2)}MB" 
    return f"{round(val / (1024**3), 2)}GB"

def bgr2ycbcr(image: np.ndarray, only_use_y_channel: bool) -> np.ndarray:
    if only_use_y_channel:
        image = np.dot(image, [24.966, 128.553, 65.481]) + 16.0
    else:
        image = np.matmul(image, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [
            16, 128, 128]

    image /= 255.
    image = image.astype(np.float32)

    return image

def expand_y(image: np.ndarray) -> np.ndarray:
    # Normalize image data to [0, 1]
    image = image.astype(np.float32) / 255.

    # Convert BGR to YCbCr, and extract only Y channel
    y_image = bgr2ycbcr(image, only_use_y_channel=True)

    # Expand Y channel
    y_image = y_image[..., None]

    # Normalize the image data to [0, 255]
    y_image = y_image.astype(np.float64) * 255.0

    return y_image

# The following is the implementation of IQA method in Python, using CPU as processing device
def _check_image(raw_image: np.ndarray, dst_image: np.ndarray):
    # check image scale
    assert raw_image.shape == dst_image.shape, \
        f"Supplied images have different sizes {str(raw_image.shape)} and {str(dst_image.shape)}"

    # check image type
    if raw_image.dtype != dst_image.dtype:
        warnings.warn(f"Supplied images have different dtypes{str(raw_image.shape)} and {str(dst_image.shape)}")


def psnr(raw_image: np.ndarray, dst_image: np.ndarray) -> float:
    psnr_metrics = 10 * np.log10((255.0 ** 2) / np.mean((raw_image - dst_image) ** 2) + 1e-8)
    return psnr_metrics

###############################################################################33
# training/testing/reporting    
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
        
def test_task2(model : nn.Module, val_dataset : DataLoader, cfg : CONFIG, metrics : List[Metric], test_params : Dict[str, int] = {"save_in_total" : None, "save_every" : 0} ,  run = None):        
    # change the model to evaluation
    model.eval()
    
    # get the number of datapoints
    number_of_datapoints = len(val_dataset.dataset)    

    # allocate the memory for these datapoints (no need to keep appending the data, which will make it slower)
    metrics_names = [metric.name for metric in metrics]
    metrics_vals = np.zeros((number_of_datapoints, len(metrics_names)))

    save_images_every = 0
    if "save_in_total" in test_params and test_params["save_in_total"] is not None and test_params["save_in_total"] > 0:
        save_images_every = number_of_datapoints // test_params["save_in_total"]
        
    elif "save_every" in test_params:
        save_images_every = test_params["save_every"]

    if save_images_every > 0:
        shape = val_dataset.dataset[0][1].shape
        
        saved_images_pred = np.zeros((number_of_datapoints // save_images_every, shape[1], shape[2], shape[0]))
        saved_images_true = np.zeros((number_of_datapoints // save_images_every, shape[1], shape[2], shape[0]))
        img_c = 0
    
    # get the number of batches
    dataset_len = len(val_dataset)

    # create the progreess bar 
    pbar = tqdm(val_dataset)

    # variable that will track where we are in terms of all data (after iteration add batch size to it)
    c = 0
    for i, (x,y) in enumerate(pbar): 
        # get the predictions
        pred = model(x.to(CONFIG.DEVICE))
 
        # get the batch size
        bs = x.shape[0]

        # calculate the metric for every image in the batch:
        for img_i in range(bs):
            y_pred, y_ = pred[img_i], y[img_i]
            for j, metric in enumerate(metrics):
                metrics_vals[c, j] = metric.eval(torch.stack([y_pred.cpu()]), torch.stack([y_.cpu()]))
                
            if save_images_every > 0 and c % save_images_every == 0:
                saved_images_pred[img_c] = y_pred.detach().cpu().numpy().transpose(1, 2, 0)
                saved_images_true[img_c] = y_.detach().cpu().numpy().transpose(1, 2, 0)
                img_c += 1
                
            c += 1
                  
        if i % max((dataset_len//10),1) == 0 or i == dataset_len -1:
            s = ""

            for i,metric in enumerate(metrics):
                if metric.average:
                    s += f"{metric.name}={np.mean(metrics_vals[:(c-1), i])} ; "

            pbar.set_description(f"examples seen so far : {c} " + s)
 
    ret = {}
    
    for i,metric in enumerate(metrics):
        ret[metric.name] = metrics_vals[:, i]   
        
    if save_images_every > 0:
        ret["img_pred"] = saved_images_pred
        ret["img_true"] = saved_images_true
    
    return ret 

def report_metrics_task2(results : Dict, epoch : int, metrics : List[Metric], WANDB_ON : bool = True, prefix="val", run=None) -> Dict:
    
    ret = {}
    for metric in metrics:
        if metric.average:
            avg = np.average(results[metric.name])
            name_to_save = f"{prefix}_{metric.name}"
            ret[name_to_save] = avg

            if WANDB_ON:
                wandb.log({name_to_save : avg})
    
    if "img_pred" in results and "img_true" in results:
        size = results["img_pred"].shape
        imgs = []
        
        for b in range(size[0]):
            img = np.concatenate([results["img_pred"][b], results["img_true"][b]], axis=1)
            img_to_save = wandb.Image(img, caption="Left: predicted, right : true")
            wandb.log({f"Epoch={epoch}" : img_to_save})
            
    return ret
 
def run_experiment_task2(train_dataloader : torch.utils.data.DataLoader,
                         val_dataloader : torch.utils.data.DataLoader,
                         Model : nn.Module, 
                         run_name : str, 
                         model_parameters : dict, 
                         epochs : int, 
                         learning_rate : float, 
                         optimizer : str, 
                         savepath : str,
                         cfg : CONFIG,
                         loss : str = "MSE",
                         test_params : Dict = {"save_in_total" : 50},
                         base_lr:float=1e-4, 
                         max_lr:float=1e-3, 
                         metrics : List[Metric] = [MSE_Metric(), PSNR_Metric(), SSIM_Metric()],
                         scheduler_en : bool = True,
                         metric_keyword : str = "acc",
                         lr_steps : int = 1000,
                         WANDB_ON : bool = True,
                         start_with_test : bool  = True):

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


    if loss.lower() == "mse":         
        criterion = nn.MSELoss()
    else:
        raise Exception("specify correctly the loss function")
    
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
    
    # it was useful for superresolution, to have some starting point, to see how much it improved
    if start_with_test:
        test_res = test_task2(model, train_dataloader, cfg=cfg, test_params=test_params,  metrics=metrics)
        evaluation = report_metrics_task2(test_res, epoch=-1, metrics=metrics, prefix="train", WANDB_ON=WANDB_ON)
        
        test_res = test_task2(model, val_dataloader, cfg=cfg, test_params=test_params,  metrics=metrics)
        evaluation = report_metrics_task2(test_res, epoch=-1, metrics=metrics, prefix="val", WANDB_ON=WANDB_ON)


    for epoch in range(epochs):
        train(train_dataloader, model, optimizer_, scheduler, criterion, epoch=epoch, WANDB_ON=WANDB_ON, cfg=cfg)

        test_res = test_task2(model, train_dataloader, cfg=cfg, test_params=test_params,  metrics=metrics)
        evaluation = report_metrics_task2(test_res, epoch=epoch, metrics=metrics, prefix="train", WANDB_ON=WANDB_ON)
        
        test_res = test_task2(model, val_dataloader, cfg=cfg, test_params=test_params,  metrics=metrics)
        evaluation = report_metrics_task2(test_res, epoch=epoch, metrics=metrics, prefix="val", WANDB_ON=WANDB_ON)

        best_metric = save_model(model, evaluation, metric_keyword, best_metric, savepath)
    
    if WANDB_ON:
        wandb.finish()