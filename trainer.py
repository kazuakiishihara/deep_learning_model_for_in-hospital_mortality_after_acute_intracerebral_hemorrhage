
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torchvision import transforms as transforms

import dataloader, resnet

class Trainer:
    def __init__(
        self, 
        model, 
        device, 
        optimizer, 
        criterion
    ):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion

        self.best_valid_score = np.inf
        self.n_patience = 0
        self.lastmodel = None
        
        self.train_loss_list, self.valid_loss_list  = [], []
        self.train_auc_list, self.valid_auc_list  = [], []
        
    def fit(self, epochs, train_loader, valid_loader):        
        for n_epoch in range(1, epochs + 1):
            self.info_message("EPOCH: {}", n_epoch)
            
            train_loss, train_auc, train_time = self.train_epoch(train_loader)
            valid_loss, valid_auc, valid_time = self.valid_epoch(valid_loader)
            
            self.train_loss_list.append(train_loss)
            self.valid_loss_list.append(valid_loss)

            self.train_auc_list.append(train_auc)
            self.valid_auc_list.append(valid_auc)
            
            self.info_message(
                "[Epoch Train: {}] loss: {:.4f}, auc: {:.4f}, time: {:.2f} s            ",
                n_epoch, train_loss, train_auc, train_time
            )
            
            self.info_message(
                "[Epoch Valid: {}] loss: {:.4f}, auc: {:.4f}, time: {:.2f} s",
                n_epoch, valid_loss, valid_auc, valid_time
            )

            if self.best_valid_score > valid_loss: 
                self.best_valid_score = valid_loss
            self.save_model(n_epoch)
        
        self.print_loss_auc()
    
    def train_epoch(self, train_loader):
        self.model.train()
        scaler = torch.cuda.amp.GradScaler()
        t = time.time()
        sum_loss = 0

        prob_li = []
        label_li = []

        for step, batch in enumerate(train_loader, 1):
            X = batch["X"].to(self.device, non_blocking=True)
            targets = batch["y"].to(self.device, non_blocking=True)
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs, latent = self.model(X)
                outputs = outputs.squeeze(1)
                prob = torch.sigmoid(outputs)
                prob_li.extend(prob.detach().clone().to('cpu').tolist())
                label_li.extend(batch["y"].detach().clone().to('cpu').tolist())

                loss = self.criterion(outputs, targets)
                sum_loss += loss.detach().item()

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            
            message = 'Train Step {}/{}, train_loss: {:.4f}'
            self.info_message(message, step, len(train_loader), sum_loss/step, end="\r")
            
        auc = roc_auc_score(label_li, prob_li)

        return sum_loss/len(train_loader), auc, int(time.time() - t)
    
    def valid_epoch(self, valid_loader):
        self.model.eval()
        t = time.time()
        sum_loss = 0

        prob_li = []
        label_li = []

        for step, batch in enumerate(valid_loader, 1):
            with torch.no_grad():
                X = batch["X"].to(self.device, non_blocking=True)
                targets = batch["y"].to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    outputs, latent = self.model(X)
                    outputs = outputs.squeeze(1)
                    prob = torch.sigmoid(outputs)
                    prob_li.extend(prob.detach().clone().to('cpu').tolist())
                    label_li.extend(batch["y"].detach().clone().to('cpu').tolist())

                    loss = self.criterion(outputs, targets)
                    sum_loss += loss.detach().item()

            message = 'Valid Step {}/{}, valid_loss: {:.4f}'
            self.info_message(message, step, len(valid_loader), sum_loss/step, end="\r")
            
        auc = roc_auc_score(label_li, prob_li)
        
        return sum_loss/len(valid_loader), auc, int(time.time() - t)
    
    def save_model(self, n_epoch):
        self.lastmodel =  "./ResNet_epoch_{}.pth".format(n_epoch) # Save trained model
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_valid_score,
                "n_epoch": n_epoch,
            },
            self.lastmodel,
        )
        
    def info_message(self, message, *args, end="\n"):
        print(message.format(*args), end=end)

    def print_loss_auc(self):
        df = pd.DataFrame({
            'Epoch': range(1, len(self.train_auc_list)+1),
            'Train AUC': self.train_auc_list,
            'Valid AUC': self.valid_auc_list,
            'Train Loss': self.train_loss_list,
            'Valid Loss': self.valid_loss_list,
        })
        df.to_csv('./loss_auc.csv', index=False) # output loss-auc performance


def train_run(df_train, df_valid, slice_num=22, img_size=224):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = dataloader.Dataset(df_train, df_train["pid"].values, slice_num, img_size)
    valid_dataset = dataloader.Dataset(df_valid, df_valid["pid"].values, slice_num, img_size)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)
    
    model = resnet.generate_model(18)
    model.to(device, non_blocking=True)

    ratio = np.sum(df_train['obj'].values == 0) / np.sum(df_train['obj'].values == 1)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ratio))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    trainer = Trainer(model, device, optimizer, criterion)
    
    history = trainer.fit(2, train_loader, valid_loader)
    
    return trainer.lastmodel