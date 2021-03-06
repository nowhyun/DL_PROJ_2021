import argparse
import random
import time
import os
import warnings
warnings.filterwarnings("ignore")

from copy import deepcopy
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use("ggplot")

import transformers
from transformers import BertTokenizer, BertModel, ElectraTokenizer, ElectraModel, AdamW, get_linear_schedule_with_warmup

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.utils.data.sampler import BatchSampler

import torchvision
from torchvision import transforms, datasets
from Models.BertClf import *
from Models.LstmClf import *
from Models.ElectraClf import *
from Models.ConvClf import *


#################################################################################################################
# Reproducible
#################################################################################################################
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42)


#################################################################################################################
# Initialized model
#################################################################################################################
def initialize_model(opt, len_train_dataloader, device):
    """
    Initialize the mdoel, the optimizer and the learning rate scheduler
    
    :param opt: hyper parameters
    
    :return model, optimizer, scheduler
    """
    if opt.model == "BERT":
        model = BertClassifier(opt)
        model.to(device)
        
        optimizer = AdamW([
            {"params": model.bert.parameters(), "lr": opt.lr_pretrained, "eps": opt.eps},
            {"params": model.classifier.parameters(), "lr": opt.lr_clf, "eps": opt.eps}
                           ])
        
        # Total number of training steps
        total_steps = len_train_dataloader * opt.max_epoch

        # Set up the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)
        
    elif opt.model == "ELECTRA":
        model = ElectraClassifier(opt)
        model.to(device)
        
        optimizer = AdamW([
            {"params": model.electra.parameters(), "lr": opt.lr_pretrained, "eps": opt.eps},
            {"params": model.classifier.parameters(), "lr": opt.lr_clf, "eps": opt.eps}
                           ])
        
        # Total number of training steps
        total_steps = len_train_dataloader * opt.max_epoch

        # Set up the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)
        
    elif opt.model == "BILSTM":
        model = LstmClassifier(opt, 30522)
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr_clf, weight_decay=opt.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.0001)
        
    elif opt.model == "CNN":
        model = ConvClassifier(opt, 30522)
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr_clf, weight_decay=opt.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.0001)
    
    return model, optimizer, scheduler


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
    
    
################################################################################################################
# Make Submission
################################################################################################################
def make_preds(model, opt, device, test_dataloader):
    """
    Perform a forward pass on the trained model to predict probabilities on the test set.
    0 - negative
    1 - somewhat negative
    2 - neutral
    3 - somewhat positive
    4 - positive
    
    :param model: trained model
    :param test_dataloader: dataloader which is obtained by data_load method
    
    :return preds: predictions [array]
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()
    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_ids_tsr, b_masks_tsr = tuple(tsr.to(device) for tsr in batch)[:2]

        # Compute logits
        with torch.no_grad():
            if opt.model in ["BILSTM", "CNN"]:
                logits = model(b_ids_tsr)
            else:
                logits = model(b_ids_tsr, b_masks_tsr)
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()
    preds = np.argmax(probs, axis=1)
    return preds


def make_submission(model, opt, device, test_dataloader, full=False):
    """
    Save a submission.csv
    
    :param model: trained model
    :param opt: hyper parameters
    :param test_dataloader: dataloader which is obtained by data_load method
    """
    submission_templete_df = pd.read_csv(opt.data_path +'/sample_sub.csv') # a sample submission file in the correct format
    # Early stop model
    if opt.save == 1:
        if full == True:
            model_save_path = str(opt.save_model_path) + "/" + opt.signature +'_full.model'
            model.load_state_dict(torch.load(model_save_path))
        else:
            model_save_path = str(opt.save_model_path) + "/" + opt.signature +'.model'
            model.load_state_dict(torch.load(model_save_path))
    preds = make_preds(model, opt, device, test_dataloader)
    
    assert len(preds) == 4311
    submission_templete_df.Category = preds
    print(submission_templete_df)
    if full == True:
        submission_save_path = str(opt.save_submission_path) + "/" + opt.signature +'_full.csv'
    else:
        submission_save_path = str(opt.save_submission_path) + "/" + opt.signature +'.csv'
    print(f"\t...Save complete at {submission_save_path}")
    submission_templete_df.to_csv(submission_save_path, index=False)
    
    
################################################################################################################
# ETC...
################################################################################################################
def numOfparams(model):
    """Return the number of parameters
    """
    num_of_total_params = sum(p.numel() for p in model.parameters())
    return num_of_total_params

def get_sent_embedding_and_label(dataloader, model, batch_size, device):
    """Get sentence embeddings array and labels array
    """
    sent_embeddings_arr = []
    labels_arr = []
    n = 0
    for i, batch in enumerate(dataloader):
        b_ids_tsr, b_masks_tsr, b_labels_tsr = tuple(tsrs.to(device) for tsrs in batch)
        if str(model)[:4] == 'Lstm':
            embeddings_tsr = model.extract_sent_embd(b_ids_tsr)
            sent_embeddings_tsr = embeddings_tsr
            n = 0
        else:
            embeddings_tsr = model.extract_sent_embd(b_ids_tsr, b_masks_tsr)
            sent_embeddings_tsr = embeddings_tsr[:,0,:]
            n = 7
        sent_embeddings_arr.append(sent_embeddings_tsr)
        labels_arr.append(b_labels_tsr)
        if i == n:
            print(f"the number of sample: {batch_size * (i+1)}")
            break
    sent_embeddings_arr = torch.cat(sent_embeddings_arr, dim=0).detach().cpu().numpy()
    labels_arr = torch.cat(labels_arr).detach().cpu().numpy()
    return sent_embeddings_arr, labels_arr

def draw_embeddings(sent_embeddings_arr, labels_arr, option="pca"):
    """Project the sentence embedding vector into a 2-dimensional space
    """
    if option == "pca":
        from sklearn.decomposition import PCA
        model = PCA(n_components=2)
    elif option == "tsne":
        from sklearn.manifold import TSNE 
        model = TSNE(n_components=2)
    projection_res = model.fit_transform(sent_embeddings_arr)
    projection_df = pd.DataFrame()
    projection_df["proj_x"] = projection_res[:,0]
    projection_df["proj_y"] = projection_res[:,1]
    projection_df["label"] = labels_arr
    plt.figure(figsize=(16,10))
    sns.scatterplot(x="proj_x",
                    y="proj_y",
                    hue="label",
                    palette=sns.color_palette("hls", 5),
                    data=projection_df,
                    legend="full",
                    alpha=0.9)
    plt.show()

def vis_attention(embeddings):
    return None