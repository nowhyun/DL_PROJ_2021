import argparse
import random
import time
import os
import warnings
warnings.filterwarnings("ignore")

from copy import deepcopy
import pandas as pd
import numpy as np

import transformers
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

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
        model = BertClassifier(opt, freeze_bert=False)
        model.to(device)

        optimizer = AdamW(model.parameters(),
                          lr=opt.lr,
                          eps=opt.eps)
        # Total number of training steps
        total_steps = len_train_dataloader * opt.max_epoch

        # Set up the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)
        
    elif opt.model == "BILSTM":
        model = LstmClassifier(opt, 30522)
        model.to(device)

        optimizer = optim.SGD(model.parameters(), opt.lr,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay)
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 100*len_train_dataloader)
        
    
    return model, optimizer, scheduler


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
            if opt.model == "BILSTM":
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

def draw_embeddings(embeddings):
    return None

def vis_attention(embeddings):
    return None