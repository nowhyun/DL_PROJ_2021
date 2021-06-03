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
# Build Model
#################################################################################################################
# Create the BertClassifier class
class BertClassifier(nn.Module):
    def __init__(self, opt):
        """
        :param opt: hyper parameters
        :param freeze_bert: set 'False' to fine-tune the BERT model [bool]
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our cls, num_class
        self.bert = BertModel.from_pretrained('bert-base-uncased',
                                              output_hidden_states=True,output_attentions=True)
        self.opt = opt
        if opt.sent_embedding == 0:
            D_in, H, D_out = 768, opt.hidden_dim, 5
        if opt.sent_embedding == 1:
            D_in, H, D_out = 3072, opt.hidden_dim, 5

        self.classifier = nn.Sequential(nn.Linear(D_in, H), 
                                        nn.ReLU(),
                                        nn.Linear(H, D_out))
        self.dropout = nn.Dropout(p=opt.dropout)

        if opt.freeze_pretrained == 1:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_masks):
        """
        Feed input to BERT and the classifier to compute logits
        
        :param input_ids: an input tensor with shape (batch_size, max_length) [torch.Tensor]
        :param attention_masks: a tensor that hold attention mask [torch.Tensor]
        
        :return logits: an output tensor with shape (batch_size, num_class) [torch.Tensor]
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_masks)

        if self.opt.sent_embedding == 0:
            # Extract the last hidden state of the token '[CLS]'
            sent_embeddings_tsr = outputs[0][:, 0, :] # [batch_size, 768]
            sent_embeddings_tsr = self.dropout(sent_embeddings_tsr)

#             print(f"BertClassifier - forward - sent_embeddings_tsr.shape: {sent_embeddings_tsr.shape}")

        if self.opt.sent_embedding == 1:
            # Concatenate last 4 layers
            hidden_states_tuple = outputs[2]  # [13, batch_size, num. tokens, 768], [tuple]

            hidden_states_tsr = torch.stack(hidden_states_tuple,
                                            dim=0)  # each layer들을 stacking
            hidden_states_tsr = hidden_states_tsr.permute(1, 0, 2, 3) # [batch_size, 13, num. tokens, 768], [tensor]
#             print(f"BertClassifier - forward - hidden_states_tsr.shape: {hidden_states_tsr.shape}") 

            hidden_states_list = []  # [batch_size, 768*4]
            for hidden_state_tsr in hidden_states_tsr:
#                 print(f"\tBertClassifier - forward - hidden_state_tsr.shape: {hidden_state_tsr.shape}") # [13, num. tokens, 768], [tensor]
                concated_hidden_state_tsr = torch.cat(
                    (torch.mean(hidden_state_tsr[-1], dim=0), 
                     torch.mean(hidden_state_tsr[-2], dim=0),
                     torch.mean(hidden_state_tsr[-3], dim=0), 
                     torch.mean(hidden_state_tsr[-4], dim=0)), 
                    dim=0)
#                 print(f"\tBertClassifier - forward - concated_hidden_state_tsr.shape: {concated_hidden_state_tsr.shape}") # [4*768], [tensor]
                hidden_states_list.append(concated_hidden_state_tsr)
            sent_embeddings_tsr = torch.stack(hidden_states_list, dim=0)
#             print(f"BertClassifier - forward - sent_embeddings_tsr.shape: {sent_embeddings_tsr.shape}")

        # Feed input to classifier to compute logits
        logits = self.classifier(sent_embeddings_tsr)

        return logits
    
    def extract_sent_embd(self, input_ids, attention_masks):
        """After training is completed, extract the embdding vectors
        """
        # Feed input to ELECTRA
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_masks)
        embeddings_tsr = outputs[0] # [batch_size, max_len, 768] e.g. CLS = [:,0,:], 1st token = [:,1,:]
        return embeddings_tsr
    
    def extract_sent_feature(self, input_ids, attention_masks):
        """After training is completed, extract the embdding vectors
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_masks)
        feature_tsr = outputs # [batch_size, max_len, 768] e.g. CLS = [:,0,:], 1st token = [:,1,:]
        
        return feature_tsr