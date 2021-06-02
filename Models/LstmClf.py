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
# Create the LstmClassifier class
class LstmClassifier(nn.Module):
    def __init__(self, opt, vocab_size=30522):
        """
        :param opt: hyper parameters
        """
        super(LstmClassifier, self).__init__()

        self.embedding = nn.Embedding(vocab_size, opt.max_len)
        self.lstm = nn.LSTM(input_size=opt.max_len,
                            hidden_size=opt.hidden_dim,
                            num_layers=opt.num_layer, # opt.num_layer
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=opt.dropout) # opt.dropout

        # fc original
        # self.fc = nn.Linear(2*opt.hidden_dim, 5)
        # fc wo dropout
        self.fc = nn.Sequential(nn.Linear(2*opt.hidden_dim, opt.hidden_dim), 
                                        nn.ReLU(),
                                        nn.Linear(opt.hidden_dim, 5))
        # # fc w dropout
        # self.fc = nn.Sequential(nn.Linear(2*opt.hidden_dim, opt.hidden_dim), 
        #                                 nn.ReLU(),
        #                                 self.drop,
        #                                 nn.Linear(opt.hidden_dim, 5))


    def forward(self, input_ids):
        """
        Feed input to embedding layer and the classifier to compute logits
        
        :param input_ids: an input tensor with shape (batch_size, max_length) [torch.Tensor]
        
        :return logits: an output tensor with shape (batch_size, num_class) [torch.Tensor]
        """
        # Feed input to embedding layer
        sent_embeddings_tsr = self.embedding(input_ids) # [batch_size, max_len, embedding_dim] ? 
#         print(f"LstmClassifier - forward - sent_embeddings_tsr.shape: {sent_embeddings_tsr.shape}")
        sent_embeddings_tsr = self.drop(sent_embeddings_tsr)

        # Feed embedding to Lstm
        output, (hidden_states_tsr, cell_states_tsr) = self.lstm(sent_embeddings_tsr) # hidden_states_tsr = [2, batch_size, hidden_size]
#         print(f"LstmClassifier - forward - hidden_states_tsr.shape: {hidden_states_tsr.shape}")

        # Feed input to classifier to compute logits
        right_hidden_states_tsr = hidden_states_tsr[-1] # [batch_size, hidden_size]
#         print(f"LstmClassifier - forward - right_hidden_states_tsr.shape: {right_hidden_states_tsr.shape}")
        left_hidden_states_tsr = hidden_states_tsr[-2] # [batch_size, hidden_size]
#         print(f"LstmClassifier - forward - left_hidden_states_tsr.shape: {left_hidden_states_tsr.shape}")
        final_hidden_states_tsr = torch.cat((right_hidden_states_tsr, left_hidden_states_tsr), dim=1)  # [batch_size, 2*hidden_size]
#         print(f"LstmClassifier - forward - final_hidden_states_tsr.shape: {final_hidden_states_tsr.shape}")

        logits = self.fc(final_hidden_states_tsr) # [batch_size, 5]
#         print(f"LstmClassifier - forward - logits.shape: {logits.shape}")

        return logits

    def extract_sent_embd(self, input_ids):
        """After training is completed, extract the embdding vectors
        """
        # Feed input to embedding layer
        sent_embeddings_tsr = self.embedding(input_ids) # [batch_size, max_len, embedding_dim] ? 
        sent_embeddings_tsr = self.drop(sent_embeddings_tsr)

        # Feed embedding to Lstm
        output, (hidden_states_tsr, cell_states_tsr) = self.lstm(sent_embeddings_tsr) # hidden_states_tsr = [2, batch_size, hidden_size]

        # Feed input to classifier to compute logits
        right_hidden_states_tsr = hidden_states_tsr[-1] # [batch_size, hidden_size]
        left_hidden_states_tsr = hidden_states_tsr[-2] # [batch_size, hidden_size]
        final_hidden_states_tsr = torch.cat((right_hidden_states_tsr, left_hidden_states_tsr), dim=1)  # [batch_size, 2*hidden_size]
        return final_hidden_states_tsr