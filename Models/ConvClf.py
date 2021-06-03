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
class ConvClassifier(nn.Module):
    def __init__(self, opt, vocab_size=30522):
        """
        :param opt: hyper parameters
        """
        super(ConvClassifier, self).__init__()
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=opt.embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=opt.embedding_dim,
                      out_channels=opt.kernel_depth,
                      kernel_size=opt.kernel_sizes[i])
            for i in range(len(opt.kernel_sizes))
        ])
        
        
        self.drop = nn.Dropout(p=opt.dropout) # opt.dropout

        self.fc = nn.Linear(len(opt.kernel_sizes) * opt.kernel_depth, 5)

    def forward(self, input_ids):
        """
        Feed input to embedding layer and the classifier to compute logits
        
        :param input_ids: an input tensor with shape (batch_size, max_length) [torch.Tensor]
        
        :return logits: an output tensor with shape (batch_size, num_class) [torch.Tensor]
        """
        sent_embeddings_tsr = self.embedding(input_ids)
        sent_embeddings_tsr = self.drop(sent_embeddings_tsr)
        sent_embeddings_tsr = sent_embeddings_tsr.permute(0, 2, 1)
        sent_embeddings_tsrs = [F.relu(conv1d(sent_embeddings_tsr)) for conv1d in self.convs]
        sent_embeddings_tsrs = [F.max_pool1d(sent_embeddings_tsr, kernel_size=sent_embeddings_tsr.shape[2]) for sent_embeddings_tsr in sent_embeddings_tsrs]
        sent_embeddings_tsr = torch.cat([sent_embeddings_tsr.squeeze(dim=2) for sent_embeddings_tsr in sent_embeddings_tsrs], dim=1)
        
        logits = self.fc(sent_embeddings_tsr)

        return logits

    def extract_sent_embd(self, input_ids):
        """After training is completed, extract the embdding vectors
        """
        sent_embeddings_tsr = self.embedding(input_ids)
        sent_embeddings_tsr = self.drop(sent_embeddings_tsr)
        sent_embeddings_tsr = sent_embeddings_tsr.permute(0, 2, 1)
        sent_embeddings_tsrs = [F.relu(conv1d(sent_embeddings_tsr)) for conv1d in self.convs]
        sent_embeddings_tsrs = [F.max_pool1d(sent_embeddings_tsr, kernel_size=sent_embeddings_tsr.shape[2]) for sent_embeddings_tsr in sent_embeddings_tsrs]
        sent_embeddings_tsr = torch.cat([sent_embeddings_tsr.squeeze(dim=2) for sent_embeddings_tsr in sent_embeddings_tsrs], dim=1)
        return sent_embeddings_tsr
    
    def extract_sent_feature(self, input_ids):
        """After training is completed, extract the embdding vectors
        """
        # Feed input to CNN
        sent_embeddings_tsr = self.embedding(input_ids)
        sent_embeddings_tsr = sent_embeddings_tsr.permute(0, 2, 1)
        sent_embeddings_tsrs = [F.relu(conv1d(sent_embeddings_tsr)) for conv1d in self.convs]
        sent_embeddings_tsrs = [F.max_pool1d(sent_embeddings_tsr, kernel_size=sent_embeddings_tsr.shape[2]) for sent_embeddings_tsr in sent_embeddings_tsrs]
        sent_embeddings_tsr = torch.cat([sent_embeddings_tsr.squeeze(dim=2) for sent_embeddings_tsr in sent_embeddings_tsrs], dim=1)
        
        
        return sent_embeddings_tsr
    
    
    def attention_net(self, lstm_output, final_hidden_state):

        """ 
        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.
        Arguments
        ---------
        lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        final_state : Final time-step hidden state (h_n) of the LSTM
        ---------

        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
         new hidden state.

        Tensor Size :
        hidden.size() = (batch_size, hidden_size)
        attn_weights.size() = (batch_size, num_seq)
        soft_attn_weights.size() = (batch_size, num_seq)
        new_hidden_state.size() = (batch_size, hidden_size)

        """
        hidden = final_hidden_state
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        #new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return soft_attn_weights