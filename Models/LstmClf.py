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
    def __init__(self, opt, vocab_size):
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

        self.fc = nn.Linear(2*opt.hidden_dim, 5)

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
    
    def extract_sent_feature(self, input_ids):
        """After training is completed, extract the embdding vectors
        """
        # Feed input to embedding layer
        sent_embeddings_tsr = self.embedding(input_ids) # [batch_size, max_len, embedding_dim] ? 
        
        # Feed embedding to Lstm
        output, (hidden_states_tsr, cell_states_tsr) = self.lstm(sent_embeddings_tsr) 
        # output = [batch,seq_len,hidden_size*direction_num]
        #hidden_states_tsr = [layer_num*direction_num, batch_size, hidden_size]
        #cell_states_tsr = [layer_num*direction_num, batch_size, hidden_size]

        # Feed input to classifier to compute logits
        backward_hidden_states_tsr = hidden_states_tsr[-1] 
        # backward direction hidden state(feature for sentence) of last layer [batch_size, hidden_size]
        forward_hidden_states_tsr = hidden_states_tsr[-2] 
        # forward direction hidden state(feature for sentence) of last layer[batch_size, hidden_size]
        final_hidden_states_tsr = torch.cat((forward_hidden_states_tsr, backward_hidden_states_tsr), dim=1)  # [batch_size, 2*hidden_size]

        return output, final_hidden_states_tsr
    
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