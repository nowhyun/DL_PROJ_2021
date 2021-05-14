import argparse
import random
import time
import os
import warnings
warnings.filterwarnings("ignore")

from copy import deepcopy
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use("ggplot")

import sklearn
from sklearn.model_selection import train_test_split

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


# Create a function to tokenize a set of texts
def preprocessing_for_bert(data, max_len):
    """
    Perform required preprocessing steps for pretrained BERT
    
    :param data: array of texts to be processed [array]
    :param max_len: max length for encoding [int]
    
    :return input_ids: tensor of token ids to be fed to a model [tensor]
    :return attention_masks: tensor of indices specifying wich tokens should be attended to by the model [tensor]
    """
    input_ids, attention_masks = [], []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # For every sentence
    for sent_str in data:
        # encode_plus will:
        # 1. Tokenize the sentence
        # 2. Add the '[CLS]', '[SEP]' token to the start and end
        # 3. Truncate/Pad sentence to max length
        # 4. Map tokens to their IDs
        # 5. Create attention mask
        # 6. Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
                text=sent_str,
                add_special_tokens=True,
                max_length=max_len,
                pad_to_max_length=True,
                return_attention_mask=True)
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    return input_ids, attention_masks


def data_load(opt, flag=None):
    """
    Load the datasets and Return the dataloaders
    
    :param opt: hyper parameters
    :param max_len: max length for encoding [int]
    
    :return train_dataloader, valid_dataloader, test_dataloader
    """
    if flag != None:
        full_dataset = FullDataset(opt)
        full_dataloader = DataLoader(full_dataset, batch_size=opt.batch_size)
        return full_dataloader
    train_dataset = TrainDataset(opt)
    if opt.balanced == 0:
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=opt.batch_size,
                                      shuffle=False)
    else:
        train_sampler = BalancedBatchSampler(train_dataset, opt.n_classes, opt.n_samples)
        train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, pin_memory=True)

    valid_dataset = ValidDataset(opt)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=opt.batch_size,
                                  shuffle=False)
    
    test_dataset = TestDataset(opt)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size)
    print(f"num of train_loader: {len(train_dataset)}")
    print(f"num of valid_loader: {len(valid_dataset)}")
    print(f"num of test_loader: {len(test_dataset)}")
    
    return train_dataloader, valid_dataloader, test_dataloader


class TrainDataset(Dataset):
    """Create the train_dataset
    """
    def __init__(self, opt):
        # Load
        train_dataset_df = pd.read_csv(opt.data_path +'/pre_train_opt' + str(opt.replace) + '.csv') # the training set with 11.5k sentences
        train_X_arr = train_dataset_df.Sentence.values
        train_y_arr = train_dataset_df.Category.values
        max_len = opt.max_len
        # Preprocessing
        # Run function `preprocessing_for_bert` on the train set and the validation set
        print('Tokenizing data...')
        train_X_ids_tsr, train_X_masks_tsr = preprocessing_for_bert(train_X_arr, max_len)
        print(f"train_X_ids_tsr.shape: {train_X_ids_tsr.shape}\ntrain_X_masks_tsr.shape: {train_X_masks_tsr.shape}")
        
        self.ids_tsr = train_X_ids_tsr
        self.masks_tsr = train_X_masks_tsr
        self.labels = torch.LongTensor(train_y_arr)

    def __getitem__(self, index):
        ids_tsr = self.ids_tsr[index]
        masks_tsr = self.masks_tsr[index]
        label_tsr = self.labels[index]
        return [ids_tsr, masks_tsr, label_tsr]

    def __len__(self):
        return len(self.ids_tsr)

    
class BalancedBatchSampler(BatchSampler):
    """Stratified sampler which can be used with train_dataset
    """
    def __init__(self, dataset, n_classes, n_samples):
        self.labels = dataset.labels
        self.labels_set = list(set(self.labels.numpy())) # unique label set
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set} # {label: index arr, label: index arr, ...}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_classes * self.n_samples
        print(f"BalancedBatchSampler - init - batch_size: {self.batch_size}")

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][self.used_label_indices_count[class_]:
                                                             self.used_label_indices_count[class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size
    

class ValidDataset(Dataset):
    """Create the valid_dataset
    """
    def __init__(self, opt):
        # Load
        valid_dataset_df = pd.read_csv(opt.data_path +'/pre_val_opt' + str(opt.replace) + '.csv') # the training set with 11.5k sentences
        valid_X_arr = valid_dataset_df.Sentence.values
        valid_y_arr = valid_dataset_df.Category.values
        max_len = opt.max_len
        # Preprocessing
        # Run function `preprocessing_for_bert` on the train set and the validation set
        print('Tokenizing data...')
        valid_X_ids_tsr, valid_X_masks_tsr = preprocessing_for_bert(valid_X_arr, max_len)
        print(f"valid_X_ids_tsr.shape: {valid_X_ids_tsr.shape}\nvalid_X_masks_tsr.shape: {valid_X_masks_tsr.shape}")
        
        self.ids_tsr = valid_X_ids_tsr
        self.masks_tsr = valid_X_masks_tsr
        self.labels = torch.LongTensor(valid_y_arr)

    def __getitem__(self, index):
        ids_tsr = self.ids_tsr[index]
        masks_tsr = self.masks_tsr[index]
        label_tsr = self.labels[index]
        return [ids_tsr, masks_tsr, label_tsr]

    def __len__(self):
        return len(self.ids_tsr)
    

class TestDataset(Dataset):
    """Create the test_dataset
    """
    def __init__(self, opt):
        # Load
        test_dataset_df = pd.read_csv(opt.data_path +'/eval_final_open.csv') # the test set with 4.3k sentences
        test_X_arr = test_dataset_df.Sentence.values
        max_len = opt.max_len
        # Preprocessing
        # Run function `preprocessing_for_bert` on the train set and the validation set
        print('Tokenizing data...')
        test_X_ids_tsr, test_X_masks_tsr = preprocessing_for_bert(test_X_arr, max_len)
        print(f"test_X_ids_tsr.shape: {test_X_ids_tsr.shape}\ntest_X_masks_tsr.shape: {test_X_masks_tsr.shape}")
        
        self.ids_tsr = test_X_ids_tsr
        self.masks_tsr = test_X_masks_tsr

    def __getitem__(self, index):
        ids_tsr = self.ids_tsr[index]
        masks_tsr = self.masks_tsr[index]
        return [ids_tsr, masks_tsr]

    def __len__(self):
        return len(self.ids_tsr)

    
class FullDataset(Dataset):
    """Create the full_dataset
    """
    def __init__(self, opt):
        # Load
        train_dataset_df = pd.read_csv(opt.data_path +'/pre_train_opt' + str(opt.replace) + '.csv') # the training set with 11.5k sentences
        valid_dataset_df = pd.read_csv(opt.data_path +'/pre_val_opt' + str(opt.replace) + '.csv') # the training set with 11.5k sentences
        full_dataset_df = pd.concat([train_dataset_df, valid_dataset_df])
        full_X_arr = full_dataset_df.Sentence.values
        full_y_arr = full_dataset_df.Category.values
        max_len = opt.max_len
        # Preprocessing
        # Run function `preprocessing_for_bert` on the train set and the validation set
        print('Tokenizing data...')
        full_X_ids_tsr, full_X_masks_tsr = preprocessing_for_bert(full_X_arr, max_len)
        print(f"full_X_ids_tsr.shape: {full_X_ids_tsr.shape}\nfull_X_masks_tsr.shape: {full_X_masks_tsr.shape}")
        
        self.ids_tsr = full_X_ids_tsr
        self.masks_tsr = full_X_masks_tsr
        self.labels = torch.LongTensor(full_y_arr)

    def __getitem__(self, index):
        ids_tsr = self.ids_tsr[index]
        masks_tsr = self.masks_tsr[index]
        label_tsr = self.labels[index]
        return [ids_tsr, masks_tsr, label_tsr]

    def __len__(self):
        return len(self.ids_tsr)