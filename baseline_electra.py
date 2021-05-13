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
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import ElectraTokenizer, ElectraModel, ElectraConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

import torchvision
from torchvision import transforms, datasets

#################################################################################################################
# Library Version
#################################################################################################################
print(f"pandas version: {pd.__version__}")
print(f"numpy version: {np.__version__}")
print(f"matplotlib version: {mpl.__version__}")
print(f"sklearn version: {sklearn.__version__}")
print(f"transformers version: {transformers.__version__}")
print(f"torch version: {torch.__version__}")

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
# Hyperparameters Setting
#################################################################################################################
parser = argparse.ArgumentParser()

# model
parser.add_argument('--model', type=str, default='ELECTRA', help='BERT, BILSTM, ELECTRA')
parser.add_argument('--sent_embedding', type=int, default=0, help='0: CLS, 1: 4-layer concat')
parser.add_argument('--hidden_dim', type=int, default=768, help='for wide models')

# training
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--gpu', type=int, default=0, help='0,1,2,3')
parser.add_argument('--max_epoch', type=int, default=5)
parser.add_argument('--save', type=int, default=1, help='0: false, 1:true')

parser.add_argument('--optimizer', type=int, default=1, help='1: SGD, 2: RMSProp, 3: Adam')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate, 5e-5, 3e-5 or 2e-5')
parser.add_argument('--eps', type=float, default=1e-8, help='epsilon')

# dataset
parser.add_argument('--data_path', type=str, default='/home/jangyj0426/workspace/DL_PROJ_2021/Dataset/')
parser.add_argument('--save_model_path', type=str, default='/home/jangyj0426/workspace/DL_PROJ_2021/Saved_models')
parser.add_argument('--save_submission_path', type=str, default='/home/jangyj0426/workspace/DL_PROJ_2021/Submissions')
parser.add_argument('--author', type=str, default='yj')
parser.add_argument('--valid_ratio', type=float, default=1/6)

#     opt = parser.parse_args() # in .py env
opt, _ = parser.parse_known_args() # in .ipynb env

#################################################################################################################
# Training Device
#################################################################################################################
device = torch.device("cuda:" + str(opt.gpu)) if torch.cuda.is_available() else torch.device("cpu")
torch.cuda.set_device(device) # change allocation of current GPU
print(f'training device: {device, torch.cuda.get_device_name()}')
signature = str(opt.author) + "_" + str(opt.model) + "_" + str(opt.sent_embedding) + "_" + str(opt.hidden_dim) + "_" + str(opt.batch_size) + "_" + str(opt.max_epoch) + "_" + str(opt.lr) + "_" + str(opt.eps) 
print(f'signature: {signature}')

#################################################################################################################
# Data Load
#################################################################################################################
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

def data_load(opt, max_len):
    """
    Load, Split and Create the dataloaders
    
    :param opt: hyper parameters
    :param max_len: max length for encoding [int]
    
    :return train_dataloader, valid_dataloader, test_dataloader
    """
    # Load
    full_dataset_df = pd.read_csv(opt.data_path +'/train_final.csv') # the training set with 11.5k sentences
    test_dataset_df = pd.read_csv(opt.data_path +'/eval_final_open.csv') # the test set with 4.3k sentences
    print(f"num. train dataset: {len(full_dataset_df)}, num. test dataset: {len(test_dataset_df)}")
    
    # Train Valid Split
    full_X_arr = full_dataset_df.Sentence.values
    full_y_arr = full_dataset_df.Category.values
    train_X_arr, valid_X_arr, train_y_arr, valid_y_arr = train_test_split(full_X_arr, full_y_arr, test_size=opt.valid_ratio, random_state=42, shuffle=True, stratify=full_y_arr)
    print(f"train_X_arr shape: {train_X_arr.shape}, train_y_arr shape: {train_y_arr.shape}, valid_X_arr shape: {valid_X_arr.shape}, valid_y_arr shape: {valid_y_arr.shape}")
    test_X_arr = test_dataset_df.Sentence.values

    # Preprocessing
    if opt.model.lower() == "bert":
        # Run function `preprocessing_for_bert` on the train set and the validation set
        print('Tokenizing data...')
        train_X_ids_tsr, train_X_masks_tsr = preprocessing_for_bert(train_X_arr, max_len)
        valid_X_ids_tsr, valid_X_masks_tsr = preprocessing_for_bert(valid_X_arr, max_len)
        test_X_ids_tsr, test_X_masks_tsr = preprocessing_for_bert(test_X_arr, max_len)
        print(f"train_X_ids_tsr.shape: {train_X_ids_tsr.shape}\ntrain_X_masks_tsr.shape: {train_X_masks_tsr.shape}\nvalid_X_ids_tsr.shape: {valid_X_ids_tsr.shape}\nvalid_X_masks_tsr.shape: {valid_X_masks_tsr.shape}")

    if opt.model.lower() == "electra":
        # Run function `preprocessing_for_bert` on the train set and the validation set
        print('Tokenizing data...')
        train_X_ids_tsr, train_X_masks_tsr = preprocessing_for_bert(train_X_arr, max_len)
        valid_X_ids_tsr, valid_X_masks_tsr = preprocessing_for_bert(valid_X_arr, max_len)
        test_X_ids_tsr, test_X_masks_tsr = preprocessing_for_bert(test_X_arr, max_len)
        print(f"train_X_ids_tsr.shape: {train_X_ids_tsr.shape}\ntrain_X_masks_tsr.shape: {train_X_masks_tsr.shape}\nvalid_X_ids_tsr.shape: {valid_X_ids_tsr.shape}\nvalid_X_masks_tsr.shape: {valid_X_masks_tsr.shape}")
    
    # Convert other data types to torch.Tensor
    train_y_tsr = torch.tensor(train_y_arr)
    valid_y_tsr = torch.tensor(valid_y_arr)

    # Create the DataLoader
    train_dataset = TensorDataset(train_X_ids_tsr, train_X_masks_tsr, train_y_tsr)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=opt.batch_size,
                                  shuffle=True)

    valid_dataset = TensorDataset(valid_X_ids_tsr, valid_X_masks_tsr, valid_y_tsr)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=opt.batch_size,
                                  shuffle=True)
    
    test_dataset = TensorDataset(test_X_ids_tsr, test_X_masks_tsr)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size)
    print(f"num of train_loader: {len(train_dataset)}")
    print(f"num of valid_loader: {len(valid_dataset)}")
    
    return train_dataloader, valid_dataloader, test_dataloader

#################################################################################################################
# Build Model
#################################################################################################################
# Create the BertClassifier class
class ElectraClassifier(nn.Module):
    def __init__(self, opt, freeze_param=False):
        """
        :param opt: hyper parameters
        :param freeze_bert: set 'False' to fine-tune the BERT model [bool]
        """
        super(ElectraClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our cls, num_class
        self.electra = ElectraModel.from_pretrained('google/electra-base-discriminator',
                                              output_hidden_states=True)
        if opt.sent_embedding == 0:
            D_in, H, D_out = 768, opt.hidden_dim, 5
        # if opt.sent_embedding == 1:
        #     D_in, H, D_out = 3072, opt.hidden_dim, 5

        self.classifier = nn.Sequential(nn.Linear(D_in, H), 
                                        nn.ReLU(),
                                        nn.Linear(H, D_out))

        if freeze_param:
            for param in self.electra.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_masks):
        """
        Feed input to BERT and the classifier to compute logits
        
        :param input_ids: an input tensor with shape (batch_size, max_length) [torch.Tensor]
        :param attention_masks: a tensor that hold attention mask [torch.Tensor]
        
        :return logits: an output tensor with shape (batch_size, num_class) [torch.Tensor]
        """
        # Feed input to BERT
        outputs = self.electra(input_ids=input_ids,
                            attention_mask=attention_masks)

        if opt.sent_embedding == 0:
            # Extract the last hidden state of the token '[CLS]'
            sent_embeddings_tsr = outputs[0][:, 0, :] # [batch_size, 768]
#             print(f"BertClassifier - forward - sent_embeddings_tsr.shape: {sent_embeddings_tsr.shape}")

        if opt.sent_embedding == 1:
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
        logits = self.classifier(sent_embeddings_tsr) # 16*768
        # print(logits)

        return logits
    
# To fine-tune our Bert Cls, we need to create an optimizer
# The authors recommend following hyper-parameters
# Batch size: 16 or 32
# Learning rate (Adam): 5e-5, 3e-5 or 2e-5
# Number of epochs: 2, 3, 4
def initialize_model(opt):
    """
    Initialize the Bert Cls, the optimizer and the learning rate scheduler
    
    :param opt: hyper parameters
    
    :return bert_classifier, optimizer, scheduler
    """
    electra_classifier = ElectraClassifier(opt, freeze_param=False)
    electra_classifier.to(device)
    
    optimizer = AdamW(electra_classifier.parameters(),
                      lr=opt.lr,
                      eps=opt.eps)
    # Total number of training steps
    total_steps = len(train_dataloader) * opt.max_epoch
    
    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    return electra_classifier, optimizer, scheduler

# encoded_doc = [tokenizer.encode(sent, add_special_tokens=True) for sent in full_X_arr]
# MAX_LEN = max([len(sent) for sent in encoded_doc])
MAX_LEN = 50

# Load the ELECTRA tokenizer
tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')

train_dataloader, valid_dataloader, test_dataloader = data_load(opt, MAX_LEN)

#################################################################################################################
# Train and Evaluate
#################################################################################################################
def train(model, train_dataloader, valid_dataloader=None, epochs=4, evaluation=False):
    """
    Train the BertClassifier model with early stop trick.
    
    :param model: untrained model
    :param train_dataloader: dataloader which is obtained by data_load method
    :param valid_dataloader: dataloader which is obtained by data_load method
    :param epochs: opt.max_epoch [int]
    :param evaluation: [bool]
    """
    # Start training loop
    print("Start training...\n")
    es_eval_dict = {"epoch": 0, "train_loss": 0, "valid_loss": 0, "valid_acc": 0} # early stop
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            b_ids_tsr, b_masks_tsr, b_labels_tsr = tuple(tsrs.to(device) for tsrs in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_ids_tsr, b_masks_tsr)
            
            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels_tsr)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)
        # =======================================
        #               Evaluation
        # =======================================
        model_save_path = str(opt.save_model_path) + "/" + signature +'.model'
        if evaluation == True:
            previous_valid_acc = es_eval_dict["valid_acc"] # early stop
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            valid_loss, valid_acc = evaluate(model, valid_dataloader)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {valid_loss:^10.6f} | {valid_acc:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
            if previous_valid_acc < valid_acc:
                es_eval_dict["epoch"]=epoch_i
                es_eval_dict["train_loss"]=avg_train_loss
                es_eval_dict["valid_loss"]=valid_loss
                es_eval_dict["valid_acc"]=valid_acc
                if opt.save == 1: 
                    torch.save(model.state_dict(), model_save_path)
                    print('\tthe model is improved... save at', model_save_path)
        print("\n")
    print("Final results table")
    print("-"*70)
    print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
    final_epoch, final_train_loss, final_valid_loss, final_valid_acc = es_eval_dict["epoch"], es_eval_dict["train_loss"], es_eval_dict["valid_loss"], es_eval_dict["valid_acc"]
    print(f"{final_epoch + 1:^7} | {'-':^7} | {final_train_loss:^12.6f} | {final_valid_loss:^10.6f} | {final_valid_acc:^9.2f} | {0:^9.2f}")
    print("-"*70)
    print("Training complete!")

def evaluate(model, valid_dataloader):
    """
    After the completion of each training epoch, measure the model's performance on our validation set.
    
    :param model: trained model
    :param valid_dataloader: dataloader which is obtained by data_load method
    
    :return valid_loss: validation loss [array]
    :return valid_acc: validation accuracy [array]
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    valid_acc = []
    valid_loss = []

    # For each batch in our validation set...
    for batch in valid_dataloader:
        # Load batch to GPU
        b_ids_tsr, b_masks_tsr, b_labels_tsr = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_ids_tsr, b_masks_tsr)

        # Compute loss
        loss = loss_fn(logits, b_labels_tsr)
        valid_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels_tsr).cpu().numpy().mean() * 100
        valid_acc.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    valid_loss = np.mean(valid_loss)
    valid_acc = np.mean(valid_acc)

    return valid_loss, valid_acc

loss_fn = nn.CrossEntropyLoss() # Specify loss function
electra_classifier, optimizer, scheduler = initialize_model(opt)
train(electra_classifier, train_dataloader, valid_dataloader, epochs=opt.max_epoch, evaluation=True)

################################################################################################################
# Make Submission
################################################################################################################
def bert_predict(model, test_dataloader):
    """
    Perform a forward pass on the trained BERT model to predict probabilities on the test set.
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
            logits = model(b_ids_tsr, b_masks_tsr)
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()
    preds = np.argmax(probs, axis=1)
    return preds

def make_submission(model, opt, test_dataloader):
    """
    Save a submission.csv
    
    :param model: trained model
    :param opt: hyper parameters
    :param test_dataloader: dataloader which is obtained by data_load method
    """
    submission_templete_df = pd.read_csv(opt.data_path +'/sample_sub.csv') # a sample submission file in the correct format
    
    # Early stop model
    if opt.save == 1:
        model_save_path = str(opt.save_model_path) + "/" + signature +'.model'
        model.load_state_dict(torch.load(model_save_path))
    preds = bert_predict(model, test_dataloader)
    
    assert len(preds) == 4311
    submission_templete_df.Category = preds
    print(submission_templete_df)
    submission_save_path = str(opt.save_submission_path) + "/" + signature +'.csv'
    print(f"\t...Save complete at {submission_save_path}")
    submission_templete_df.to_csv(submission_save_path, index=False)

make_submission(electra_classifier, opt, test_dataloader)

num_of_total_params = sum(p.numel() for p in electra_classifier.parameters())
print(num_of_total_params)