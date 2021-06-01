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
from transformers import ElectraTokenizer, ElectraModel, AdamW, get_linear_schedule_with_warmup

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

import torchvision
from torchvision import transforms, datasets

# from Utils.tokenizer import Tokenizer

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
    if opt.train_val_split:
        # Load
        full_dataset_df = pd.read_csv('/home/jangyj0426/workspace/DL_PROJ_2021/Dataset/train_final.csv') # the training set with 11.5k sentences
        test_dataset_df = pd.read_csv('/home/jangyj0426/workspace/DL_PROJ_2021/Dataset/eval_final_open.csv') # the test set with 4.3k sentences
        print(f"num. train dataset: {len(full_dataset_df)}, num. test dataset: {len(test_dataset_df)}")
        
        # Train Valid Split
        full_X_arr = full_dataset_df.Sentence.values
        full_y_arr = full_dataset_df.Category.values
        train_X_arr, valid_X_arr, train_y_arr, valid_y_arr = train_test_split(full_X_arr, full_y_arr, test_size=opt.valid_ratio, random_state=42, shuffle=True, stratify=full_y_arr)
        print(f"train_X_arr shape: {train_X_arr.shape}, train_y_arr shape: {train_y_arr.shape}, valid_X_arr shape: {valid_X_arr.shape}, valid_y_arr shape: {valid_y_arr.shape}")
        test_X_arr = test_dataset_df.Sentence.values
    else:
        # Load
        train_dataset_df = pd.read_csv('/home/jangyj0426/workspace/DL_PROJ_2021/Dataset/train_dd_ratio_{}.csv'.format(opt.ratio)) 
        valid_dataset_df = pd.read_csv('/home/jangyj0426/workspace/DL_PROJ_2021/Dataset/valid_dd_ratio_{}.csv'.format(opt.ratio)) 
        test_dataset_df = pd.read_csv('/home/jangyj0426/workspace/DL_PROJ_2021/Dataset/eval_final_open.csv') # the test set with 4.3k sentences
        # print(f"num. train dataset: {len(full_dataset_df)}, num. test dataset: {len(test_dataset_df)}")
        
        # Train Valid Split
        train_X_arr = train_dataset_df.Sentence.values
        train_y_arr = train_dataset_df.Category.values
        valid_X_arr = valid_dataset_df.Sentence.values
        valid_y_arr = valid_dataset_df.Category.values
        # train_X_arr, valid_X_arr, train_y_arr, valid_y_arr = train_test_split(full_X_arr, full_y_arr, test_size=opt.valid_ratio, random_state=42, shuffle=True, stratify=full_y_arr)
        print(f"train_X_arr shape: {train_X_arr.shape}, train_y_arr shape: {train_y_arr.shape}, valid_X_arr shape: {valid_X_arr.shape}, valid_y_arr shape: {valid_y_arr.shape}")
        test_X_arr = test_dataset_df.Sentence.values

    # Preprocessing
    # Run function `preprocessing_for_bert` on the train set and the validation set
    print('Tokenizing data...')
    train_X_ids_tsr, train_X_masks_tsr = preprocessing_for_bert(train_X_arr, max_len)
    valid_X_ids_tsr, valid_X_masks_tsr = preprocessing_for_bert(valid_X_arr, max_len)
    test_X_ids_tsr, test_X_masks_tsr = preprocessing_for_bert(test_X_arr, max_len)
    print(f"train_X_ids_tsr.shape: {train_X_ids_tsr.shape}\n\nvalid_X_ids_tsr.shape: {valid_X_ids_tsr.shape}\n")
    
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
# LSTM Model
#######################################F##########################################################################
# Create the LstmClassifier class
class LstmClassifier(nn.Module):
    def __init__(self, opt):
        """
        :param opt: hyper parameters
        """
        super(LstmClassifier, self).__init__()

        self.embedding = nn.Embedding(tokenizer.vocab_size, opt.max_len)
        self.lstm = nn.LSTM(input_size=opt.max_len,
                            hidden_size=opt.hidden_dim,
                            num_layers=opt.num_layer, # opt.num_layer
                            batch_first=True,
                            bidirectional=True)
        self.drop = nn.Dropout(p=opt.dropout) # opt.dropout

        self.fc = nn.Sequential(nn.Linear(opt.hidden_dim*2, opt.hidden_dim), 
                                        nn.ReLU(),
                                        nn.Linear(opt.hidden_dim, 5))
      

    def forward(self, input_ids):
        """
        Feed input to embedding layer and the classifier to compute logits
        
        :param input_ids: an input tensor with shape (batch_size, max_length) [torch.Tensor]
        
        :return logits: an output tensor with shape (batch_size, num_class) [torch.Tensor]
        """
        # Feed input to embedding layer
        sent_embeddings_tsr = self.embedding(input_ids) # [batch_size, max_len, embedding_dim] ? 
        # print(f"LstmClassifier - forward - sent_embeddings_tsr.shape: {sent_embeddings_tsr.shape}")
        sent_embeddings_tsr = self.drop(sent_embeddings_tsr)

        # Feed embedding to Lstm
        output, (hidden_states_tsr, cell_states_tsr) = self.lstm(sent_embeddings_tsr) # hidden_states_tsr = [2, batch_size, hidden_size]
        # print(f"LstmClassifier - forward - hidden_states_tsr.shape: {hidden_states_tsr.shape}")

        # Feed input to classifier to compute logits
        right_hidden_states_tsr = hidden_states_tsr[-1] # [batch_size, hidden_size]
        # print(f"LstmClassifier - forward - right_hidden_states_tsr.shape: {right_hidden_states_tsr.shape}")
        left_hidden_states_tsr = hidden_states_tsr[-2] # [batch_size, hidden_size]
        # print(f"LstmClassifier - forward - left_hidden_states_tsr.shape: {left_hidden_states_tsr.shape}")
        final_hidden_states_tsr = torch.cat((right_hidden_states_tsr, left_hidden_states_tsr), dim=1)  # [batch_size, 2*hidden_size]
        # print(f"LstmClassifier - forward - final_hidden_states_tsr.shape: {final_hidden_states_tsr.shape}")

        logits = self.fc(final_hidden_states_tsr) # [batch_size, 5]
        # print(f"LstmClassifier - forward - logits.shape: {logits.shape}")

        return logits

### ELECTRA
# Create the ElectraClassifier class
class ElectraClassifier(nn.Module):
    def __init__(self, opt, freeze_electra=False):
        """
        :param opt: hyper parameters
        :param freeze_electra: set 'False' to fine-tune the ELECTRA model [bool]
        """
        super(ElectraClassifier, self).__init__()
        # Specify hidden size of EL:ECTRA, hidden size of our cls, num_class
        self.electra = ElectraModel.from_pretrained('google/electra-base-discriminator',
                                              output_hidden_states=True)
        if opt.sent_embedding == 0:
            D_in, H, D_out = 768, opt.hidden_dim, 5
        if opt.sent_embedding == 1:
            D_in, H, D_out = 3072, opt.hidden_dim, 5

        self.classifier = nn.Sequential(nn.Linear(D_in, H), 
                                        nn.ReLU(),
                                        nn.Linear(H, D_out))

        if freeze_electra:
            for param in self.electra.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_masks):
        """
        Feed input to ELECTRA and the classifier to compute logits
        
        :param input_ids: an input tensor with shape (batch_size, max_length) [torch.Tensor]
        :param attention_masks: a tensor that hold attention mask [torch.Tensor]
        
        :return logits: an output tensor with shape (batch_size, num_class) [torch.Tensor]
        """
        # Feed input to ELECTRA
        outputs = self.electra(input_ids=input_ids,
                            attention_mask=attention_masks)

        # if opt.sent_embedding == 0:
            # Extract the last hidden state of the token '[CLS]'
        sent_embeddings_tsr = outputs[0][:, 0, :] # [batch_size, 768]
#             print(f"electraClassifier - forward - sent_embeddings_tsr.shape: {sent_embeddings_tsr.shape}")

        # Feed input to classifier to compute logits
        logits = self.classifier(sent_embeddings_tsr)

        return logits


def initialize_model(opt):
    """
    Initialize the Lstm Cls, the optimizer and the learning rate scheduler
    
    :param opt: hyper parameters
    
    :return lstm_classifier, optimizer, scheduler
    """
    lstm_classifier = LstmClassifier(opt)
    lstm_classifier.to(device)

    lstm_optimizer = AdamW(lstm_classifier.parameters(), lr=opt.lr, eps=opt.eps)

    electra_classifier = ElectraClassifier(opt, freeze_electra=False)
    electra_classifier.to(device)
    
    electra_optimizer = AdamW(electra_classifier.parameters(),
                      lr=opt.lr,
                      eps=opt.eps)
    # Total number of training steps
    total_steps = len(train_dataloader) * opt.max_epoch
    
    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(electra_optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    return lstm_classifier, lstm_optimizer, electra_classifier, electra_optimizer, scheduler


### Train
def train(model, train_dataloader, valid_dataloader=None, epochs=4, evaluation=False, lstm=False):
    """
    Train the ElectraClassifier model with early stop trick.
    
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

            if lstm:

                # Perform a forward pass. This will return logits.
                logits = model(b_ids_tsr)
            else:

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

            if lstm:
                # Update parameters and the learning rate
                lstm_optimizer.step()
            else:
                electra_optimizer.step()
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
        if lstm:
            model_save_path = str(opt.save_model_path) + "/" + signature + "_lstm_" + '.model'
        else:
            model_save_path = str(opt.save_model_path) + "/" + signature + "_electra_" + '.model'
        if evaluation == True:
            previous_valid_acc = es_eval_dict["valid_acc"] # early stop
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            valid_loss, valid_acc = evaluate(model, valid_dataloader, lstm)

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

def evaluate(model, valid_dataloader, lstm=False):
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
        b_ids_tsr, b_masks_tsr, b_labels_tsr = tuple(t.to(device) for t in batch)
        if lstm:

            # Compute logits
            with torch.no_grad():
                logits = model(b_ids_tsr)
        else:
            # Load batch to GPU

            # Compute logits
            with torch.no_grad():
                logits = model(b_ids_tsr, b_masks_tsr)

        # Compute loss
        loss = loss_fn(logits, b_labels_tsr)
        valid_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()
        # print(preds)

        # Calculate the accuracy rate
        accuracy = (preds == b_labels_tsr).cpu().numpy().mean() * 100
        valid_acc.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    valid_loss = np.mean(valid_loss)
    valid_acc = np.mean(valid_acc)

    return valid_loss, valid_acc


################################################################################################################
# Make Submission
################################################################################################################
def predict(lstm_model, electra_model, test_dataloader):
    """
    Perform a forward pass on the trained Electra model to predict probabilities on the test set.
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
    lstm_model.eval()
    electra_model.eval()
    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_ids_tsr, b_masks_tsr = tuple(tsr.to(device) for tsr in batch)[:2]

        # Compute logits
        with torch.no_grad():
            lstm_logits = lstm_model(b_ids_tsr)
            electra_logits = electra_model(b_ids_tsr, b_masks_tsr)

        # trial_1
        # print(lstm_logits.shape)
        # print(electra_logits.shape)
        logits = []
        for i in range(electra_logits.shape[0]):
            # print(lstm_logits[i].shape, lstm_logits[i])
            lstm_ls = lstm_logits[i].tolist()
            lstm_ls.sort(reverse=True)
            lstm_diff = lstm_ls[0] - lstm_ls[1]
            electra_ls = electra_logits[i].tolist()
            electra_ls.sort(reverse=True)
            electra_diff = electra_ls[0] - electra_ls[1]
            if lstm_diff>electra_diff:
                logits.append(lstm_logits[i].tolist())
            else:
                logits.append(electra_logits[i].tolist())
            # print(logits)
        logits = torch.Tensor(logits)
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()
    preds = np.argmax(probs, axis=1)

    # union_arr = (np.where((gt == bert_pred) | (gt ==lstm_pred)))
    # intersection_arr = (np.where((gt == bert_pred) & (gt ==lstm_pred)))
    # sizeOfunion_int = len(union_arr[0])
    # sizeOfintersection_int = len(intersection_arr[0])
    # HDR = (sizeOfunion_int - sizeOfintersection_int) / sizeOfunion_int

    return preds

def lstm_predict(model, test_dataloader):
    """
    Perform a forward pass on the trained Electra model to predict probabilities on the test set.
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
            logits = model(b_ids_tsr)
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()
    preds = np.argmax(probs, axis=1)

    return preds

def electra_predict(electra_model, test_dataloader):
    """
    Perform a forward pass on the trained Electra model to predict probabilities on the test set.
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
    electra_model.eval()
    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_ids_tsr, b_masks_tsr = tuple(tsr.to(device) for tsr in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = electra_model(b_ids_tsr, b_masks_tsr)
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()
    preds = np.argmax(probs, axis=1)

    return preds


def make_submission(lstm_model, electra_model, opt, test_dataloader):
    """
    Save a submission.csv
    
    :param model: trained model
    :param opt: hyper parameters
    :param test_dataloader: dataloader which is obtained by data_load method
    """
    submission_templete_df = pd.read_csv(opt.data_path +'/sample_sub.csv') # a sample submission file in the correct format
    
    # Early stop model
    if opt.save == 1:

        lstm_model_save_path = str(opt.save_model_path) + "/" + signature + "_lstm_" + '.model'
        lstm_model.load_state_dict(torch.load(lstm_model_save_path))
        electra_model_save_path = str(opt.save_model_path) + "/" + signature + "_electra_" + '.model'
        electra_model.load_state_dict(torch.load(electra_model_save_path))
    preds = predict(lstm_model, electra_model, test_dataloader)
    
    assert len(preds) == 4311
    submission_templete_df.Category = preds
    print(submission_templete_df)
    submission_save_path = str(opt.save_submission_path) + "/" + signature +'.csv'
    print(f"\t...Save complete at {submission_save_path}")
    submission_templete_df.to_csv(submission_save_path, index=False)

def check_hdr(gt, bert_pred, lstm_pred):
    union_arr = (np.where((gt == bert_pred) | (gt ==lstm_pred)))
    intersection_arr = (np.where((gt == bert_pred) & (gt ==lstm_pred)))
    sizeOfunion_int = len(union_arr[0])
    sizeOfintersection_int = len(intersection_arr[0])
    HDR = (sizeOfunion_int - sizeOfintersection_int) / sizeOfunion_int
    return HDR # 높다는건 교집합이 작다 즉 서로 알려줄게 많다.

if __name__ == "__main__":
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
    parser.add_argument('--model', type=str, default='ELECTRA', help='BERT, ELECTRA, BILSTM')
    parser.add_argument('--sent_embedding', type=int, default=0, help='0: CLS, 1: 4-layer concat')
    parser.add_argument('--hidden_dim', type=int, default=768, help='for wide models')
    parser.add_argument('--num_layer', type=int, default=2, help='for deep models')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')
    parser.add_argument('--max_len', type=int, default=64)

    # training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gpu', type=int, default=0, help='0,1,2,3')
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--save', type=int, default=1, help='0: false, 1:true')

    parser.add_argument('--optimizer', type=int, default=1, help='1: SGD, 2: RMSProp, 3: Adam')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate, 5e-5, 3e-5 or 2e-5')
    parser.add_argument('--eps', type=float, default=1e-8, help='epsilon')
    parser.add_argument('--t_max', type=int, default=0)
    parser.add_argument('--eta_min', type=float, default=0)

    # dataset
    parser.add_argument('--reopt', type=int, default=0)
    parser.add_argument('--ratio', type=int, default=1)
    parser.add_argument('--data_path', type=str, default='/home/jangyj0426/workspace/DL_PROJ_2021/Dataset/')
    parser.add_argument('--save_model_path', type=str, default='/home/jangyj0426/workspace/DL_PROJ_2021/Saved_models')
    parser.add_argument('--save_submission_path', type=str, default='/home/jangyj0426/workspace/DL_PROJ_2021/Submissions')
    parser.add_argument('--author', type=str, default='yj')
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('--train_val_split', type=bool, default=False) # train_valid data split 필요 유무

    opt = parser.parse_args() # in .py env
    # opt, _ = parser.parse_known_args() # in .ipynb env

    #################################################################################################################
    # Training Device
    #################################################################################################################
    device = torch.device("cuda:" + str(opt.gpu)) if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(device) # change allocation of current GPU
    print(f'training device: {device, torch.cuda.get_device_name()}')
    signature = str(opt.author) + "_" + str(opt.model) + "_" + str(opt.hidden_dim) + "_" + str(opt.num_layer) + "_" + str(opt.dropout) + "_" + str(opt.batch_size) + "_" + str(opt.max_epoch) + "_" + str(opt.lr) + "_" + str(opt.eps) + "_lstm_electra" 
    print(f'signature: {signature}')


    tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
    # MAX_LEN = opt.max_len

    train_dataloader, valid_dataloader, test_dataloader = data_load(opt, opt.max_len)

    loss_fn = nn.CrossEntropyLoss() # Specify loss functions
    lstm_classifier, lstm_optimizer, electra_classifier, electra_optimizer, scheduler = initialize_model(opt)
    
    train(electra_classifier, train_dataloader, valid_dataloader, epochs=opt.max_epoch, evaluation=True, lstm=False) # electra
    train(lstm_classifier, train_dataloader, valid_dataloader, epochs=20, evaluation=True, lstm=True) # lstm

    make_submission(lstm_classifier, electra_classifier, opt, test_dataloader)

    num_of_total_params = sum(p.numel() for p in electra_classifier.parameters())


