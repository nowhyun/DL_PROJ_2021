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

import torchvision
from torchvision import transforms, datasets

from Utils.dataset import *
from Utils.utils import *
from Models.BertClf import *
from Models.LstmClf import *

#################################################################################################################
# Train and Evaluate
#################################################################################################################
def train(model, train_dataloader, valid_dataloader=None, evaluation=False):
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
    for epoch_i in range(opt.max_epoch):
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
            if opt.model == "BILSTM":
                logits = model(b_ids_tsr)
            else:
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
        model_save_path = str(opt.save_model_path) + "/" + opt.signature +'.model'
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
    return model

    
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
            if opt.model == "BILSTM":
                logits = model(b_ids_tsr)
            else:
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


if __name__ == "__main__":
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
    parser.add_argument('--model', type=str, default='BILSTM', help='BERT, BILSTM, ELECTRA')
    parser.add_argument('--sent_embedding', type=int, default=0, help='0: CLS, 1: 4-layer concat')
    parser.add_argument('--hidden_dim', type=int, default=64, help='for wide models')
    parser.add_argument('--num_layer', type=int, default=2, help='for deep models')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout ratio')

    # training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gpu', type=int, default=0, help='0,1,2,3')
    parser.add_argument('--max_epoch', type=int, default=15)
    parser.add_argument('--save', type=int, default=1, help='0: false, 1:true')
    parser.add_argument('--optimizer', type=int, default=1, help='1: SGD, 2: RMSProp, 3: Adam')
    parser.add_argument('--lr', type=float, default=5e-5, help='learning rate, 5e-5, 3e-5 or 2e-5')
    parser.add_argument('--eps', type=float, default=1e-8, help='epsilon')
    parser.add_argument('--momentum', type=float, default=0.9, help='epsilon')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='epsilon')

    # dataset
    parser.add_argument('--data_path', type=str, default='./Dataset')
    parser.add_argument('--save_model_path', type=str, default='./Saved_models')
    parser.add_argument('--save_submission_path', type=str, default='./Submissions')
    parser.add_argument('--balanced', type=int, default=0, help='0: default, 1: easy to hard')
    parser.add_argument('--n_classes', type=int, default=5, help='for balanced')
    parser.add_argument('--n_samples', type=int, default=5, help='for balanced')
    parser.add_argument('--max_len', type=int, default=80, help='max length of the sentence')
    parser.add_argument('--replace', type=int, default=1, help='preprocessed option 0 or 1')
    parser.add_argument('--valid_ratio', type=float, default=1/6)
    parser.add_argument('--author', type=str, default='who')


    opt = parser.parse_args() # in .py env
    # opt, _ = parser.parse_known_args() # in .ipynb env

    #################################################################################################################
    # Training Device
    #################################################################################################################
    device = torch.device("cuda:" + str(opt.gpu)) if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(device) # change allocation of current GPU
    print(f'training device: {device, torch.cuda.get_device_name()}')
    signature = str(opt.author) + "_" + str(opt.model) + "_" + str(opt.sent_embedding) + "_" + str(opt.hidden_dim) + "_" + str(opt.batch_size) + "_" + str(opt.max_epoch) + "_" + str(opt.lr) + "_" + str(opt.eps) 
    opt.signature = signature
    print(f'signature: {signature}')
    
    # Load the DataLoaders
    train_dataloader, valid_dataloader, test_dataloader = data_load(opt)

    # Specify the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Initialize the model
    untrained_model, optimizer, scheduler = initialize_model(opt, len(train_dataloader), device)

    trained_model = train(untrained_model, train_dataloader, valid_dataloader=valid_dataloader, evaluation=True)
    
    # Train the model on the entire training data
    if opt.save == 1:
        model_save_path = str(opt.save_model_path) + "/" + signature +'.model'
        untrained_model.load_state_dict(torch.load(model_save_path))
    opt.lr = opt.lr * 0.1
    full_trained_model = train(untrained_model, valid_dataloader, evaluation=False)
    model_save_path = str(opt.save_model_path) + "/" + opt.signature +'_full.model'
    if opt.save == 1: 
        torch.save(full_trained_model.state_dict(), model_save_path)
        print('\tthe model is improved... save at', model_save_path)

    # full_dataloader = data_load(opt, flag="full")
    # full_trained_model = train(untrained_model, full_dataloader, evaluation=False)
    
    make_submission(untrained_model, opt, device, test_dataloader)
    make_submission(untrained_model, opt, device, test_dataloader, full=True)