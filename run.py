import argparse
import json
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

import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold

import transformers
from transformers import BertTokenizer, BertModel, ElectraTokenizer, ElectraModel, AdamW, get_linear_schedule_with_warmup

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
from Models.ElectraClf import *
from Models.ConvClf import *


#################################################################################################################
# Train and Evaluate
#################################################################################################################
def train_fn(model,
             optimizer,
             scheduler,
             loss_fn,
             loss_num, # for loss func's input
             train_dataloader,
             valid_dataloader=None,
             evaluation=False):
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
    es_eval_dict = {
        "epoch": 0,
        "train_loss": 0,
        "valid_loss": 0,
        "valid_acc": 0
    }  # early stop
    for epoch_i in range(opt.max_epoch):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(
            f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}"
        )
        print("-" * 70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            # Load batch to GPU
            b_ids_tsr, b_masks_tsr, b_labels_tsr = tuple(
                tsrs.to(device) for tsrs in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            if opt.model in ["BILSTM", "CNN"]:
                logits = model(b_ids_tsr)
            else:
                logits = model(b_ids_tsr, b_masks_tsr)

            # Compute loss and accumulate the loss values
            if loss_num == 1 :
                y = torch.randint(0,5,(logits.shape[0],))
                onehot = torch.zeros((logits.shape[0],5))
                onehot[range(len(b_labels_tsr)),b_labels_tsr] = 1
                y = onehot.cuda()
                #y = y.cpu().data.numpy()
                #print(logits)
                #print(y)
                #print('loss')
                loss = loss_fn(logits, y)
            
            else:
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
            if (step % 20 == 0
                    and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(
                    f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}"
                )

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-" * 70)
        # =======================================
        #               Evaluation
        # =======================================
        model_save_path = str(
            opt.save_model_path) + "/" + opt.signature + '.model'
        if evaluation == True:
            previous_valid_acc = es_eval_dict["valid_acc"]  # early stop
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            valid_loss, valid_acc = evaluate_fn(model, loss_fn, loss_num,
                                                valid_dataloader)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            print(
                f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {valid_loss:^10.6f} | {valid_acc:^9.2f} | {time_elapsed:^9.2f}"
            )
            print("-" * 70)
            if previous_valid_acc < valid_acc:
                es_eval_dict["epoch"] = epoch_i
                es_eval_dict["train_loss"] = avg_train_loss
                es_eval_dict["valid_loss"] = valid_loss
                es_eval_dict["valid_acc"] = valid_acc
                if opt.save == 1:
                    torch.save(model.state_dict(), model_save_path)
                    print('\tthe model is improved... save at',
                          model_save_path)
        print("\n")
    print("Final results table")
    print("-" * 70)
    print(
        f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}"
    )
    final_epoch, final_train_loss, final_valid_loss, final_valid_acc = es_eval_dict[
        "epoch"], es_eval_dict["train_loss"], es_eval_dict[
            "valid_loss"], es_eval_dict["valid_acc"]
    print(
        f"{final_epoch + 1:^7} | {'-':^7} | {final_train_loss:^12.6f} | {final_valid_loss:^10.6f} | {final_valid_acc:^9.2f} | {0:^9.2f}"
    )
    print("-" * 70)
    print("Training complete!")
    return model, final_train_loss, final_valid_loss, final_valid_acc


def evaluate_fn(model, loss_fn, loss_num, valid_dataloader):
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
        b_ids_tsr, b_masks_tsr, b_labels_tsr = tuple(
            t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            if opt.model in ["BILSTM", "CNN"]:
                logits = model(b_ids_tsr)
            else:
                logits = model(b_ids_tsr, b_masks_tsr)

        # Compute loss
        if loss_num == 1:
            y = torch.randint(0,5,(logits.shape[0],))
            onehot = torch.zeros((logits.shape[0],5))
            onehot[range(len(b_labels_tsr)),b_labels_tsr] = 1
            y = onehot.cuda()
            #y = y.cpu().data.numpy()
            loss = loss_fn(logits, y)
        
        else:
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


def cross_validation(full_dataset=None, n_splits=5):
    """Define a cross validation function
    """
    train_loss_list, valid_loss_list, valid_acc_list = [], [], []
    full_ids = full_dataset.ids_tsr.detach().cpu().numpy()
    full_labels = full_dataset.labels.detach().cpu().numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    for i, idx in enumerate(skf.split(full_ids, full_labels)):
        print(f"Start {i}-th cross validation...\n")
        train_indices, valid_indices = idx[0], idx[1]
        print(train_indices)
        print(valid_indices)

        train_subset = torch.utils.data.dataset.Subset(full_dataset,
                                                       train_indices)
        valid_subset = torch.utils.data.dataset.Subset(full_dataset,
                                                       valid_indices)

        print(
            f"len of train set: {len(train_subset)}, len of valid set: {len(valid_subset)}"
        )
        print()

        train_dataloader = DataLoader(
            train_subset,
            batch_size=opt.batch_size,
            shuffle=True,
        )
        valid_dataloader = DataLoader(
            valid_subset,
            batch_size=opt.batch_size,
            shuffle=True,
        )

        # Specify the loss function
        loss_fn = nn.CrossEntropyLoss()

        # Initialize the model
        untrained_model, optimizer, scheduler = initialize_model(
            opt, len(train_dataloader), device)

        _, train_loss, valid_loss, valid_acc = train_fn(untrained_model,
                                                        optimizer,
                                                        scheduler,
                                                        loss_fn,
                                                        loss_num,
                                                        train_dataloader,
                                                        valid_dataloader,
                                                        evaluation=True)

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_acc)

        print(f"...Complete {i}-th cross validation\n")
    train_loss_arr = np.array(train_loss_list)
    valid_loss_arr = np.array(valid_loss_list)
    valid_acc_arr = np.array(valid_acc_list)
    valid_avg_score = np.mean(valid_acc_arr)
    print("=" * 60)
    print(f"Average valid accuracy: {valid_avg_score}")
    print("=" * 60)
    return train_loss_arr, valid_loss_arr, valid_acc_arr, valid_avg_score


if __name__ == "__main__":
    #################################################################################################################
    # Library Version
    #################################################################################################################
    print(f"pandas version: {pd.__version__}")
    print(f"numpy version: {np.__version__}")
    print(f"seaborn version: {sns.__version__}")
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
    parser.add_argument('--model', type=str, default='CNN', help='BERT, BILSTM, ELECTRA, CNN')
    parser.add_argument('--sent_embedding', type=int, default=0, help='0: CLS, 1: 4-layer concat')
    parser.add_argument('--hidden_dim', type=int, default=768, help='BERT or ELECTRA: hidden dimension of classifier, BILSTM: hidden dimension of lstm')
    parser.add_argument('--num_layer', type=int, default=2, help='BILSTM: number of layers of lstm')
    parser.add_argument("--embedding_dim", type=int, default=256, help='embedding dimension of CNN')
    parser.add_argument("--kernel_sizes", nargs='+', default=[3, 4, 5], type=int, help='kernel sizes of CNN')
    parser.add_argument("--kernel_depth", default=500, type=int, help='kernel depth of CNN')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout ratio')

    # training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--gpu', type=int, default=0, help='0,1,2,3')
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--save', type=int, default=1, help='0: false, 1:true')
    parser.add_argument('--lr_pretrained', type=float, default=1e-05, help='learning rate, 5e-5, 3e-5 or 2e-5')
    parser.add_argument('--lr_clf', type=float, default=0.0001, help='learning rate, 5e-5, 3e-5 or 2e-5')
    parser.add_argument('--freeze_pretrained', type=int, default=0, help='0: false, 1:true')
    parser.add_argument('--eps', type=float, default=1e-8, help='epsilon for AdamW, 1e-8')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay for AdamW, 5e-4')

    # dataset
    parser.add_argument('--data_path', type=str, default='./Dataset')
    parser.add_argument('--save_model_path', type=str, default='./Saved_models')
    parser.add_argument('--save_submission_path', type=str, default='./Submissions')
    parser.add_argument('--max_len', type=int, default=50, help='max length of the sentence')
    parser.add_argument('--aug', type=int, default=0, help='0: false, 1: true(ru)')
    parser.add_argument('--split_ratio', type=int, default=1, help='k/10, k in [1,2,3]')
    parser.add_argument('--author', type=str, default='jh')


    opt = parser.parse_args() # in .py env
#     opt, _ = parser.parse_known_args() # in .ipynb env

    #################################################################################################################
    # Training Device
    #################################################################################################################
    device = torch.device("cuda:" + str(opt.gpu)) if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(device) # change allocation of current GPU
    print(f'training device: {device, torch.cuda.get_device_name()}')
    curr_time = time.localtime()
    signature = f"{opt.author}_{opt.model}_{curr_time.tm_mon}M_{curr_time.tm_mday}D_{curr_time.tm_hour}H_{curr_time.tm_min}M"
    opt.signature = signature
    print(f'signature: {signature}')
    with open('./Saved_models/' + signature + '_opt.txt', 'w') as f:
        json.dump(opt.__dict__, f, indent=2)

    #################################################################################################################
    # Main
    #################################################################################################################
    # Load the DataLoaders
    train_dataloader, valid_dataloader, test_dataloader = data_load(opt)

    #Specify the loss function
    #loss_fn = nn.CrossEntropyLoss() #loss_num = 0
    #loss_num = 1 #for input (label) : one hot encoding 
    loss_num = 0
    #loss_fn = FocalLoss() #loss_num = 0
    loss_fn = nn.MultiMarginLoss(margin= 0.5) #  better than crossentropy , loss_num = 0

    # Initialize the model
    untrained_model, optimizer, scheduler = initialize_model(opt, len(train_dataloader), device)

    trained_model, _, _, _ = train_fn(untrained_model, optimizer, scheduler, loss_fn, loss_num, train_dataloader, valid_dataloader=valid_dataloader, evaluation=True)
    
    #################################################################################################################
    # Full train
    #################################################################################################################
    full_dataloader = data_load(opt, flag="full")
    untrained_model, optimizer, scheduler = initialize_model(opt, len(train_dataloader), device)
    full_trained_model, _, _, _ = train_fn(untrained_model, optimizer, scheduler, loss_fn, loss_num, full_dataloader, evaluation=False)
    model_save_path = str(opt.save_model_path) + "/" + opt.signature +'_full.model'
    if opt.save == 1: 
        torch.save(full_trained_model.state_dict(), model_save_path)
        print('\tthe model is improved... save at', model_save_path)
        
    #################################################################################################################
    # Save the submission file
    #################################################################################################################
    make_submission(trained_model, opt, device, test_dataloader)
    make_submission(trained_model, opt, device, test_dataloader, full=True)
    
    # Print the number of parameters
    numOfparams(trained_model)