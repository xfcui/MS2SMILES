import torch
import torch.nn as nn
import csv
import numpy as np
import os
import grammar as gm
import time
import molmass
import multiprocessing as mp
from itertools import product
import token_process as tp
import model
import argparse
import os
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  

def get_model(path):
        
    my_model = model.Encode_Decode()
    state_dict = torch.load(path)
    my_model.load_state_dict(state_dict)
    return my_model

def get_data(fp_path, mol_path):
    fp = []
    with open(fp_path) as f:
        reader = csv.reader(f)
        for row in reader:
            row = [round(float(x)) for x in row]
            fp.append(row)

    ELEMENTS = ['C', 'H', 'O', 'N', 'P', 'S', 'F', 'I', 'Cl', 'Br']
    with open(mol_path) as f:
        reader = csv.reader(f)
        count_all = []
        element_all = []
        for row in reader:
            print(row[0])
            mf1 = molmass.Formula(row[0]).composition()
            element_ = []
            count_ = []
            for element, count in mf1.items():
                count = str(count).split('count=')[1].split(',',1)[0]
                element_.append(element)
                count_.append(count)
            count_all.append(count_)
            element_all.append(element_)
            element_=[]
            count_ = []
        
        fm = np.zeros((len(count_all), 10), dtype = 'int64')
        for i in range(len(count_all)):
            for j in range(len(element_all[i])):
                pos = ELEMENTS.index(element_all[i][j])
                fm[i][pos] = count_all[i][j]
    
    fp = torch.tensor(fp)
    mol = torch.tensor(fm)
    
    element_all = []
    for i in range(len(fm)):
        ELEMENTS = ['C', 'H', 'O', 'N', 'P', 'S', 'F', 'I', 'Cl', 'Br']
        element = []
        for j in range(len(fm[i])):
            if fm[i][j] !=0 :
                element.append(ELEMENTS[j])
        element_all.append(element)

    return fp, mol, element_all

def all_beam_search(model_path, fp, mol, element_all, beam_width, p):
    max_length = 127
    start_idx = '$0'
    start_idx1 = '$'

    model = get_model(model_path)
    model.eval()
    new_list = tp.get_new_list(element_all, len(fp))
    remain_elem = mol
    sequences = [([start_idx], 1.0, remain_elem[p], 0.0, 0.0, start_idx1, 0, 0)]

    H = mol[p][1]
    not_elem = new_list[p]
    for i in range(max_length):
        candidates = []
        for seq in sequences:
            y = tp.token_encode_one(seq[0])
            one_hot_y = tp.tokens_onehot(torch.tensor(y))
            dim = len(one_hot_y.size())
            if dim == 2:
                one_hot_y = torch.unsqueeze(one_hot_y, dim=0)
            y_pred = model(fp[p].unsqueeze(0), mol[p].unsqueeze(0), one_hot_y)
            y_pred = y_pred[:, i, :].cpu().detach().numpy()
            y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=-1, keepdims=True)
            for j in range(beam_width):
                idx = np.argmax(y_pred)
                temp_score = np.log(y_pred[0, idx])
                score, temp_elem, temp_elem_penalty, GRAMMAR_score, ring_score = gm.score(idx,H.item(),not_elem, seq, temp_score)
                xx = []
                if seq[0] == '$0':
                    xx.append(seq[0])
                else:
                    for j in seq[0]:
                        xx.append(j)
                xx.append(tp.VOC[idx])
                candidate = [xx, score, temp_elem, temp_elem_penalty, GRAMMAR_score, seq[5] + tp.VOC[idx][:-1], seq[6] + int(tp.VOC[idx][-1]), seq[-1] + ring_score]
              
                candidates.append(candidate)
                y_pred[0, idx] = 0.0
        sequences = sorted(candidates, key= lambda x: x[1], reverse=True)[:beam_width]
    return sequences 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp_path', type=str, default = None)
    parser.add_argument('--mf_path', type=str, default = None)
    parser.add_argument('--model_path', type=str, default = None)
    parser.add_argument('--beam_width', type=int, default = 128)
    args = parser.parse_args()
    fp, mol, element_all = get_data(args.fp_path, args.mf_path)
    split_smiles = []
    for i in range(len(fp)):
        all_candidates = all_beam_search(args.model_path, fp, mol, element_all, args.beam_width, i)
       