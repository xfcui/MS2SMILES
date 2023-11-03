import numpy as np
import token_process as tp

inner = ['10', '20', '30', '40', '50', '60', '70']
def score(idx,H,not_elem, seq, temp_score):
    temp_elem = np.array(seq[2])
    remain_score = 0
    GRAMMAR_score = seq[4]
    if H - (seq[6] + int(tp.VOC[idx][-1])) < 0:
        remain_score = -np.inf
    if '=' in tp.VOC[idx]:
        if seq[0][-1] == '=0' or seq[0][-1] == '#0':
            remain_score = -np.inf
    
    for char, index in tp.char_to_index_map.items():
        if char in tp.VOC[idx].lower():
            temp_elem[index] -= 1
            if temp_elem[index] < 0:
                remain_score = -np.inf
    if '(' in tp.VOC[idx]:
        GRAMMAR_score = seq[4] -1.0
    if ')' in tp.VOC[idx]:
        if seq[4] == 0:
            GRAMMAR_score = -np.inf
        else:
            GRAMMAR_score = seq[4] + 1.0
    ring_score = 0
    if tp.VOC[idx] in inner:
        if seq[0].count(tp.VOC[idx])%2 == 0:
            ring_score = 1
        else:
            ring_score = -1
    temp_elem[1] -= int(tp.VOC[idx][-1])
    all_temp_elem = sum(temp_elem)
    temp_elem_penalty = all_temp_elem 
    if tp.VOC[idx] == '*0':  
        if temp_elem_penalty > 0:
            remain_score = -np.inf
        if GRAMMAR_score != 0.0:
            remain_score = -np.inf
        if seq[-1] != 0:
            remain_score = -np.inf
    if tp.VOC[idx] in not_elem:
        score = seq[1] + temp_score + -np.inf + remain_score 
    else:
        score = seq[1] + temp_score  + remain_score 
    return score, temp_elem, temp_elem_penalty, GRAMMAR_score, ring_score    