import torch
import torch.nn as nn
import torch.nn.functional as F
import re
INITIAL_CHAR='$0'
FINAL_CHAR="*0"
PAD_CHAR="&0"
SEQUENCE_LEN=128

VOC = ['O0', 'O1', 'O2',
    '=0',
    'C0', 'C1', 'C2', 'C3', 'C4',
    '10',
    'c0', 'c1', 
    '20',
    '-0',
    '30',
    '40',
    '(0',
    ')0',
    'n0', 'n1', 
    '50',
    '60',
    '[nH]1',
    'o0',
    'N0', 'N1', 'N2', 'N3',
    '[N+]0',
    '[O-]0',
    'L0', 'L1',
    '[NH+]1',
    'S0', 'S1', 'S2',
    'F0', 'F1', 
    's0',
    'R0', 'R1',
    '#0',
    'P0', 'P1',  'P2', 'P3',
    'I0', 'I1',
    '[N-]0',
    '70',
    'p0',
    '[n+]0',
    '[NH3+]3',
    '[C-]0',
    '[NH2+]2',
    '[H]0',
    '[nH+]1',
    '[CH+]1',
    '[OH2+]2',
    '[CH2+]2',
    '[CH2-]2',
    '[O+]0',
    '[n-]0',
    '[S-]0',
    '[NH-]1',
    '[C+]0',
    '[pH]1',
    '[CH-]1',
    '[P+]0',
    '[OH+]1',
    '[S+]0',
    '[c-]0',
    '[S]0',
    '[I-]0',
    '[o+]0',
    '[cH-]1',
    '[SH2+]2']

VOC.extend([INITIAL_CHAR, FINAL_CHAR, PAD_CHAR])

def get_new_list(element_all, num):
    new_list = []
    for i in range(num):
        ELEMENTS = ['C', 'H', 'O', 'N', 'P', 'S', 'F', 'I', 'Cl', 'Br']
        s = list(set(ELEMENTS) - set(element_all[i]))
        new_lst = [elem.replace('Cl', 'L').replace('Br', 'R') for elem in s]
        new_list.append(new_lst)
    return new_list

def tokens_from_smiles(smiles):
    tokens = [re.findall('[^[]|\[.*?\]', s.replace('Cl','L').replace('Br','R')) for s in smiles]
    return tokens

def tokens_pad_one(tokens, 
                length = SEQUENCE_LEN,
               initial_char = INITIAL_CHAR,
               final_char = FINAL_CHAR,
               pad_char = PAD_CHAR):
    tokens.insert(0, initial_char)
    tokens.append(final_char)
    tokens.extend([pad_char] * (length - len(tokens)))
    return tokens[:length]

def tokens_pad(tokens,
               length = SEQUENCE_LEN,
               initial_char = INITIAL_CHAR,
               final_char = FINAL_CHAR,
               pad_char = PAD_CHAR):
    return [tokens_pad_one(x, length, initial_char, final_char, pad_char) for x in tokens]

def token_encode_one(tokens):
    VOC_MAP = {s: i for i, s in enumerate(VOC)}
    return [VOC_MAP.get(c, 0) for c in tokens]

def tokens_encode(tokens):
    return torch.stack([torch.tensor(token_encode_one(x), dtype=torch.int32) for x in tokens], dim=0)
def tokens_to_torch_wrap(smiles, H_num):
    tokens_ = tokens_from_smiles(smiles)
    new_tokens = []
    for i in range(len(tokens_)):
        new_token = []
        length = min(127, len(tokens_[i]))
        for j in range(length):
            new_token.append(tokens_[i][j] + str(int(H_num[i][j].item())))
        new_tokens.append(new_token)
    tokens_ = tokens_pad(new_tokens)
    tokens_mat = tokens_encode(tokens_)
    return tokens_mat

def tokens_decode(tokens_mat, one_hot=True):
    string = []
    if one_hot:
        tokens_mat = tokens_mat.argmax(dim=2)
    
    for i in range(len(tokens_mat)):     
        VOC_MAP1 = {i: s for i, s in enumerate(VOC)}
        m = ''.join([str(VOC_MAP1[token]) for token in tokens_mat[i].numpy()])
        string.append(m)
        m = ''
    return string

def tokens_onehot(tokens_mat):
    
    return torch.nn.functional.one_hot(tokens_mat.to(torch.int64), num_classes=len(VOC))

def tokens_to_smiles(tokens):
    return [bytes.decode(bytes(x, encoding='utf-8')).replace('L','Cl').replace('R','Br') for x in tokens]

