import os
import sys
import random
import string
import ast
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

def are_templates_equivalent(t1,t2,wildcard_alphabet):
    sW = wildcard_alphabet
    assert(len(t1) == len(t2))

    # Check if wildcard positions match
    for i in range(len(t1)):
        if t1[i] in sW:
            if t2[i] not in sW:
                return False
        elif t2[i] in sW:
            return False
        elif not t1[i] == t2[i]:
            assert(t1[i] not in sW)
            assert(t2[i] not in sW)
            return False
        else:
            assert(t1[i] not in sW)
            assert(t2[i] not in sW)

    # Check if within word relative token equality matches
    for i in range(len(t1)):
        for j in range(len(t2)):
            if t1[i] == t1[j]:
                if not t2[i] == t2[j]:
                    return False
            if t2[i] == t2[j]:
                if not t1[i] == t1[j]:
                    return False
    return True

# Used to generate a data frame for a list of templates, probabilities, labels, and alphabet sizes
class GeneratingDatasetFromTemplateList(object):
    def __init__(self, template_list, template_probs, template_labels, wildcard_alphabet, 
                 label_type,
                 train_substitution_alphabet_size,
                 val_substitution_alphabet_size,
                 test_substitution_alphabet_size,
                 num_train_samples, 
                 num_val_samples,
                 num_test_samples,
                 one_hot_vecs=True, ):

        sWild = set(wildcard_alphabet)

        ####### FIRST, PROCESS THE TEMPLATE LIST TO MAKE SURE SANITY CHECKS ARE PASSED

        ## NO TWO TEMPLATES SHOULD BE EQUIVALENT
        for i, t1 in enumerate(template_list):
            for j in range(i+1,len(template_list)):
                if are_templates_equivalent(t1,template_list[j],wildcard_alphabet=sWild):
                    print('Templates ' + t1 + ' and ' + template_list[j] + ' equivalent')
                    assert(False)

        ## RUN MORE SANITY CHECKS ON LIST LENGTHS AND LABEL TYPE
        assert(len(template_list) == len(template_probs))
        assert(len(template_list) == len(template_labels))
        assert(len(template_list) > 0)
        assert(label_type in ['regression', 'multiclass'])
        if label_type == 'regression':
            for l in template_labels:
                l = float(l) # Check that can cast to float
        elif label_type == 'multiclass':
            for l in template_labels:
                assert(len(l) == 1) # Check that label is single token
        # All templates should be same length
        template_len = len(template_list[0])
        for templ in template_list:
            assert(len(templ) == template_len)


        ####### SECOND, CONSTRUCT THE REGULAR, WILDCARD, AND SUBSTITUTION ALPHABETS
                
        ## COMPUTE WHICH WILDCARD TOKENS AND REGULAR TOKENS ARE USED BY TEMPLATES
        wildcards_in_templates = set()
        regular_tokens_in_templates = set()
        for t in template_list:
            wildcards_in_templates.update(set(t).intersection(sWild))
            regular_tokens_in_templates.update(set(t).difference(sWild))
            
        if label_type == 'multiclass':
            for l in template_labels:
                if l not in wildcards_in_templates:
                    regular_tokens_in_templates.add(l)

        wildcards_in_templates = list(wildcards_in_templates)
        regular_tokens_in_templates = list(regular_tokens_in_templates)

        print('Wildcards', wildcards_in_templates)
        print('Regular', regular_tokens_in_templates)
        
        
        ## COMPUTE TOTAL SIZE OF ALPHABET, AND ASSIGN AN INDEX TO EACH REGULAR/WILDCARD TOKEN
        size_W = len(wildcards_in_templates)
        size_R = len(regular_tokens_in_templates)
        tot_tokens = size_W + size_R + train_substitution_alphabet_size + val_substitution_alphabet_size + test_substitution_alphabet_size
        self.vocab_size = tot_tokens
        i = 0
        self.token_to_index = {}
        for w in wildcards_in_templates:
            self.token_to_index[w] = i
            i +=1
        for r in regular_tokens_in_templates:
            self.token_to_index[r] = i
            i += 1
        
        train_sub_range = (size_W + size_R, size_W + size_R + train_substitution_alphabet_size)
        val_sub_range = (size_W + size_R + train_substitution_alphabet_size, size_W + size_R + train_substitution_alphabet_size + val_substitution_alphabet_size)
        test_sub_range = (size_W + size_R + train_substitution_alphabet_size + val_substitution_alphabet_size, size_W + size_R + train_substitution_alphabet_size + val_substitution_alphabet_size + test_substitution_alphabet_size)
        

        ####### THIRD, GENERATE THE SAMPLES BY RANDOMLY SUBSTITUTING FROM ALPHABET
        ####### GENERATE THESE SAMPLES THROUGH SUBSTITUTION WITH REPLACEMENT (CHANGE TO W/OUT REPLACEMENT LATER)
        
        def _gen_samples(num_samples, sub_range):
            contexts = np.zeros((num_samples, template_len), dtype=np.int32)
            if label_type == 'regression':
                labels = np.zeros(num_samples)
            elif label_type == 'multiclass':
                labels = np.zeros(num_samples, dtype = np.int64)

            template_selections = np.random.multinomial(num_samples, template_probs)
            curr_sample = 0
            for t_idx, templ in enumerate(template_list):
                currW = sWild.intersection(set(templ))
                for s_idx in range(template_selections[t_idx]):
                    
                    # Select a wildcard-to-token map without replacement
                    curr_out = random.sample(range(sub_range[0], sub_range[1]), len(currW))
                    curr_out_idx = 0
                    curr_map = {}
                    for w in currW:
                        curr_map[w] = curr_out[curr_out_idx]
                        curr_out_idx += 1
                        
                    for v_idx in range(len(templ)):
                        if templ[v_idx] in currW:
                            contexts[curr_sample,v_idx] = curr_map[templ[v_idx]]
                        else:
                            contexts[curr_sample,v_idx] = self.token_to_index[templ[v_idx]]
                    if label_type == 'regression':
                        labels[curr_sample] = template_labels[t_idx]
                    elif label_type == 'multiclass':
                        cl = template_labels[t_idx]
                        if cl in currW:
                            labels[curr_sample] = curr_map[cl]
                        else:
                            labels[curr_sample] = self.token_to_index[cl]
                    else:
                        assert(False)
                    curr_sample += 1
            return contexts, labels
            
        
        self.train_contexts, self.train_labels = _gen_samples(num_train_samples, train_sub_range)
        self.val_contexts, self.val_labels = _gen_samples(num_val_samples, val_sub_range)
        self.test_contexts, self.test_labels = _gen_samples(num_test_samples, test_sub_range)

    def generate_setup(self):
        return TemplateDataset(self.train_contexts, self.train_labels), TemplateDataset(self.val_contexts, self.val_labels), TemplateDataset(self.test_contexts, self.test_labels)

class TemplateDataset(torch.utils.data.Dataset):
    def __init__(self, contexts, labels):
        self.contexts = contexts
        self.labels = labels
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.contexts[idx], self.labels[idx]
    
    
def make_dataloaders(config):
    templates = config[('data','templates')]
    wildcards = config[('data', 'wildcards')]
    template_probs = config[('data', 'template_probs')]
    template_labels = config[('data', 'template_labels')]
    label_type = config[('data', 'label_type')]
    
    batch_size = config[('training', 'batch_size')]
    num_workers = config[('training', 'num_workers')]

    gen_dataset = GeneratingDatasetFromTemplateList(template_list=templates,template_probs=template_probs,
                                                    template_labels=template_labels,wildcard_alphabet=wildcards,
                                                    label_type=label_type,
                                                    train_substitution_alphabet_size=config[('data', 'train_substitution_alphabet_size')],
                                                    val_substitution_alphabet_size=config[('data', 'val_substitution_alphabet_size')],
                                                    test_substitution_alphabet_size=config[('data', 'test_substitution_alphabet_size')],
                                                    num_train_samples=config[('data', 'num_train_samples')],
                                                    num_val_samples=config[('data', 'num_val_samples')],
                                                    num_test_samples=config[('data', 'num_test_samples')])

    train_dataset, val_dataset, test_dataset = gen_dataset.generate_setup()
    vocab_size = gen_dataset.vocab_size
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    return {'train' : train_loader, 'test' : test_loader, 'val' : val_loader, 'vocab_size' : vocab_size}