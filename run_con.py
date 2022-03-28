# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
from copy import Error
import os
import sys
from torch.nn.modules.activation import Threshold  

from transformers.tokenization_utils_base import ENCODE_KWARGS_DOCSTRING
import bleu
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from model import Seq2Seq
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

language_dict = {}
all_lang_situation = {}
####################################################################
tau = 10
best_threshold = 0
####################################################################
class Example(object):
    """A single training/test example."""
    def __init__(self,
                 source1,
                 source2,
                 target,
                 lang1,
                 lang2,
                 ):
        self.source = []
        self.lang = []
        self.source.append(source1)
        self.source.append(source2)
        self.lang.append(lang1)
        self.lang.append(lang2)
        self.target = target


def get_catagory_id(catagory):
    if language_dict.get(catagory) is None:
        catagory_id = len(language_dict)
        language_dict[catagory] = catagory_id
    else:
        catagory_id = language_dict[catagory]
    return catagory_id

def read_examples(filename):
    """Read examples from filename."""
    examples=[]
    with open(filename,encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            js=json.loads(line)
            code1 = ' '.join(js['Code1'].replace('\n', ' ').strip().split())
            code2 = ' '.join(js['Code2'].replace('\n', ' ').strip().split())
            task = ' '.join(str(js['Task']).replace('-', ' ').split())
            lang1 = get_catagory_id(js['Category1'])
            lang2 = get_catagory_id(js['Category2'])     
            examples.append(
                Example(
                    source1=code1,
                    source2=code2, 
                    target=task,
                    lang1=lang1,
                    lang2=lang2,
                    ) 
            )
    return examples


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 source_ids1,
                 source_ids2,
                 target_ids1,
                 target_ids2,
                 source_mask1,
                 source_mask2,
                 target_mask1,
                 target_mask2,
                 lang1,
                 lang2,
    ):
        self.source_ids1 = source_ids1
        self.source_ids2 = source_ids2
        self.target_ids1 = target_ids1
        self.target_ids2 = target_ids2
        self.source_mask1 = source_mask1
        self.source_mask2 = source_mask2
        self.target_mask1 = target_mask1
        self.target_mask2 = target_mask2
        self.source_lang1 = lang1
        self.source_lang2 = lang2   
        


def convert_examples_to_features(examples, tokenizer, args, portion=1):
    features = []
    for example_index, example in enumerate(examples):
        #source
        source_ids1, source_mask1 = source_process(tokenizer, args, example, 0)
        source_ids2, source_mask2 = source_process(tokenizer, args, example, 1)
 
        #target
        target_ids1, target_mask1 = target_process(tokenizer, args, example)   
        target_ids2 = target_ids1
        target_mask2 = target_mask1
       
        features.append(
            InputFeatures(
                 source_ids1,
                 source_ids2,
                 target_ids1,
                 target_ids2,
                 source_mask1,
                 source_mask2,
                 target_mask1,
                 target_mask2,
                 example.lang[0],
                 example.lang[1],
            )
        )
    features = features[:int(len(features)*portion)]
    return features

def target_process(tokenizer, args, example):
    target_tokens = example.target[:args.max_target_length-2]
    target_ids = [int(target_tokens)]
    target_mask = [1] *len(target_ids)
    padding_length = args.max_target_length - len(target_ids)
    
    return target_ids,target_mask

def source_process(tokenizer, args, example, i):
    source_tokens = tokenizer.tokenize(example.source[i])[:args.max_source_length-2]
    source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens) 
    source_mask = [1] * (len(source_tokens))
    padding_length = args.max_source_length - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    source_mask+=[0]*padding_length
    return source_ids,source_mask


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters  
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )   
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str, 
                        help="Path to trained model: Should contain the .bin files" )    
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str, 
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str, 
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str, 
                        help="The test filename. Should contain the .jsonl files for this task.")  
    
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name") 
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 
    
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")   
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    # print arguments
    args = parser.parse_args()
    logger.info(args)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    # Set seed
    set_seed(args.seed)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
        
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,do_lower_case=args.do_lower_case)
    
    #budild model
    encoder = model_class.from_pretrained(args.model_name_or_path,config=config)    
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model=Seq2Seq(encoder=encoder,decoder=decoder,config=config,
                  beam_size=args.beam_size,max_length=args.max_target_length,
                  sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
    
    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))
        
    model.to(device)
    if args.local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    if args.do_train:
        # Prepare training data loader
        train_examples, train_data = prepare_dataset(args, args.train_filename, tokenizer, 1)
        
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size//args.gradient_accumulation_steps, drop_last=True)

        num_train_optimization_steps =  args.train_steps

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total*0.1),
                                                    num_training_steps=t_total)
    
        #Start training
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num epoch = %d", args.num_train_epochs)
        

        model.train()
        dev_dataset={}
        nb_tr_examples, nb_tr_steps,global_step,best_bleu,best_loss = 0,0,0,0,1e6
        best_recall, best_precision, best_f1 = 0,0,0
        for epoch in range(args.num_train_epochs):
            bar = tqdm(train_dataloader,total=len(train_dataloader))
            for idx, batch in enumerate(bar):
                batch = tuple(t.to(device) for t in batch)
                source_ids,source_mask,task1_ids,_,target_ids,target_mask,_,_ = batch
                sen_vec1, sen_vec2 = model(source_ids=source_ids,source_mask=source_mask,target_ids=target_ids,target_mask=target_mask)
                

                loss_temp = torch.zeros((len(sen_vec1),len(sen_vec1)*2-1),device=device, dtype=torch.float)
                for i in range(len(sen_vec1)):
                    loss_temp[i][0] = (nn.CosineSimilarity(dim=0)(sen_vec1[i],sen_vec2[i]) + 1) * 0.5 * tau
                    indice = 1
                    for j in range(len(sen_vec1)):
                        if i == j:
                            continue
                        temp = j
                        while torch.equal(task1_ids[i], task1_ids[temp]):
                            temp = (temp + 1) % (len(sen_vec1))
                        loss_temp[i][indice] = (nn.CosineSimilarity(dim=0)(sen_vec1[i],sen_vec2[temp]) + 1) * 0.5 * tau
                        indice += 1
                        loss_temp[i][indice] = (nn.CosineSimilarity(dim=0)(sen_vec1[i],sen_vec1[temp]) + 1) * 0.5 * tau
                        indice += 1
                con_loss = -torch.nn.LogSoftmax(dim=1)(loss_temp)
                con_loss = torch.sum(con_loss, dim=0)[0]
                con_loss = con_loss / len(sen_vec1)
                
                loss = con_loss
                if args.n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss = loss
                bar.set_description("epoch {} loss {} ".format(epoch, tr_loss))
                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
                    #Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1

                
            if args.do_eval:
                #Eval model with dev dataset
                tr_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0                     
                if 'dev_loss' in dev_dataset:
                    eval_examples,eval_data=dev_dataset['dev_loss']
                else:
                    eval_examples, eval_data = prepare_dataset(args, args.dev_filename, tokenizer)
                    dev_dataset['dev_loss']=eval_examples,eval_data
                eval_sampler = RandomSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=True)

                logger.info("\n***** Running evaluation *****")
                logger.info("  Num examples = %d", len(eval_examples))
                logger.info("  Batch size = %d", args.eval_batch_size)

                #Start Evaling model
                model.eval()
                eval_loss, tokens_num = 0,0
                
                cos_right = []
                cos_wrong = []
                for batch in eval_dataloader:
                    batch = tuple(t.to(device) for t in batch)

                    source_ids,source_mask,target,_,target_ids,target_mask,_,_ = batch           
                    with torch.no_grad():
                        sen_vec1, sen_vec2= model(source_ids=source_ids,source_mask=source_mask,target_ids=target_ids,target_mask=target_mask)
                         
                    cos = nn.CosineSimilarity(dim=1)(sen_vec1,sen_vec2)
                    cos_right += cos.tolist()

                    for i in range(len(sen_vec1)):
                        nag_count = 0
                        for j in range(len(sen_vec1)):
                            if i == j:
                                continue
                            if torch.equal(target[i],target[j]):
                                continue
                            cos_wrong += [nn.CosineSimilarity(dim=0)(sen_vec1[i],sen_vec1[j]).item()]
                            break
                            nag_count += 1
                            cos_wrong += [nn.CosineSimilarity(dim=0)(sen_vec1[i],sen_vec2[j]).item()]
                            nag_count += 1
                            if nag_count == 6:
                                break
                temp_best_f1 = 0
                temp_best_recall = 0
                temp_best_precision = 0
                temp_count = 0
                temp_error_count = 0
                temp_error_total = 0
                temp_total = 0
                temp_best_threshold = 0
                for i in tqdm(range(1, 100)):
                    count = 0
                    error_count = 0
                    threshold = i/100
                    for i in cos_right:
                        if i >= threshold:
                            count += 1
                    total = len(cos_right)
                    for i in cos_wrong:
                        if i < threshold:
                            error_count += 1
                    error_total = len(cos_wrong)
                    correct_recall = count/total
                    if error_total-error_count+count == 0:
                        continue
                    precision = count/(error_total-error_count+count) 
                    if precision+correct_recall == 0:
                        continue
                    F1 = 2*precision*correct_recall/(precision+correct_recall)
                    if F1 > temp_best_f1:
                        temp_best_f1 = F1
                        temp_best_recall = correct_recall
                        temp_best_precision = precision
                        temp_count = count
                        temp_error_count = error_count
                        temp_error_total = error_total
                        temp_total = total
                        temp_best_threshold = threshold

                #Pring loss of dev dataset    
                model.train()
                # eval_loss = eval_loss / tokens_num
                print("eval_loss", temp_count, temp_error_count, temp_total, temp_error_total)
                
                result = {'recall': temp_best_recall, 'precision': temp_best_precision, 'F1': temp_best_f1,
                          'global_step': global_step+1, 'threshold': temp_best_threshold,
                          'train_loss': round(tr_loss,5)}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  "+"*"*20)   
                writeresult = 'recall: ' + str(round(temp_best_recall, 3)) + ' precision:' + str(round(temp_best_precision, 3)) + \
                    ' F1:'+str(round(temp_best_f1, 3))+' tau:' + str(tau) + \
                    ' threshold:' + \
                    str(round(temp_best_threshold, 1)) + \
                    ' epoch:'+str(epoch)+'\n'
                f = open('result.txt','a+')
                f.write(writeresult)
                f.close()
                #save last checkpoint
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(last_output_dir, str(epoch)+" pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)                    
                if temp_best_f1 > best_f1:
                    best_f1 = temp_best_f1
                    best_precision = temp_best_precision
                    best_recall = temp_best_recall
                    best_threshold = temp_best_threshold
                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)  

                logger.info("  Best F1:%s", best_f1)
                logger.info("  "+"*"*20)
                logger.info("  Recall:%s", best_recall)
                logger.info("  "+"*"*20)
                logger.info("  Precision:%s", best_precision)
                logger.info("  "+"*"*20)
               
    if args.do_test:
        #Eval model with dev dataset
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0                     
        eval_examples, eval_data = prepare_dataset(args, args.test_filename, tokenizer)
        eval_sampler = RandomSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, drop_last=True)

        logger.info("\n***** Running test *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)

        #Start Evaling model
        model.eval()
        eval_loss, tokens_num = 0,0
        cos_right = []
        cos_wrong = []
        for batch in eval_dataloader:

            batch = tuple(t.to(device) for t in batch)
            source_ids,source_mask,target,_,target_ids,target_mask,_,_ = batch           
            with torch.no_grad():
                sen_vec1, sen_vec2= model(source_ids=source_ids,source_mask=source_mask,target_ids=target_ids,target_mask=target_mask)
            
            cos = nn.CosineSimilarity(dim=1)(sen_vec1,sen_vec2)
            cos_right += cos.tolist()
            for i in range(len(sen_vec1)):
                nag_count = 0
                for j in range(len(sen_vec1)):
                    if i == j:
                        continue
                    if torch.equal(target[i],target[j]):
                        continue
                    cos_wrong += [nn.CosineSimilarity(dim=0)(sen_vec1[i],sen_vec1[j]).item()]
                    break
                    nag_count += 1
                    cos_wrong += [nn.CosineSimilarity(dim=0)(sen_vec1[i],sen_vec2[j]).item()]
                    nag_count += 1
                    if nag_count == 6:
                        break
                    
        temp_best_f1 = 0
        temp_best_recall = 0
        temp_best_precision = 0
        temp_count = 0
        temp_error_count = 0
        temp_error_total = 0
        temp_total = 0

        count = 0
        error_count = 0
        if args.do_eval == False:
            best_threshold = 0.32
        logger.info("using eval_threshold: %s", best_threshold)
        for i in cos_right:
            if i >= best_threshold:
                count += 1
        total = len(cos_right)
        for i in cos_wrong:
            if i < best_threshold:
                error_count += 1
        error_total = len(cos_wrong)
        correct_recall = count/total
        precision = count/(error_total-error_count+count) 
        F1 = 2*precision*correct_recall/(precision+correct_recall)
        result = {'recall': correct_recall, 'precision': precision, 'F1': F1,
                    'threshold': best_threshold}
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))

        logger.info("using best threshold")
        for i in tqdm(range(1, 100)):
            count = 0
            error_count = 0
            threshold = i/100
            for i in cos_right:
                if i >= threshold:
                    count += 1
            total = len(cos_right)
            for i in cos_wrong:
                if i < threshold:
                    error_count += 1
            error_total = len(cos_wrong)
            correct_recall = count/total
            if error_total-error_count+count == 0:
                continue
            precision = count/(error_total-error_count+count) 
            F1 = 2*precision*correct_recall/(precision+correct_recall)
            if F1 > temp_best_f1:
                temp_best_f1 = F1
                temp_best_recall = correct_recall
                temp_best_precision = precision
                temp_count = count
                temp_error_count = error_count
                temp_error_total = error_total
                temp_total = total
                best_threshold = threshold

        result = {'recall': temp_best_recall, 'precision': temp_best_precision, 'F1': temp_best_f1,
                     'threshold': best_threshold}
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
        logger.info("  "+"*"*20)   
        writeresult = 'recall: ' + str(round(temp_best_recall, 3)) + ' precision:' + str(round(temp_best_precision, 3)) + \
            ' F1:'+str(round(temp_best_f1, 3))+' tau:' + str(tau) + \
            ' threshold:' + \
            str(round(best_threshold, 1)) 
        f = open('result.txt','a+')
        f.write(writeresult)
        f.close()

def gen_sen_vec(source_mask, encoder_output):
    # encoder_output = encoder_output.permute(1,0,2)
    # encoder output shape [16, 256, 776]
    output_mask = source_mask.unsqueeze(-1).expand(encoder_output.shape)
    encoder_output = encoder_output * output_mask

    # acquire the length of each sentence
    source_lengths = torch.sum(source_mask, dim=1)
    sentence_vector = torch.sum(encoder_output, dim=1)
    sentence_vector = sentence_vector / source_lengths.unsqueeze(-1)
    return sentence_vector

def prepare_dataset(args, filename, tokenizer, portion=1):
    train_examples = read_examples(filename)
    train_features = convert_examples_to_features(train_examples, tokenizer,args, portion)
    all_source_ids1 = torch.tensor([f.source_ids1 for f in train_features], dtype=torch.long)
    all_source_mask1 = torch.tensor([f.source_mask1 for f in train_features], dtype=torch.long)
    all_target_ids1 = torch.tensor([f.target_ids1 for f in train_features], dtype=torch.long)
    all_target_mask1 = torch.tensor([f.target_mask1 for f in train_features], dtype=torch.long)   
    all_source_ids2 = torch.tensor([f.source_ids2 for f in train_features], dtype=torch.long)
    all_source_mask2 = torch.tensor([f.source_mask2 for f in train_features], dtype=torch.long)
    all_target_ids2 = torch.tensor([f.target_ids2 for f in train_features], dtype=torch.long)
    all_target_mask2 = torch.tensor([f.target_mask2 for f in train_features], dtype=torch.long) 
    train_data = TensorDataset(all_source_ids1, all_source_mask1, all_target_ids1, all_target_mask1, all_source_ids2, all_source_mask2, all_target_ids2, all_target_mask2)
    return train_examples,train_data   

# reverse index
convert_lang_num = {} 


class lang_pair:
    def __init__(self, i, j):
        self.total = 0
        self.count = 0
        self.error_total = 0
        self.error_count = 0
        self.precision = 0
        self.recall = 0
        self.F1 = 0
        self.code1 = convert_lang_num[i]
        self.code2 = convert_lang_num[j]

    def add_total(self):
        self.total = self.total+1

    def add_count(self):
        self.count = self.count+1

    def add_error_total(self):
        self.error_total = self.error_total+1

    def add_error_count(self):
        self.error_count = self.error_count+1

    def cal(self):
        if self.total == 0:
            self.recall = "INF"
        else:
            self.recall = self.count/self.total
        if self.error_total-self.error_count+self.count == 0:
            self.precision = "INF"
        else:
            self.precision = self.count / \
                (self.error_total-self.error_count+self.count)
        if self.recall != "INF" and self.precision != "INF" and self.precision + self.recall > 0:
            self.F1 = 2*self.precision*self.recall/(self.precision+self.recall)
        else:
            self.F1 = "INF"

    def str(self):
        return ("{} and {}: Precision = {} Recall = {} F1 = {} total = {} error_total = {}\n".format(self.code1, self.code2, self.precision, self.recall, self.F1, self.total, self.error_total))


def get_pair(i, j):
    i = int(i)
    j = int(j)
    if i > j:
        return (j, i)
    else:
        return (i, j)


def initial_cal_lang():
    for key in language_dict:
        convert_lang_num[language_dict[key]] = key
    total_lang = len(language_dict)
    # construct the language pair
    for i in range(total_lang):  
        for j in range(i, total_lang):
            all_lang_situation[(i, j)] = lang_pair(i, j)


def cal_spec(filename):
    f = open(filename, mode='w')
    total = 0
    count = 0
    error_total = 0
    error_count = 0
    precision = 0
    recall = 0
    F1 = 0
    for i in convert_lang_num: 
        for j in convert_lang_num:
            if i == j:
                continue
            total = total + all_lang_situation[get_pair(i, j)].total
            count = count + all_lang_situation[get_pair(i, j)].count
            error_total = error_total + \
                all_lang_situation[get_pair(i, j)].error_total
            error_count = error_count + \
                all_lang_situation[get_pair(i, j)].error_count
        if total == 0:
            recall = "INF"
        else:
            recall = count/total
        if error_total-error_count+count == 0:
            precision = "INF"
        else:
            precision = count/(error_total-error_count+count)
        if recall != "INF" and precision != "INF" and precision + recall > 0:
            F1 = 2*precision*recall/(precision+recall)
        else:
            F1 = "INF"
        f.write("{}: Precision = {} Recall = {} F1 = {} total = {} error_total = {}\n".format(
            convert_lang_num[i], precision, recall, F1, total, error_total))

    f.close()
    exit()


def cal_print_all_lang(filename):
    f = open(filename, mode='w')
    for key in all_lang_situation:
        all_lang_situation[key].cal()
        f.write(all_lang_situation[key].str())
    f.close()
                
if __name__ == "__main__":
    main()

