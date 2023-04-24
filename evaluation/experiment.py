import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
from transformers.file_utils import cached_property
from typing import Tuple
from sklearn.model_selection import train_test_split
import gc
from tqdm import tqdm

import pandas as pd
from sklearn.utils import shuffle

from collections import Counter
from tqdm import tqdm
from typing import List, Dict, Union, NoReturn

# from contextlib import redirect_stdout
# import io
from IPython.display import clear_output
import os
import subprocess

import itertools
from evaluation.std_answers2table import answers




# ------ evaluate single model ------

class evaluate_model:
    def __init__(self, model, tokenizer, device):
    
        self.tokenizer = tokenizer
        self.device = device
        self.initial_model = model
        self.model = model.to(device)
        
    def cleanup(self):
        gc.collect()
        torch.cuda.empty_cache()
        
    def tokenize(self, data_train, target_train, data_test, target_test):
        self.TrainDataset = PairsDataset(self.tokenizer(data_train), self.tokenizer(target_train))
        self.TestDataset = PairsDataset(self.tokenizer(data_test), self.tokenizer(target_test))
        
    def model_init(self, TrainArgs, data_collator):
        args = TrainingArguments(save_total_limit = 1, **TrainArgs)
        
        self.trainer = Trainer(
            model = self.model,
            args = args,
            train_dataset = self.TrainDataset,
            eval_dataset = self.TestDataset,
            tokenizer = self.tokenizer,
            data_collator = data_collator
        )
        
    def model_train(self):
        self.trainer.train()
        
    def predict(self, data_test, prediction_name: str, GenArgs: dict, trainer_exists, SelectStrategy=None, PredForm=None):
        prediciton = []
        for text in tqdm(data_test):

            input_ids = self.tokenizer.encode(text, return_tensors='pt')
            outputs = self.model.generate(input_ids.to(self.device), **GenArgs)
            
            decoded_list = []
            for output in outputs:
                decoded = self.tokenizer.decode(output, skip_special_tokens=True).split(", ")
                decoded_list.extend(decoded)
                
            if SelectStrategy is None:
                predicted_answer = [i[0] for i in Counter(decoded_list).most_common()]
            else:
                predicted_answer = SelectStrategy(decoded_list)
                
                
            if PredForm is None:
                predicted_answer = '\t'.join(predicted_answer)
            else:
                predicted_answer = PredForm(predicted_answer)
            
            prediciton.append(predicted_answer)
    
        self.prediciton = pd.DataFrame(prediciton)
        self.pred_path = os.path.join(os.getcwd(), prediction_name)
        self.prediciton.to_csv(self.pred_path, header=None, index=None)
        
        self.cleanup()
        
        
    def SemEval2018_metrics(self, path_data, path_prediciton, param_inx, general_param):
        # redirect stdout
        _std_out = subprocess.check_output(['python', os.path.join(os.getcwd(), 'evaluation/custom_scorer.py'), 
                                            path_data, path_prediciton])
        _std_out = _std_out.decode('UTF-8')
        _std_out = _std_out  + param_inx

        # get answers
        columns_name = []
        values = []
        for ind, metrics in enumerate(_std_out.split('\n')):
            if ind == 6:
                _name = general_param
                number = metrics
            else:
                _name, number = metrics.split(' ')
                number = round(float(number), 5)
                _name = _name[:-1]

            columns_name.append(_name)
            values.append([number])



        table = pd.DataFrame(values).T
        table.columns = columns_name
        table.set_index(general_param, inplace=True)
        return table
    
    def custom_metrics(self, path_data, path_prediciton, answer, param, general_param):
        _std_out = subprocess.check_output(['python', os.path.join(os.getcwd(), 'evaluation/custom_scorer.py'), 
                                            path_data, path_prediciton])
        _std_out = _std_out.decode('UTF-8')
        table = answer(_std_out, param, general_param)
        return table
        
        
# ------ Pairs Dataset ------

class PairsDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        assert idx <= len(self.x['input_ids']), (idx, len(self.x['input_ids']))
        item = {key: val[idx] for key, val in self.x.items()}
        item['decoder_attention_mask'] = self.y['attention_mask'][idx]
        item['labels'] = self.y['input_ids'][idx]
        return item
    
    @property
    def n(self):
        return len(self.x['input_ids'])

    def __len__(self):
        return self.n
    
# ------ Experiment ------

class Experiment:
    def __init__(self, 
                 check_param, freezed_param,
                 output_dir, model, tokenizer, device, 
                 data_train, target_train, data_test, target_test,
                 strategy,
                 path_to_test, 
                 data_collator,
                 collate_answer=answers,
                 experiment_output_dir='experiment',
                 pred_file_name='experiment.txt'
                ):
        self.check_param = check_param
        self.freezed_param = freezed_param
        
        self.output_dir = output_dir
        self.model = model
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.device = device
        
        self.data_train = data_train
        self.target_train = target_train
        self.data_test = data_test
        self.target_test = target_test
        
        self.collate_answer = collate_answer
        self.strategy = strategy
        
        self.path_to_test=path_to_test
        self.pred_file_name = pred_file_name
        self.experiment_output_dir = experiment_output_dir
        
        self.results = []
        # self.current_model = evaluate_model(self.model, self.tokenizer, self.device)
        
        if not os.path.exists(self.experiment_output_dir):
            os.mkdir(self.experiment_output_dir)
        
    def train_model(self, TrainArgs, number_of_experiment):
        out_dir = os.path.join(self.experiment_output_dir, str(number_of_experiment))
        TrainArgs['output_dir'] = os.path.join(self.output_dir, out_dir)
        
        self.current_model = evaluate_model(self.model, self.tokenizer, self.device)
        self.current_model.tokenize(self.data_train, self.target_train, 
                                    self.data_test, self.target_test)
        
        self.current_model.model_init(TrainArgs, self.data_collator)
        self.current_model.model_train()
        
    def predict_model(self, 
                      prediction_output_dir,
                      trainer_exists,
                      GenArgs,
                      param_inx, 
                      general_param='Meta-Params'):
        
        self.current_model.predict(self.data_test,
                                   prediction_output_dir,
                                   GenArgs,
                                   trainer_exists=trainer_exists,
                                  )
        
        metrics = self.current_model.custom_metrics(path_data=self.path_to_test, 
                                                    path_prediciton=prediction_output_dir,
                                                    answer=self.collate_answer, 
                                                    param=param_inx, 
                                                    general_param=general_param)
        
        self.results.append(metrics)
        
    def run_experiment(self):
        # Get checked params sets
        self.current_model = evaluate_model(self.model, self.tokenizer, self.device)
        GEN_ARGS = self.unzip_args(self.check_param, 'GenArgs')
        
        
        if sum(self.strategy) == 2:
            TRAIN_ARGS = self.unzip_args(self.check_param, 'TrainArgs')
            
            param_sets = []
            for val in list(itertools.product(TRAIN_ARGS, GEN_ARGS)):
                param_sets.append({'TrainArgs':val[0], 'GenArgs':val[1]})
        else:
            param_sets = [{'GenArgs': GEN_ARGS[i]} for i in range(len(GEN_ARGS))]
            
            
            
        if sum(self.strategy) == 1:
            self.train_model(TrainArgs=self.freezed_param['TrainArgs'], number_of_experiment=-1)
                      
        for num_exp, param_set in enumerate(param_sets):
            curr_out_dir = os.path.join(self.experiment_output_dir, str(num_exp))
            
            if not os.path.exists(curr_out_dir):
                os.mkdir(curr_out_dir)
            
            
            param = self.merge_freeze_param(self.freezed_param, param_set) # merged with freezed
            
            if sum(self.strategy) == 2:
                self.train_model(TrainArgs=param['TrainArgs'], number_of_experiment=num_exp)
             
            trainer_exists = sum(self.strategy) > 0
            self.predict_model(prediction_output_dir=os.path.join(curr_out_dir, self.pred_file_name),
                               trainer_exists=trainer_exists,
                               param_inx=str(param_set),
                               GenArgs=param['GenArgs'])
            
            if sum(self.strategy) == 2:
                del self.current_model
                self.current_model = evaluate_model(self.model, self.tokenizer, self.device)
            
        if sum(self.strategy) == 1:
                del self.current_model
            
        return pd.concat(self.results)
                      
    def unzip_args(self, param_dict, ARGS_NAME):
        ARGS=[]
        number = len(param_dict[ARGS_NAME].keys())

        for key, val in param_dict[ARGS_NAME].items():
            for v in val:
                ARGS.append({key:v})

        ARGS_2 = []
        for comb in list(itertools.combinations(ARGS, number)):

            check_list =[list(param.keys())[0] for param in comb]
            if len(check_list) == len(set(check_list)):
                output = {list(p_dict.keys())[0]:list(p_dict.values())[0] for p_dict in comb}

                ARGS_2.append(output)

        return ARGS_2
                      
    def merge_freeze_param(self, freezed_param, check_param):
        param_set_final = {}
        for args in freezed_param.keys():
            if args in check_param.keys():
                param_set_final[args] = dict(list(check_param[args].items()) + list(freezed_param[args].items()))

        param_set_final['SelectStrategy'] = freezed_param['SelectStrategy']
        param_set_final['PredForm'] = freezed_param['PredForm']

        return param_set_final
