import time
import os
import random
import argparse
import configparser
from ast import literal_eval
from easydict import EasyDict as edict

import numpy as np
from tqdm import tqdm
import pandas as pd
import PIL

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from utils.base_trainer import abnormal_trainer

id_columns = ['environment_id', 'episode_id', 'step_id']
current_states = ['current_state', 'env_raw', 'env_rgb']
model_outs = ['score_output','subgoals','action',]
next_states = ['next_state', 'next_env_raw', 'next_env_rgb',]
next_info = ['reward', 'done', 'action_name', 'action_moved_player','action_moved_box',]
info = ['timestamp_start','timestamp_end']

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.reshape((batch_size, -1))
    
class CNN(nn.Module):
    def __init__(self, cfgs:edict, device:str, in_channel:int = None):
        super(CNN, self).__init__(cfgs, device)
        self.cfgs = cfgs
        self.device = device
        self.criterion = getattr(nn, cfgs.criterion)()

        _i = 0
        if in_channel is None :
            in_channel = cfgs.input_channel
        self.model = nn.Sequential()
        self.batchNorm = cfgs.batchnorm
        for hidden, kz, sz, pz in zip(cfgs.hidden_channels,
                                      cfgs.kernel_size,
                                      cfgs.stride,
                                      cfgs.padding) :
            conv = [nn.Conv2d(in_channel, hidden,
                               kernel_size=kz, stride=sz, padding=pz, # same dimension
                               bias=not self.batchNorm),]
            if self.batchNorm :
                conv.append(
                    nn.BatchNorm2d(hidden)
                )
            conv.append(nn.LeakyReLU(0.1))
            
            in_channel = hidden
            self.model.add_module(f'conv_{_i}', nn.Sequential(*conv))
            _i += 1
        
        last_layer = []
        if cfgs.flatten :
            last_layer.append(Flatten())
            # last_layer.append(nn.Linear(in_channel, cfgs.output_channel))
            last_layer.append(nn.LazyLinear(cfgs.output_channel))
        else :
            last_layer.append(nn.Conv2d(hidden, cfgs.output_channel,
                    kernel_size=3, stride=1, padding=1, # same dimension
            ))
        last_layer.append(nn.LeakyReLU(0.1))
        last_layer.append(getattr(nn, cfgs.output_activation)())
        self.model.add_module(f'output_{_i}', nn.Sequential(*last_layer))

    def forward(self, x, y = None) :
        out = self.model(x)
        if y is None :
            y = x
        try:
            loss = self.criterion(out.squeeze(-1), y)
        except ValueError :
            loss = None
        return out, loss, None
    
def getRawData(root, f_name) :
    info = pd.read_csv(f'{root}/raw/{f_name}/info.csv', index_col=0)
    columns = info.columns
    
    for column_name in current_states + next_states :
        ns = np.load(f'{root}/raw/{f_name}/{column_name}.npy', allow_pickle=True)
        assert ns.shape[0] == len(info)
        if type(ns[0]) == PIL.Image.Image :
            ns =list(map(lambda n: np.moveaxis(np.array(n).astype(np.float32),2,0)/255, ns))
        info = pd.concat([info, pd.DataFrame({column_name:ns})], axis=1, ignore_index = True)
        info.columns = list(columns) + [column_name]
        columns = info.columns
    info['id'] = [f_name for _ in range(len(info))]
    return info

class experienceReplay(torch.utils.data.Dataset):
    def __init__(self, size, base_path, load = False):
        super(experienceReplay, self).__init__()
        data_columns = id_columns + current_states + model_outs + next_states + next_info + info
        self.data_columns = data_columns
        self.columns = data_columns
        self.experience = pd.DataFrame(columns=data_columns)
        if size :
            self.size = int(size)
        
        self.base_path = base_path
        self.id_columns = id_columns
        self.np_format_columns = current_states + next_states
        
        self.groupby = False
        self.shuffled = False

        if load :
            self.load(base_path)
        self.index = list(self.experience.index)

    def __len__(self):
        return len(self.experience)
    
    @property
    def max_length(self):
        return self.size
    
    def set_data(self, columns):
        self.columns = columns

    def load(self, root = None):
        dataset = []
        num_sep = root.count(os.path.sep)
        for path, subdirs, files in tqdm(os.walk(root)):
            if not 'raw' in path : continue
            for name in files:
                if not 'csv' in name : continue
                num_sep_this = path.count(os.path.sep)
                if num_sep + 2 >= num_sep_this :
                    d_path = os.path.join(path, name)
                    _root = d_path.split("/raw")[0]
                    _name = d_path.split("/")[-2]
                    dataset.append(getRawData(_root, _name))
        self.experience = pd.concat(dataset).reset_index(drop=True)
        self.index = list(self.experience.index)

    def shuffle(self):
        self.suffled = True
        random.shuffle(self.index)
    
    def preprocessing(self, n_frames = 3):
        if not self.groupby :
            self.grouping()
        
        new_experience = pd.DataFrame(columns=self.columns)
        index = np.asarray([])
        for idx in range(len(self.index)) :
            group_idx = self.experience.indices[self.index[idx]]
            datum = self.experience.obj.iloc[group_idx][self.columns]
            new_datum = pd.DataFrame(columns = self.columns)
            new_datum.reindex(index = datum.index)
            for c in self.columns :
                _datum = datum[c].values
                _datum = np.stack(_datum, 0)
                if len(_datum.shape) == 1 :
                    _datum = np.expand_dims(_datum, -1)
                stacked_datum = np.concatenate([_datum[i:-n_frames + 1 + i] if -n_frames + 1 + i < 0 else _datum[i:]
                                                for i in range(n_frames)], 1)
                if c in ['action_moved_box', 'label'] :
                    new_datum[c] = [bool(sum(s)) for s in stacked_datum]
                    continue
                if 'id' in c :
                    new_datum[c] = [s[0] for s in stacked_datum]
                    continue
                new_datum[c] = [s for s in stacked_datum]
            index = np.concatenate([index, datum.index.values[:-n_frames+1]])
            new_experience = pd.concat([new_experience, new_datum])
        new_experience.set_index(index.astype(int))
        # FIXME
        self.experience = new_experience[new_experience['action_moved_box']]
        self.groupby = False
    
    def __getitem__(self, idx):
        if self.groupby:
            group_idx = self.experience.indices[self.index[idx]]
            datum = self.experience.obj.iloc[group_idx]
        else :
            datum =  self.experience.iloc[idx]
        if self.columns is None :
            return np.vstack(datum.to_numpy())
        return datum[self.columns].values.tolist()

    def grouping(self, keys = ['expert_id', 'environment_id', 'episode_id']) :
        self.groupby = True
        self.experience = self.experience.groupby(keys)
        self.group_keys = keys
        self.index = list(self.experience.indices.keys())

    def ungroup(self):
        self.groupby = False
        self.experience = self.experience.obj
        self.index = list(self.experience.index)


def init_seed(seed):
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
 
class Train(object):
    def __init__(self, args, cfgs, logger=None, torch_logger=None, **kwargs) :
        self.data_parallel = False
        if args.device == "cuda" and torch.cuda.is_available() :
            device = f"{args.device}:{cfgs.TRAIN.device_ids[0]}"
            self.data_parallel = len(cfgs.TRAIN.device_ids) > 1
        else :
            device = "cpu"
        self.device = device
        
        self.base_path = args.base_path
        self.cfgs = cfgs.TRAIN
        self.logger = logger
        self.torch_logger = torch_logger

    def build_model(self, weight_path = None):
        task_cfgs = get_cfgs(cfgs.MODELS.task_details)

        net = CNN(task_cfgs.NETWORK, self.device),

        self.recognizer = abnormal_trainer(cfgs, net,
                                  self.device,
                                  getattr(optim, task_cfgs.OPTIMIZER.name),
                                  task_cfgs.OPTIMIZER)
        
        # initialize network
        dummy_input = (torch.zeros((1,12,10,10), requires_grad = False),
                        torch.ones((1,), requires_grad = False))
        self.recognizer(x = dummy_input[0], y = dummy_input[1])
        
        # self.recognizer.to(self.device)
        # for state in self.recognizer.optimizer.state.values():
        #     for k, v in state.items():
        #         if torch.is_tensor(v):
        #             state[k] = v.to(self.device)
        if weight_path :
            chk = torch.load(weight_path)
            self.recognizer.net.load_state_dict(chk['model'])

    def eval(self, valid_data):
        self.recognizer.eval()
        dataloader = DataLoader(valid_data, batch_size=self.cfgs.batch_size, shuffle=True,
                                num_workers = self.cfgs.num_workers)
        test_loss = 0.0
        outs = np.asarray([])
        ys = np.asarray([])
        for step_id, data in enumerate(dataloader) :
            x, _, _, _, _, y = data
            x = torch.tensor(x).type(torch.float)
            y = torch.tensor(y).type(torch.float)
            out, loss, info = self.recognizer(x, y)
            
            test_loss += loss.detach().cpu()
            if outs.size :
                outs = np.concatenate([outs, out.detach().cpu().numpy()], 0)
            else :
                outs = out.detach().cpu().numpy()
            if ys.size :
                ys = np.concatenate([ys, y.detach().cpu().numpy()], 0)
            else :
                ys = y.detach().cpu().numpy()
        
        # Draw figure
        resolution = 100
        ys = ys.astype(bool)
        precision = []
        recall = []
        thresholds = []
        
        for i in range(resolution) :
            tp = (outs[ys]>= i/resolution).sum() + 1e-5
            tn = (outs[ys]< i/resolution).sum()
            fn = (outs[~ys]< i/resolution).sum()
            fp = (outs[~ys]>= i/resolution).sum()
            _p = tp/(tp+fp)
            _r = tp/(tn+tp)
            recall.append(_r)
            precision.append(_p)
            print(f"Precision : {_p:.4f} | Recall : {_r:.4f} | threshold : {i/resolution:.2f} | tp: {round(tp):4d}  | tn: {tn:4d}  | fn: {fn:4d}  | fp: {fp:4d}")
            thresholds.append(i/resolution)
            
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(recall, precision)
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_title("Precision-Recall curves")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        return test_loss, fig
    
    def save(self):
        torch.save({
            'model' : self.recognizer.net.state_dict(),
            'optimizer' : self.recognizer.optimizer.state_dict()
            }, 'results/weights.pth')


def get_cfgs(file_name = 'config.ini'):
    cfg = configparser.ConfigParser()
    
    cfg.read(file_name)
    cfg_keys = set(cfg.sections())
    
    cfgs = []
    for field_name in cfg_keys :
        cfgs.append(edict({k:get_values(v) for k,v in cfg.items(field_name)}))
    return edict(dict(zip(cfg_keys, cfgs)))

def get_values(v:str) :
    try:
        return literal_eval(v)
    except :
        return v
    
if __name__ == '__main__' :
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default="results")
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda', help="Which device to use")
    parser.add_argument('--cpus', type=int, default=4, help="How many CPUs to use")
    parser.add_argument('--seed', type=int, default=None, help="Random seed")
    parser.add_argument('--save_dir', type=str, default="results")
    # --------------------------------------------------------------------------
    args = parser.parse_args()
    cfgs = get_cfgs()
    
    data_root = args.data_root
    
    target_data = ['env_raw','next_env_raw','action','action_moved_box','id']
    
    test_dataset = experienceReplay(None, data_root, load = True)
    test_dataset.experience['label'] = [False for _ in range(len(test_dataset))]
    test_dataset.set_data(target_data + ['label'])
    test_dataset.grouping()
    test_dataset.preprocessing()
    
    trainer = Train(args = args,
                    cfgs = cfgs,)

    trainer.build_model(weight_path = args.weight_path)

    passed = 0
    min_loss = 1e9
    global_step = 0
    start = time.time()
        
    loss, pr_curve = trainer.eval(test_dataset)
    print(f"loss/val : {loss}")
    pr_curve.save("precision_recall_curve.png")
    