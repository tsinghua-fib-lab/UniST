import numpy as np
import torch as th
import json
import torch
import datetime
import copy
import random

class MinMaxNormalization(object):
    """
        MinMax Normalization --> [-1, 1]
        x = (x - min) / (max - min).
        x = x * 2 - 1
    """

    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X



def data_load_single(args, dataset): 

    folder_path = '../dataset/{}_{}.json'.format(dataset,args.task)
    f = open(folder_path,'r')
    data_all = json.load(f)

    X_train = torch.tensor(data_all['X_train'][0]).unsqueeze(1)
    X_test = torch.tensor(data_all['X_test'][0]).unsqueeze(1)
    X_val = torch.tensor(data_all['X_val'][0]).unsqueeze(1)

    X_train_period = torch.tensor(data_all['X_train'][1]).permute(0,2,1,3,4)
    X_test_period = torch.tensor(data_all['X_test'][1]).permute(0,2,1,3,4)
    X_val_period = torch.tensor(data_all['X_val'][1]).permute(0,2,1,3,4)

    # X_train = torch.randn([1000, 1, 12, 16, 8])
    # X_test = torch.randn([100, 1, 12, 16, 8])
    # X_val = torch.randn([100, 1, 12, 16, 8])

    # X_train_period = torch.randn([1000, 3, 12, 16, 8])
    # X_test_period = torch.randn([100, 3, 12, 16, 8])
    # # X_val_period = torch.randn([100, 3, 12, 16, 8])

    # data_all = {}
    # data_all['timestamps'] = {'train':torch.zeros([1000,12,2]).long(), 'test':torch.zeros([100,12,2]).long(), 'val':torch.zeros([100,12,2]).long()}

    args.seq_len = X_train.shape[2]
    H, W = X_train.shape[3], X_train.shape[4]  

    if 'TaxiBJ' in dataset:
        X_train_ts = data_all['timestamps']['train']
        X_test_ts = data_all['timestamps']['test']
        X_val_ts = data_all['timestamps']['val']

        X_train_ts = torch.tensor([[(datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').weekday(),datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').hour*2+int(datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').minute>=30)) for i in t] for t in X_train_ts])
        X_test_ts = torch.tensor([[(datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').weekday(),datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').hour*2+int(datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').minute>=30)) for i in t] for t in X_test_ts])
        X_val_ts = torch.tensor([[(datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').weekday(),datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').hour*2+int(datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').minute>=30)) for i in t] for t in X_val_ts])

    elif 'Crowd' in dataset or 'Cellular' in dataset or 'Traffic_log' in dataset:
        X_train_ts = data_all['timestamps']['train']
        X_test_ts = data_all['timestamps']['test']
        X_val_ts = data_all['timestamps']['val']

        X_train_ts = torch.tensor([[((i%(24*2*7)//(24*2)+2)%7,i%(24*2)) for i in t] for t in X_train_ts])
        X_test_ts = torch.tensor([[((i%(24*2*7)//(24*2)+2)%7, i%(24*2)) for i in t] for t in X_test_ts])
        X_val_ts = torch.tensor([[((i%(24*2*7)//(24*2)+2)%7, i%(24*2)) for i in t] for t in X_val_ts])

    elif 'TaxiNYC' in dataset or 'BikeNYC' in dataset or 'TDrive' in dataset or 'Traffic' in dataset or 'DC' in dataset or 'Austin' in dataset or 'Porto' in dataset or 'CHI' in dataset:
        X_train_ts = torch.tensor(data_all['timestamps']['train'])
        X_test_ts = torch.tensor(data_all['timestamps']['test'])
        X_val_ts = torch.tensor(data_all['timestamps']['val'])

    my_scaler = MinMaxNormalization()
    MAX = max(torch.max(X_train).item(), torch.max(X_test).item(), torch.max(X_val).item())
    MIN = min(torch.min(X_train).item(), torch.min(X_test).item(), torch.min(X_val).item())
    my_scaler.fit(np.array([MIN, MAX]))

    X_train = my_scaler.transform(X_train.reshape(-1,1)).reshape(X_train.shape)
    X_test = my_scaler.transform(X_test.reshape(-1,1)).reshape(X_test.shape)
    X_val = my_scaler.transform(X_val.reshape(-1,1)).reshape(X_val.shape)
    X_train_period = my_scaler.transform(X_train_period.reshape(-1,1)).reshape(X_train_period.shape)
    X_test_period = my_scaler.transform(X_test_period.reshape(-1,1)).reshape(X_test_period.shape)
    X_val_period = my_scaler.transform(X_val_period.reshape(-1,1)).reshape(X_val_period.shape)

    data = [[X_train[i], X_train_ts[i], X_train_period[i]] for i in range(X_train.shape[0])]
    test_data = [[X_test[i], X_test_ts[i], X_test_period[i]] for i in range(X_test.shape[0])]
    val_data = [[X_val[i], X_val_ts[i], X_val_period[i]] for i in range(X_val.shape[0])]

    if args.mode == 'few-shot':
        data = data[:int(len(data)*args.few_ratio)]

    if H + W < 32:
        batch_size = args.batch_size_1
    elif H + W < 48:
        batch_size = args.batch_size_2
    elif H + W < 64:
        batch_size = args.batch_size_3

    data = th.utils.data.DataLoader(data, num_workers=4, batch_size=batch_size, shuffle=True) 
    test_data = th.utils.data.DataLoader(test_data, num_workers=4, batch_size = 4 * batch_size, shuffle=False)
    val_data = th.utils.data.DataLoader(val_data, num_workers=4, batch_size = 4 * batch_size, shuffle=False)

    return  data, test_data, val_data, my_scaler

def data_load(args):

    data_all = []
    test_data_all = []
    val_data_all = []
    my_scaler_all = []
    my_scaler_all = {}

    for dataset_name in args.dataset.split('*'):
        data, test_data, val_data, my_scaler = data_load_single(args,dataset_name)
        data_all.append([dataset_name, data])
        test_data_all.append(test_data)
        val_data_all.append(val_data)
        my_scaler_all[dataset_name] = my_scaler

    data_all = [(name,i) for name, data in data_all for i in data]
    random.seed(1111)
    random.shuffle(data_all)
    
    return data_all, test_data_all, val_data_all, my_scaler_all


def data_load_main(args):

    data, test_data, val_data, scaler = data_load(args)

    return data, test_data, val_data, scaler

