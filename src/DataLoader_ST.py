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
    f.close()

    X_train = torch.tensor(data_all['X_train'][0]).unsqueeze(1)
    X_test = torch.tensor(data_all['X_test'][0]).unsqueeze(1)
    X_val = torch.tensor(data_all['X_val'][0]).unsqueeze(1)


    X_train_period = torch.tensor(data_all['X_train'][1]).permute(0,2,1,3,4)
    X_test_period = torch.tensor(data_all['X_test'][1]).permute(0,2,1,3,4)
    X_val_period = torch.tensor(data_all['X_val'][1]).permute(0,2,1,3,4)


    args.seq_len = X_train.shape[2]
    H, W = X_train.shape[3], X_train.shape[4]


    if 'TaxiBJ' in dataset:
        X_train_ts = data_all['timestamps']['train']
        X_test_ts = data_all['timestamps']['test']

        X_val_ts = copy.deepcopy(X_test_ts[:int(len(X_test_ts)*0.5)])
        X_test_ts = copy.deepcopy(X_test_ts[int(len(X_test_ts)*0.5):])

        X_train_ts = torch.tensor([[(datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').weekday(),datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').hour*2+int(datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').minute>=30)) for i in t] for t in X_train_ts])
        X_test_ts = torch.tensor([[(datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').weekday(),datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').hour*2+int(datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').minute>=30)) for i in t] for t in X_test_ts])
        X_val_ts = torch.tensor([[(datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').weekday(),datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').hour*2+int(datetime.datetime.strptime(i,'%Y-%m-%d %H:%M:%S').minute>=30)) for i in t] for t in X_val_ts])

    elif 'Crowd' in dataset or 'Cellular' in dataset:
        X_train_ts = data_all['timestamps']['train']
        X_test_ts = data_all['timestamps']['test']
        X_val_ts = data_all['timestamps']['val']

        X_train_ts = torch.tensor([[((i%(24*2*7)//(24*2)+2)%7,i%(24*2)) for i in t] for t in X_train_ts])
        X_test_ts = torch.tensor([[((i%(24*2*7)//(24*2)+2)%7, i%(24*2)) for i in t] for t in X_test_ts])
        X_val_ts = torch.tensor([[((i%(24*2*7)//(24*2)+2)%7, i%(24*2)) for i in t] for t in X_val_ts])

    elif 'TaxiNYC' in dataset or 'BikeNYC' in dataset or 'TDrive' in dataset or 'Traffic' in dataset or 'DC' in dataset or 'CHI' in dataset or 'Porto' in dataset or 'Austin' in dataset:
        X_train_ts = torch.tensor(data_all['timestamps']['train'])
        X_test_ts = torch.tensor(data_all['timestamps']['test'])
        X_val_ts = torch.tensor(data_all['timestamps']['val'])


    my_scaler = MinMaxNormalization()
    MAX = max(torch.max(X_train).item(), torch.max(X_test).item(), torch.max(X_val).item())
    MIN = min(torch.min(X_train).item(), torch.min(X_test).item(), torch.min(X_val).item())
    my_scaler.fit(np.array([MIN, MAX]))

    args.height = X_train.shape[-2]
    args.width = X_train.shape[-1]

    X_train = my_scaler.transform(X_train.reshape(-1,1)).reshape(X_train.shape)
    X_test = my_scaler.transform(X_test.reshape(-1,1)).reshape(X_test.shape)
    X_val = my_scaler.transform(X_val.reshape(-1,1)).reshape(X_val.shape)
    X_train_period = my_scaler.transform(X_train_period.reshape(-1,1)).reshape(X_train_period.shape)
    X_test_period = my_scaler.transform(X_test_period.reshape(-1,1)).reshape(X_test_period.shape)
    X_val_period = my_scaler.transform(X_val_period.reshape(-1,1)).reshape(X_val_period.shape)

    data = [[X_train[i], X_train_ts[i], X_train_period[i]] for i in range(X_train.shape[0])]
    test_data = [[X_test[i], X_test_ts[i], X_test_period[i]] for i in range(X_test.shape[0])]
    val_data = [[X_val[i], X_val_ts[i], X_val_period[i]] for i in range(X_val.shape[0])]

    if 'long' not in args.task:
        if 'TaxiBJ' in dataset:
            random.seed(555)
            data = random.sample(data, 5000)
            test_data = random.sample(test_data, 500)
            val_data = random.sample(val_data, 500)

            if len(args.dataset.split('*'))>=10:
                random.seed(520)
                data = random.sample(data, 1000)

        elif 'TDrive' in dataset:
            random.seed(555)
            test_data = random.sample(test_data, 500)
            val_data = random.sample(val_data, 500)

    if args.mode == 'few-shot':
        data = data[:int(len(data)*args.few_ratio)]

    if 'BikeNYC' in dataset or 'TaxiNYC' in dataset  or 'DC' in dataset or 'CHI' in dataset or 'Porto' in dataset or 'Austin' in dataset:
        batch_size = args.batch_size_nyc
    elif 'weather' in dataset:
        batch_size = args.batch_size_weather
    elif 'Crowd' in dataset or 'Traffic_log' in dataset:
        batch_size = args.batch_size_nj
    elif 'TaxiBJ' in dataset or 'TDrive' in dataset or 'Traffic' in dataset:
        batch_size = args.batch_size_taxibj

        if H + W < 48:
            batch_size *= 2

    data = th.utils.data.DataLoader(data, num_workers=4, batch_size=batch_size, shuffle=True) 
    
    test_batch_size = 4 * batch_size

    test_data = th.utils.data.DataLoader(test_data, num_workers=4, batch_size = test_batch_size, shuffle=False)
    val_data = th.utils.data.DataLoader(val_data, num_workers=4, batch_size = test_batch_size, shuffle=False)

    return  data, test_data, val_data, my_scaler


def data_load_mix(args, data_list):
    data_all = []

    for data in data_list:
        data_all += data

    data_all = th.utils.data.DataLoader(data_all, num_workers=4, batch_size=args.batch_size, shuffle=True)

    return data_all


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

