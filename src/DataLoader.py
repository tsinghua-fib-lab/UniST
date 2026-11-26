import numpy as np
import torch as th
import json
import pickle
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


def taxibj_timestamp_features(windows, external, ext_table):
    """
    windows: list (one entry per window) of lists of 'YYYY-MM-DD HH:MM:SS' strings.
    Returns a tensor (N, T, C). Columns: weekday, half-hour of day  (as in the original loader),
    then, if external >= 1: holiday flag; if external >= 2: temperature, wind, weather type.
    ext_table comes from make_external.py: {slot_label: (holiday, temp, wind, weather_cat)}.
    """
    out = []
    for window in windows:
        rows = []
        for s in window:
            d = datetime.datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
            slot = d.hour * 2 + int(d.minute >= 30)
            row = [d.weekday(), slot]
            if external >= 1:
                label = d.strftime('%Y%m%d') + '%02d' % (slot + 1)
                holiday, temp, wind, cat = ext_table[label]
                row.append(holiday)
                if external >= 2:
                    row += [temp, wind, cat]
            rows.append(row)
        out.append(rows)
    return torch.tensor(out)


def data_load_single(args, dataset): 

    external = getattr(args, 'external', 0)

    if 'TaxiBJ' in dataset:
        # TaxiBJ comes from a compact .npz built by make_taxibj_npz.py (a JSON of this size would not fit in memory)
        z = np.load('../dataset/{}_{}.npz'.format(dataset, args.task))

        X_train = torch.from_numpy(z['X_train']).unsqueeze(1)
        X_test = torch.from_numpy(z['X_test']).unsqueeze(1)
        X_val = torch.from_numpy(z['X_val']).unsqueeze(1)

        # periodic frames are stored once per window, (N, 3, H, W); the JSON format repeats them over
        # the window positions, giving (N, 12, 3, H, W) and, after the permute below, (N, 3, 12, H, W)
        T = X_train.shape[2]
        X_train_period = torch.from_numpy(z['P_train']).unsqueeze(2).repeat(1, 1, T, 1, 1)
        X_test_period = torch.from_numpy(z['P_test']).unsqueeze(2).repeat(1, 1, T, 1, 1)
        X_val_period = torch.from_numpy(z['P_val']).unsqueeze(2).repeat(1, 1, T, 1, 1)

        ext_table = pickle.load(open('../dataset/external_taxibj.pkl', 'rb')) if external else None
        X_train_ts = taxibj_timestamp_features(z['T_train'].tolist(), external, ext_table)
        X_test_ts = taxibj_timestamp_features(z['T_test'].tolist(), external, ext_table)
        X_val_ts = taxibj_timestamp_features(z['T_val'].tolist(), external, ext_table)
        print('TaxiBJ windows: train {}, val {}, test {} | external = {} | timestamp columns = {}'.format(
            len(X_train), len(X_val), len(X_test), external, X_train_ts.shape[-1]))

    else:
        folder_path = '../dataset/{}_{}.json'.format(dataset,args.task)
        f = open(folder_path,'r')
        data_all = json.load(f)

        X_train = torch.tensor(data_all['X_train'][0]).unsqueeze(1)
        X_test = torch.tensor(data_all['X_test'][0]).unsqueeze(1)
        X_val = torch.tensor(data_all['X_val'][0]).unsqueeze(1)

        X_train_period = torch.tensor(data_all['X_train'][1]).permute(0,2,1,3,4)
        X_test_period = torch.tensor(data_all['X_test'][1]).permute(0,2,1,3,4)
        X_val_period = torch.tensor(data_all['X_val'][1]).permute(0,2,1,3,4)

        if 'Crowd' in dataset or 'Cellular' in dataset or 'Traffic_log' in dataset:
            X_train_ts = data_all['timestamps']['train']
            X_test_ts = data_all['timestamps']['test']
            X_val_ts = data_all['timestamps']['val']

            X_train_ts = torch.tensor([[((i%(24*2*7)//(24*2)+2)%7,i%(24*2)) for i in t] for t in X_train_ts])
            X_test_ts = torch.tensor([[((i%(24*2*7)//(24*2)+2)%7, i%(24*2)) for i in t] for t in X_test_ts])
            X_val_ts = torch.tensor([[((i%(24*2*7)//(24*2)+2)%7, i%(24*2)) for i in t] for t in X_val_ts])

        elif 'TaxiNYC' in dataset or 'BikeNYC' in dataset or 'TDrive' in dataset or 'Traffic' in dataset or 'DC' in dataset or 'Austin' in dataset or 'Porto' in dataset or 'CHI' in dataset or 'METR-LA' in dataset or 'CrowdBJ' in dataset:
            X_train_ts = torch.tensor(data_all['timestamps']['train'])
            X_test_ts = torch.tensor(data_all['timestamps']['test'])
            X_val_ts = torch.tensor(data_all['timestamps']['val'])

    args.seq_len = X_train.shape[2]
    H, W = X_train.shape[3], X_train.shape[4]  

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

    if 'TaxiBJ' in dataset and 0 < args.few_ratio < 1.0:
        # few-shot subset spread evenly over the chronological training windows (all four periods),
        # instead of only the earliest windows (the first months of 2013, with almost no holidays)
        m = max(1, int(len(data) * args.few_ratio))
        keep = np.linspace(0, len(data) - 1, m).round().astype(int)
        data = [data[i] for i in keep]
        print('few-shot: using {} training windows'.format(len(data)))
    elif args.mode == 'few-shot':
        data = data[:int(len(data)*args.few_ratio)]

    if H + W < 32:
        batch_size = args.batch_size_1
    elif H + W < 48:
        batch_size = args.batch_size_2
    elif H + W < 64:
        batch_size = args.batch_size_3
    else:
        batch_size = args.batch_size_3   # the original code had no branch for H + W >= 64 (TaxiBJ is 32 x 32)

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