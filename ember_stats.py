import torch
import numpy as np
import ember
import os

data_dir = './data'

EMBER_DATA_DIR = os.path.join(data_dir, 'ember')


try:
    x_train, y_train, x_test, y_test = ember.read_vectorized_features(
            EMBER_DATA_DIR,
            feature_version=1
        )

except:
    ember.create_vectorized_features(
            EMBER_DATA_DIR,
            feature_version=1
        )
    x_train, y_train, x_test, y_test = ember.read_vectorized_features(
            EMBER_DATA_DIR,
            feature_version=1
        )

x_train = x_train.astype(dtype='float')
x_train = torch.FloatTensor(x_train)
mean = torch.mean(x_train, dim=0)
std = torch.std(x_train, dim=0)

print(mean.shape, std.shape)

stats = {'mean' : mean, 'std' : std}
torch.save(stats, 'stats')
print('[saved]')
