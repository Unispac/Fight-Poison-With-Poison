import ember
import os
import numpy as np
import torch

data_dir = 'data'
EMBER_DATA_DIR = os.path.join(data_dir, 'ember')

# Perform feature vectorization only if necessary.
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

x_train = x_train.astype(np.float)
y_train = y_train.astype(np.long)

#x_test = x_test.astype(np.float)
#y_test = y_test.astype(np.long)

# Get rid of unknown labels
x_train = x_train[y_train != -1]
y_train = y_train[y_train != -1]
#x_test = x_test[y_test != -1]
#y_test = y_test[y_test != -1]

dir_path = os.path.join('poisoned_train_set', 'ember', 'none')
np.save(os.path.join(dir_path, 'watermarked_X.npy'), x_train)
np.save(os.path.join(dir_path, 'watermarked_y.npy'), y_train)

poison_indicies = torch.tensor([])
torch.save(poison_indicies, os.path.join(dir_path, 'poison_indicies'))