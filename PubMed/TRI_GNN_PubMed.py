import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from tri_gnn.datasets import citation
from tri_gnn.layers import TRIGConv
from tri_gnn.utils import normalized_laplacian, rescale_laplacian, normalized_adjacency, degree_power, localpooling_filter
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Random seed
path = os.getcwd()
seed = 3
np.random.seed(seed)
tf.set_random_seed(seed)

# Load data
dataset = 'pubmed'
adj, node_features, y_train, y_val, y_test, train_mask, val_mask, test_mask = citation.load_data(dataset)
np_adj = adj.toarray()
original_adj = np_adj.copy()

pubmed_wasserstein_distance = np.load('PubMed_Wasserstein_Distance_Matrix.npz', allow_pickle=True)['arr_0']
pubmed_wasserstein_distance[np.where(pubmed_wasserstein_distance < 0.)] = 0.
topo_adj = np_adj

# add edges
added_threshold = 3.
added_edge_x, added_edge_y = np.where((pubmed_wasserstein_distance > 0) & (pubmed_wasserstein_distance < added_threshold))
np_adj[added_edge_x, added_edge_y] = topo_adj[added_edge_x, added_edge_y] + 1

# remove edges
removed_threshold = 3000. 
removed_edge_x, removed_edge_y = np.where(np.triu(pubmed_wasserstein_distance) > removed_threshold)
removed_percent = 1.
sample_removed_indices = (np.random.permutation(range(len(removed_edge_x))))[:int(len(removed_edge_x) * removed_percent)]
removed_edge_x_mask = removed_edge_x[sample_removed_indices]
removed_edge_y_mask = removed_edge_y[sample_removed_indices]

for i in range(len(removed_edge_x_mask)):
    topo_adj[removed_edge_x_mask[i], removed_edge_y_mask[i]] = 0
    topo_adj[removed_edge_y_mask[i], removed_edge_x_mask[i]] = 0


# Fraction power of W^{topo}
def power_preprocess(adj, r):
    degrees_left = np.float_power(np.array(adj.sum(1)), -r).flatten()
    degrees_left[np.isinf(degrees_left)] = 0.
    normalized_D = sp.diags(degrees_left, 0)
    degrees_right = np.float_power(np.array(adj.sum(1)), (r - 1)).flatten()
    degrees_right[np.isinf(degrees_right)] = 0.
    normalized_D_right = sp.diags(degrees_right, 0)
    adj_normalized = normalized_D.dot(adj)
    adj_normalized = adj_normalized.dot(normalized_D_right)
    return adj_normalized


# STAN module
def topo_normalization(x):
    # diagonal part
    diagonal_mask = np.eye(x.shape[0], dtype=bool)
    x[diagonal_mask] = 1.
    x[np.where(x == 0)] = 1.
    inv_x = 1. / x
    # normalize
    exp_inv_x = np.exp(inv_x) / np.sum(np.exp(inv_x), axis=1)
    return exp_inv_x


weight_factor = 0.5
distance_threshold = 10.
if distance_threshold is not None:
    x_indices, y_indices = np.where(np.triu(pubmed_wasserstein_distance, k=1) > distance_threshold)
    pubmed_wasserstein_distance[x_indices, y_indices] = distance_threshold
    pubmed_wasserstein_distance[y_indices, x_indices] = distance_threshold

node_features_arr = node_features.toarray()
topo_features = weight_factor * (topo_normalization(pubmed_wasserstein_distance).dot(node_features_arr)) + 1.0 * node_features_arr
features = sp.csr_matrix(topo_features)


# Filter for training
fltr = localpooling_filter(csr_matrix(topo_adj))


# Parameters
N = node_features.shape[0]
F = node_features.shape[1]
nhid = 16
mlp_used = True
mlp_nhid = [16]
n_classes = y_train.shape[1]
dropout_rate = 0.5
l2_reg = 5e-4
learning_rate = 1e-2
epochs = 2000
es_patience = 200
alpha = 0.2
recur_num = int(np.ceil(50 * alpha))

ori_fltr = localpooling_filter(csr_matrix(topo_adj))

# Model definition
X_in = Input(shape=(F, ))
fltr_in = Input((N, ), sparse=True)

dropout_1 = Dropout(dropout_rate)(X_in)
graph_conv_1 = TRIGConv(channels = nhid,
                       num_comp = 1,
                       num_filter = 1,
                       alpha = alpha,
                       mlp_used = True,
                       mlp_hidden = mlp_nhid,
                       recur_num = recur_num,
                       dropout_rate=dropout_rate,
                       activation='elu',
                       gcn_activation='elu',
                       kernel_regularizer=l2(l2_reg))([dropout_1, fltr_in])

dropout_2 = Dropout(dropout_rate)(graph_conv_1)
graph_conv_2 = TRIGConv(channels = n_classes,
                       num_comp = 1,
                       num_filter = 1,
                       alpha = alpha,
                       mlp_used = True,
                       mlp_hidden=[],
                       recur_num = recur_num,
                       dropout_rate=dropout_rate,
                       activation='softmax',
                       gcn_activation=None,
                       kernel_regularizer=l2(l2_reg))([dropout_2, fltr_in])

# Build model
model = Model(inputs=[X_in, fltr_in], outputs=graph_conv_2)
optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              weighted_metrics=['acc'])
model.summary()

callbacks = [
    EarlyStopping(monitor='val_weighted_acc', patience=es_patience),
    ModelCheckpoint('best_model.h5', monitor='val_weighted_acc',
                    save_best_only=True, save_weights_only=True)
]

# Train model
validation_data = ([node_features, ori_fltr], y_val, val_mask)

model.fit([node_features, fltr],
              y_train,
              sample_weight=train_mask,
              epochs=epochs,
              batch_size=N,
              validation_data=validation_data,
              shuffle=False,
              callbacks=callbacks)

# Load best model
model.load_weights('best_model.h5')

# Evaluate model
print('Evaluating model.')
eval_results = model.evaluate([node_features, ori_fltr],
                                  y_test,
                                  sample_weight=test_mask,
                                  batch_size=N)
print('Done.\n'
          'Test loss: {}\n'
          'Test accuracy: {}'.format(*eval_results))
