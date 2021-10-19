from __future__ import absolute_import
from keras import activations, initializers, regularizers, constraints
from keras import backend as K
from keras.layers import Layer, LeakyReLU, Dropout, AveragePooling2D, AveragePooling1D, MaxPooling1D, MaxPooling2D, Conv1D, BatchNormalization, Conv2D, Flatten, GlobalMaxPooling1D
from keras.constraints import max_norm, non_neg, unit_norm, min_max_norm
from keras.initializers import RandomUniform
import tensorflow as tf
import numpy as np

_LAYER_UIDS = {}
def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

class TRIGConv(Layer):

    def __init__(self,
                 channels,
                 num_filter,
                 alpha=0.6,
                 mlp_used=True,
                 mlp_hidden=None,
                 mlp_activation='relu', # fixed
                 num_comp=None,
                 recur_num = None,
                 gcn_activation='relu',
                 dropout_rate=0.0,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(TRIGConv, self).__init__(**kwargs)
        self.channels = channels 
        self.num_filter = num_filter
        self.alpha = alpha
        self.mlp_used = mlp_used  # whether use mlp as update function
        self.mlp_hidden = mlp_hidden if mlp_hidden else [] # hidden dimension for MLP
        self.mlp_activation = activations.get(mlp_activation)
        self.num_comp = num_filter if num_comp is None else num_comp
        self.recur_num = recur_num
        self.activation = activations.get(activation)
        self.gcn_activation = activations.get(gcn_activation)
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.supports_masking = False

    def build(self, input_shape):
        assert len(input_shape) >= 2

        self.kernels_in = []  # Weights from input space to output space
        self.kernels_hid = []  # Weights from output space to output space
        for k in range(self.num_comp):
            self.kernels_in.append(self.get_gcn_weights(input_shape[0][-1],
                                                        input_shape[0][-1],
                                                        self.channels,
                                                        name='TRIGConv_skip_{}r_in'.format(k),
                                                        use_bias=self.use_bias,
                                                        recur_num = self.recur_num,
                                                        kernel_initializer=self.kernel_initializer,
                                                        bias_initializer=self.bias_initializer,
                                                        kernel_regularizer=self.kernel_regularizer,
                                                        bias_regularizer=self.bias_regularizer,
                                                        kernel_constraint=self.kernel_constraint,
                                                        bias_constraint=self.bias_constraint))

            if self.num_filter > 1:
                for d in range(self.num_filter-1):
                    self.kernels_hid.append(self.get_gcn_weights(self.channels,
                                                                 input_shape[0][-1],
                                                                 self.channels,
                                                                 name='TRIGConv_skip_{}r_hid'.format(
                                                                 k * len(range(self.num_filter-1)) + d),
                                                                 use_bias=self.use_bias,
                                                                 recur_num=self.recur_num,
                                                                 kernel_initializer=self.kernel_initializer,
                                                                 bias_initializer=self.bias_initializer,
                                                                 kernel_regularizer=self.kernel_regularizer,
                                                                 bias_regularizer=self.bias_regularizer,
                                                                 kernel_constraint=self.kernel_constraint,
                                                                 bias_constraint=self.bias_constraint))

        if self.mlp_used:
            # For f_update function
            self.kernels_mlp = []
            self.biases_mlp = []

            # Hidden layers
            input_dim = input_shape[0][-1]
            for i, f_update_channels in enumerate(self.mlp_hidden):
                self.kernels_mlp.append(
                    self.add_weight(shape=(input_dim, f_update_channels),
                                    initializer=self.kernel_initializer,
                                    name='kernel_mlp_{}'.format(i),
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint)
                )
                if self.use_bias:
                    self.biases_mlp.append(
                        self.add_weight(shape=(f_update_channels,),
                                        initializer=self.bias_initializer,
                                        name='bias_mlp_{}'.format(i),
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
                    )
                input_dim = f_update_channels

            # Output layer
            self.kernel_out = self.add_weight(shape=(input_dim, self.channels),
                                              initializer=self.kernel_initializer,
                                              name='kernel_mlp_out',
                                              regularizer=self.kernel_regularizer,
                                              constraint=self.kernel_constraint)
            if self.use_bias:
                self.bias_out = self.add_weight(shape=(self.channels,),
                                                initializer=self.bias_initializer,
                                                name='bias_mlp_out',
                                                regularizer=self.bias_regularizer,
                                                constraint=self.bias_constraint)

        self.built = True

    def call(self, inputs):
        features = inputs[0]
        fltr = inputs[1]

        # Convolution
        output = []
        for k in range(self.num_comp):
            output_k = features
            for d in range(self.num_filter):
                features_drop = Dropout(self.dropout_rate)(features)

                output_k = self.graph_conv_skip([output_k, features_drop, fltr],
                                                self.channels,
                                                alpha = self.alpha,
                                                recurrent_k=k,
                                                recurrent_d=d,
                                                activation=self.gcn_activation,
                                                use_bias=self.use_bias,
                                                recur_num = self.recur_num,
                                                kernel_initializer=self.kernel_initializer,
                                                bias_initializer=self.bias_initializer,
                                                kernel_regularizer=self.kernel_regularizer,
                                                bias_regularizer=self.bias_regularizer,
                                                kernel_constraint=self.kernel_constraint,
                                                bias_constraint=self.bias_constraint)
            output.append(output_k)

        output = K.concatenate(output, axis=-1)
        output = K.expand_dims(output, axis=-1)
        output_dim = K.int_shape(output)

        if len(output_dim) == 3:
            if self.channels != 6:
                output = self.gated_max_avg_pooling(pooling_input=output)

            elif self.channels == 6:
                output = self.gated_max_avg_pooling(pooling_input=output)

        elif len(output_dim) == 4:
            output = tf.reshape(output, [-1,self.channels,self.num_comp,1])
            output = AveragePooling2D(pool_size=(1, self.num_comp), padding='same')(output)
            output = tf.reshape(output, [-1,self.channels,1]) 
        else:
            raise RuntimeError('TRI-GNN layer: wrong output dimension')
        output = K.squeeze(output, axis=-1)

        if self.channels != 6:
            output = tf.nn.elu(output)+ output
            output = tf.nn.elu(output)

        elif self.channels == 6:
            output = tf.nn.softmax(output)

        return output

    def compute_output_shape(self, input_shape):
        features_shape = input_shape[0]
        if self.channels != 6:
            output_shape = features_shape[:-1] + (self.channels,)  
        elif self.channels == 6:
            output_shape = features_shape[:-1] + (self.channels,)
        return output_shape

    def get_config(self):
        config = {
            'channels': self.channels,
            'num_filter': self.num_filter,
            'alpha': self.alpha,
            'mlp_hidden': self.mlp_hidden,
            'mlp_activation': activations.serialize(self.mlp_activation),
            'num_comp': self.num_comp,
            'gcn_activation': activations.serialize(self.gcn_activation),
            'dropout_rate': self.dropout_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(TRIGConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_gcn_weights(self, input_dim, input_dim_skip, channels, name,
                        recur_num,
                        use_bias=True,
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros',
                        kernel_regularizer=None,
                        bias_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None):

        kernel_initializer = initializers.get(kernel_initializer)
        kernel_regularizer = regularizers.get(kernel_regularizer)
        kernel_constraint = constraints.get(kernel_constraint)
        bias_initializer = initializers.get(bias_initializer)
        bias_regularizer = regularizers.get(bias_regularizer)
        bias_constraint = constraints.get(bias_constraint)

        kernel_1 = self.add_weight(shape=(input_dim, channels),
                                   name=name + '_kernel_1',
                                   initializer=kernel_initializer,
                                   regularizer=kernel_regularizer,
                                   constraint=kernel_constraint)


        if use_bias:
            bias = self.add_weight(shape=(channels,),
                                   name=name + '_bias',
                                   initializer=bias_initializer,
                                   regularizer=bias_regularizer,
                                   constraint=bias_constraint)
        else:
            bias = None
        return kernel_1, bias

    # Gated-max-average pooling layer
    def gated_max_avg_pooling(self, pooling_input):

        transformed_pooling_input = tf.reshape(pooling_input, shape=[-1, 1])
        alpha_kernel = self.get_pooling_weight(input_shape=self.num_comp * self.channels)
        sigmoid_alpha_kernel = tf.matmul(alpha_kernel, transformed_pooling_input)
        sigmoid_alpha_kernel = tf.sigmoid(sigmoid_alpha_kernel)

        x1 = AveragePooling1D(pool_size=self.num_comp, padding='same')(pooling_input)
        x2 = MaxPooling1D(pool_size=self.num_comp, padding='same')(pooling_input)
        temp_output = tf.add(tf.multiply(x1, sigmoid_alpha_kernel), tf.multiply(x2, (1 - sigmoid_alpha_kernel)))
        output = temp_output

        return output

    # The weight variable for gated-max-average pooling layer
    def get_pooling_weight(self, input_shape,
                           kernel_initializer='glorot_uniform',
                           kernel_regularizer=None,
                           kernel_constraint=None):
        layer = self.__class__.__name__.lower()
        pooling_kernel = self.add_weight(shape=(1, input_shape * 3327),
                                         name='max_avg_pooling_kernel' + '_' + layer + '_' + str(
                                         get_layer_uid(layer)),
                                         initializer=kernel_initializer,
                                         regularizer=kernel_regularizer,
                                         constraint=kernel_constraint,
                                         trainable=True)
        return pooling_kernel


    def graph_conv_skip(self, x, channels,
                        alpha = None,
                        recurrent_k=None,
                        recurrent_d=None,
                        activation=None,
                        use_bias=True,
                        recur_num = None,
                        kernel_initializer='glorot_uniform',
                        bias_initializer='zeros',
                        kernel_regularizer=None,
                        bias_regularizer=None,
                        kernel_constraint=None,
                        bias_constraint=None):

        if recurrent_d == 0:
            kernel_1, bias = self.kernels_in[recurrent_k]
        else:
            kernel_1, bias = self.kernels_hid[recurrent_k * (len(range(self.num_filter-1))) + recurrent_d -1]
           
        features = x[0]
        features_skip = x[1]
        fltr = x[2]

        # Update function
        # Compute MLP hidden features
        if self.mlp_used:
            for i in range(len(self.kernels_mlp)):
                features = Dropout(self.dropout_rate)(features)
                features = K.dot(features, self.kernels_mlp[i])
                if self.use_bias:
                    features += self.biases_mlp[i]
                if self.mlp_activation is not None:
                    features = self.mlp_activation(features)

            # Compute MLP output
            mlp_out = K.dot(features, self.kernel_out)
            if self.use_bias:
                mlp_out += self.bias_out

            # TRI-GNN convolutional layer
            alpha_ = 1.0 / alpha
            recur_depth = recur_num
            new_feature = mlp_out
            for _ in range(recur_depth):
                new_feature = ((alpha_ - 1) / alpha_) * filter_dot(fltr, new_feature)
                new_feature += (1 / alpha_) * mlp_out
            output = new_feature

        else:
            # TRI-GNN convolutional layer
            alpha_ = 1.0 / alpha
            recur_depth = recur_num
            normalized_adj = fltr / alpha_
            new_feature = features
            for _ in range(recur_depth):
                new_feature = K.dot(normalized_adj, new_feature)
                new_feature += features
            new_feature = new_feature - features + features_skip
            new_feature *= (alpha_ - 1) / alpha_
            output = K.dot(new_feature, kernel_1)

        if use_bias:
            output = K.bias_add(output, bias)
        if activation is not None:
            output = activations.get(activation)(output)
        return output


# Dot-product operation
def mixed_mode_dot(fltr, features):
    _, m_, f_ = K.int_shape(features)
    features = K.permute_dimensions(features, [1, 2, 0])
    features = K.reshape(features, (m_, -1))
    features = K.dot(fltr, features)
    features = K.reshape(features, (m_, f_, -1))
    features = K.permute_dimensions(features, [2, 0, 1])

    return features


def filter_dot(fltr, features):
    if len(K.int_shape(features)) == 2:
        # Single mode
        return K.dot(fltr, features)
    else:
        if len(K.int_shape(fltr)) == 3:
            # Batch mode
            return K.batch_dot(fltr, features)
        else:
            # Mixed mode
            return mixed_mode_dot(fltr, features)