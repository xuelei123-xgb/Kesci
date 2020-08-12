
from tensorflow.keras.layers import *
from tensorflow.keras import initializers, regularizers, constraints, optimizers
from tensorflow.keras.models import load_model, Model
import tensorflow as tf



def Model_RNN3(input_shape, classes, n_pool='average', n_l2=0.001, n_init='he_normal',**kwargs):
    name=kwargs.get("name",'AtzoriNet2')
    base_channel=kwargs.get("base_channel",64)

    if n_init == 'glorot_normal':
        kernel_init = initializers.glorot_normal(seed=0)
    elif n_init == 'glorot_uniform':
        kernel_init = initializers.glorot_uniform(seed=0)
    elif n_init == 'he_normal':
        kernel_init = initializers.he_normal(seed=0)
    elif n_init == 'he_uniform':
        kernel_init = initializers.x(seed=0)
    elif n_init == 'normal':
        kernel_init = initializers.normal(seed=0)
    elif n_init == 'uniform':
        kernel_init = initializers.uniform(seed=0)
    # kernel_init = n_init
    kernel_regl = regularizers.l2(n_l2)

    ## Block 0 [Input]
    X_input = Input(input_shape, name='b0_input')
    X = X_input
    chanel=base_channel
    ################################################################
    # X=AveragePooling2D((3,1),strides=(1,1),padding='same')(X)
    X = ZeroPadding2D((0, 1))(X)
    X = Conv2D(chanel, (1, 3), padding='valid', kernel_regularizer=kernel_regl, kernel_initializer=kernel_init,
               name='b{}_conv2d_1_3x3'.format(1))(X)
    X = LayerNormalization()(X)
    X = Activation('relu', name='b{}_relu1'.format(1))(X)
    chanel = chanel * 2
    X = ZeroPadding2D((0, 1))(X)  # （8,8)
    X = Conv2D(chanel, (1, 3), strides=(1, 2), padding='valid', kernel_regularizer=kernel_regl,
               kernel_initializer=kernel_init,
               name='b{}_conv2d_2_3x3'.format(1))(X)
    X = LayerNormalization()(X)
    X = Activation('relu', name='b{}_relu2'.format(1))(X)
    ################################################################
    # CNN to RNN
    inner = Reshape(target_shape=((60, chanel*4)), name='reshape')(X)  # (None, 64, 384)
    inner = Dense(512, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, 60, 384)
    ################################################################
    # RNN layer
    gru_1 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='gru1')(inner)  # (None, 60, 512)
    gru_1b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(inner)
    # reversed_gru_1b = Lambda(lambda inputTensor: tf.reverse(inputTensor, axes=1))(gru_1b)
    reversed_gru_1b=tf.reverse(gru_1b,axis=[1])
    gru1_merged = add([gru_1, reversed_gru_1b])  # (None, 60, 512)
    gru1_merged = LayerNormalization()(gru1_merged)

    gru_2 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(
        gru1_merged)
    # reversed_gru_2b = Lambda(lambda inputTensor: tf.reverse(inputTensor, axes=1))(gru_2b)
    reversed_gru_2b=tf.reverse(gru_2b,axis=[1])
    gru2_merged = concatenate([gru_2, reversed_gru_2b])  # (None, 60, 1024)
    X=gru2_merged
    ################################################################
    # X = BatchNormalization()(X)
    X=GlobalAveragePooling1D()(X)
    X=Dropout(0.5)(X)
    ## Block 5 [Pad -> Conv -> Softmax]
    X = Dense(classes,activation="softmax")(X)
    model = Model(inputs=X_input, outputs=X, name=name,)

    return model

