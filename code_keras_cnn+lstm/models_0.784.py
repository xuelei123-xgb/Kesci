
from tensorflow.keras.layers import *
from tensorflow.keras import initializers, regularizers, constraints, optimizers
from tensorflow.keras.models import load_model, Model

def Model_beta1(input_shape, classes, n_pool='average', n_l2=0.001, n_init='glorot_normal',**kwargs):
    """
    Arguments:
        input_shape -- tuple, dimensions of the input in the form (height, width, channels)
        classes -- integer, number of classes to be classified, defines the dimension of the softmax unit
        n_pool -- string, pool method to be used {'max', 'average'}
        n_dropout -- float, rate of dropping units
        n_l2 -- float, ampunt of weight decay regularization
        n_init -- string, type of kernel initializer {'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'normal', 'uniform'}
        batch_norm -- boolean, whether BatchNormalization is applied to the input

    Returns:
        model -- keras.models.Model (https://keras.io)
    """
    with_NL=kwargs.get("with_NL",False)
    name=kwargs.get("name",'AtzoriNet2')
    base_channel=kwargs.get("base_channel",64)

    activation=kwargs.get("activation","relu")


    if n_init == 'glorot_normal':
        kernel_init = initializers.glorot_normal(seed=0)
    elif n_init == 'glorot_uniform':
        kernel_init = initializers.glorot_uniform(seed=0)
    elif n_init == 'he_normal':
        kernel_init = initializers.he_normal(seed=0)
    elif n_init == 'he_uniform':
        kernel_init = initializers.he_uniform(seed=0)
    elif n_init == 'normal':
        kernel_init = initializers.normal(seed=0)
    elif n_init == 'uniform':
        kernel_init = initializers.uniform(seed=0)
    # kernel_init = n_init
    kernel_regl = regularizers.l2(n_l2)
    # kernel_regl = regularizers.l1(n_l2)
    ## Block 0 [Input]
    X_input = Input(input_shape, name='b0_input')
    X = X_input
    chanel=base_channel
    ################################################################
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(chanel, (1, 5), padding='valid', kernel_regularizer=kernel_regl, kernel_initializer=kernel_init,
               name='b{}_conv2d_1_3x3'.format(1))(X)
    X = BatchNormalization()(X)
    X = Activation(activation, name='b{}_relu1'.format(1))(X)
    
    chanel = chanel * 2
    X = ZeroPadding2D((1, 1))(X)  # （8,8)
    X = Conv2D(chanel, (1, 5), strides=(1, 2), padding='valid', kernel_regularizer=kernel_regl,
               kernel_initializer=kernel_init,
               name='b{}_conv2d_2_3x3'.format(1))(X)
    X = BatchNormalization()(X)
    X = Activation(activation, name='b{}_relu2'.format(1))(X)
    
    chanel = chanel * 2
    ################################################################
    X = ZeroPadding2D((1, 1))(X)
    X = Conv2D(chanel, (5, 3), padding='valid', kernel_regularizer=kernel_regl, kernel_initializer=kernel_init,
               name='b{}_conv2d_1_3x3'.format(2))(X)
    X = BatchNormalization()(X)
    X = Activation(activation, name='b{}_relu1'.format(2))(X)
    
    chanel = chanel * 2
    X = ZeroPadding2D((1, 1))(X)  # （8,8)
    X = Conv2D(chanel, (5, 3),strides=(2, 2), padding='valid', kernel_regularizer=kernel_regl, kernel_initializer=kernel_init,
               name='b{}_conv2d_2_3x3'.format(2))(X)

    X = BatchNormalization()(X)
    X = Activation(activation, name='b{}_relu2'.format(2))(X)

    X=GlobalAveragePooling2D()(X)

    X=Dropout(0.5)(X)
    ## Block 5 [Pad -> Conv -> Softmax]
    X = Dense(classes,activation="softmax")(X)

    model = Model(inputs=X_input, outputs=X, name=name,)

    return model

# if __name__ == '__main__':

    # model=Model_alpha(input_shape=(8, 8, 60),classes=19)
    # model.summary()
