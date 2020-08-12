import  tensorflow as tf
from tensorflow.keras import callbacks
import  numpy as np
K = tf.keras.backend
class CosineAnnealing(callbacks.Callback):
    """Cosine annealing according to DECOUPLED WEIGHT DECAY REGULARIZATION.

    # Arguments
        eta_max: float, eta_max in eq(5).
        eta_min: float, eta_min in eq(5).
        verbose: 0 or 1.
    """
    def __init__(self, epochs,eta_max, eta_min, iteration_of_epoch=0, verbose=0, **kwargs):
        super(CosineAnnealing, self).__init__()
        self.lr_list = []
        self.epochs=sorted(epochs)
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose
        self.iteration_of_epoch=iteration_of_epoch


        #########
        self.num_period=0
        self.period_begin=0
        self.period_end=self.epochs[0]
        self.iteration = 0
        self.total_iteration = (self.period_end-self.period_begin)*self.iteration_of_epoch

    def on_train_begin(self, logs=None):
        self.lr = K.get_value(self.model.optimizer.lr)

    def on_train_end(self, logs=None):
        K.set_value(self.model.optimizer.lr, self.lr)

    def on_epoch_begin(self, epoch, logs=None):
        epos=sorted(self.epochs+[epoch])
        loc= epos.index(epoch)
        if loc==self.num_period:
            return
        self.num_period+=1
        self.period_begin, self.period_end=self.epochs[loc-1],self.epochs[loc]
        self.iteration = 0
        self.total_iteration=(self.period_end-self.period_begin)*self.iteration_of_epoch
        print(self.period_begin, self.period_end)

    def on_batch_end(self, epoch, logs=None):
        self.iteration += 1
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

        eta_t = self.eta_min + (self.eta_max - self.eta_min) * 0.5 * (
                    1 + np.cos(np.pi * self.iteration / self.total_iteration))
        # new_lr = self.lr * eta_t/
        new_lr = eta_t
        K.set_value(self.model.optimizer.lr, new_lr)
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealing '
                  'learning rate to %s.' % (epoch + 1, new_lr))
        self.lr_list.append(logs['lr'])


if __name__ == '__main__':
    from tensorflow.keras import layers
    from tensorflow import  keras
    # 准备数据集
    num_train, num_test = 200, 100
    num_features = 200

    true_w, true_b = np.ones((num_features, 1)) * 0.01, 0.05

    features = np.random.normal(0, 1, (num_train + num_test, num_features))
    noises = np.random.normal(0, 1, (num_train + num_test, 1)) * 0.01
    labels = np.dot(features, true_w) + true_b + noises

    train_data, test_data = features[:num_train, :], features[num_train:, :]
    train_labels, test_labels = labels[:num_train], labels[num_train:]

    # 选择模型
    model = keras.models.Sequential([
        layers.Dense(units=128, activation='relu', input_dim=200),
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.00)),
        layers.Dense(1)
    ])

    model.summary()
    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mse'])

    epochs = [20, 40, 80, 160, 250]


    reduce_lr = CosineAnnealing( epochs,eta_max=0.1, eta_min=0.00005, iteration_of_epoch=200//20, verbose=0, )
    hist1 = model.fit(train_data, train_labels, batch_size=20, epochs=250, validation_data=[test_data, test_labels],
                      callbacks=[reduce_lr])


    import matplotlib.pyplot as plt
    plt.plot(reduce_lr.lr_list)
    plt.show()
