import numpy as np
import pandas as pd
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from my_callbacks import  CosineAnnealing
from models import *
from xwbank_dataset.dataset import DataGenerator
from xwbank_dataset.xwdata import XWDataset
# DATA
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import os
from utils import  get_acc_combo
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config.gpu_options.per_process_gpu_memory_fraction = 0.9
data_dir="./dataset/"
sub=pd.read_csv("sub.csv")
NUM_CLASSES=19
EPOCH=250
BATCH_SIZE=512
xwdata=XWDataset(
    with_mixmure=True, mixmure_ratio=0.2,
    with_timeshift=True,
    with_nosie=True, noise_SNR_db=[5, 10],
    with_sample=True, sample_method="ROS",
)
proba_t = np.zeros((len(xwdata.get_test_data()), NUM_CLASSES))


folds=5
# train_data.stratifiedKFold(folds)
for fold in range(folds):
    #划分训练集和验证集 并返回验证集数据


    model = Model_beta1(input_shape=(60, 8, 1), classes=19)
    OPTIM = SGD(lr=0.1, momentum=0.9)
    # OPTIM=optimizers.Adam()
    def get_lr_metric(optimizer):

        def lr(y_true, y_pred):
            return optimizer.lr

        return lr
    lr_metric = get_lr_metric(OPTIM)

    model.compile(loss='categorical_crossentropy',
                  optimizer=OPTIM,
                  metrics=['acc',get_acc_combo(),lr_metric])
    model.summary()

    valid_data=xwdata.get_valid_data(fold)
    train_generator=DataGenerator(xwdata,batch_size=BATCH_SIZE,classes=19)
    plateau = ReduceLROnPlateau(monitor="val_acc_combo",
                                verbose=0,
                                mode='max',
                                factor=0.5,
                                patience=12)
    reduce_lr = CosineAnnealing( epochs=[20, 40, 80, 160, 250],
                                 eta_max=0.1, eta_min=0.000005,
                                iteration_of_epoch=len(train_generator),
                                 verbose=1, )
    early_stopping = EarlyStopping(monitor='val_acc_combo',
                                   verbose=1,
                                   mode='max',
                                   patience=20)
    checkpoint = ModelCheckpoint(f'fold{fold}.h5',
                                 monitor='val_acc_combo',
                                 verbose=0,
                                 mode='max',
                                 save_best_only=True)

    history = model.fit_generator(train_generator, epochs=EPOCH,
                                   use_multiprocessing=False,
                                  validation_data=valid_data,
                                  callbacks=[ plateau,checkpoint])
    model.load_weights(f'fold{fold}.h5')
    proba = model.predict(xwdata.get_test_data(), verbose=0, batch_size=BATCH_SIZE)
    df=pd.DataFrame(proba)
    df.to_csv("prob_fold{}.csv".format(fold))
    proba_t += proba/ 5.
    sub.behavior_id = np.argmax(proba_t, axis=1)
    sub.to_csv('submit.csv', index=False)