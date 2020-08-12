from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.signal import resample
from sklearn.model_selection import StratifiedKFold
import os
from xwbank_dataset.transfoms import  *
from imblearn.over_sampling import  RandomOverSampler,SMOTE
CURPATH=os.path.abspath(os.path.dirname(__file__))
import random
import xwbank_dataset.utils as ut
def one_hot(labels, num_classes,label_smooth=True,smooth_factor=0.9):
    labels = np.squeeze(labels).astype(np.int)
    if labels.ndim==0:
        arr = np.zeros(num_classes)
        arr[labels]=1
        return arr
    batch_size = labels.shape[0]
    idxs = np.arange(0, batch_size, 1)
    arr = np.zeros([batch_size, num_classes])
    arr[idxs, labels] = 1
    if not label_smooth:
        return arr
    else:
        noise=(1-smooth_factor)/float(num_classes-1)
        smooth_label=arr*smooth_factor+(1-arr)*noise
        return smooth_label

mean=[ 8.03889039e-03, -6.41381949e-02,  2.37856977e-02,  8.64949391e-01,
       2.80964889e+00,  7.83041714e+00,  6.44853358e-01,  9.78580749e+00,]
std=[0.6120893,  0.53693888, 0.7116134,  3.22046385, 3.01195336, 2.61300056,0.87194132, 0.68427254,]



def standardization(x,t):
    # x1 = X.transpose(0, 1, 3, 2)
    def get(data1,data2):
        data1 = data1.reshape(-1, data1.shape[-1])
        data2 = data2.reshape(-1, data2.shape[-1])
        data=np.concatenate([data1,data2],axis=0)
        mu = np.mean(data, axis=0)
        sigma = np.std(data, axis=0)
        return mu,sigma
    mean,std=get(x,t)
    x = ((x - mean) / (std))
    t = ((t - mean) / (std))
    return x,t

def mixture(x,noise,ratio=0.2): # x为其中一个样本  noise为其中一个样本
	assert ratio>=0 and ratio < 1
	assert x.shape==noise.shape
	mask=np.random.random(x.shape) # 生成 x.shape个 0-1 之间的随机数
	mask=np.where(mask<ratio,1,0) # 小于ratio的位置为1 否则为0
	return x*(1-mask)+noise*mask




def jitter(x, snr_db=25):
    """
    根据信噪比添加噪声
    :param x:
    :param snr_db:
    :return:
    """
    # 随机选择信噪比
    if isinstance(snr_db, list):
        snr_db_low = snr_db[0]
        snr_db_up = snr_db[1]
    else:
        snr_db_low = snr_db
        snr_db_up = 45
    snr_db = np.random.randint(snr_db_low, snr_db_up, (1,))[0]

    snr = 10 ** (snr_db / 10)
    Xp = np.sum(x ** 2, axis=0, keepdims=True) / x.shape[0]  # 计算信号功率
    Np = Xp / snr  # 计算噪声功率

    n = np.random.normal(size=x.shape, scale=np.sqrt(Np), loc=0.0)  # 计算噪声
    xn = x + n
    return xn


class XWDataset(object):
    def __init__(self,**kwargs):

        self.num_folds=kwargs.get("num_folds",5)
        self.channel_first=kwargs.get("channel_first",False)
        self.num_classes=kwargs.get("num_classes",19)
        self.use_ohlabel = kwargs.get("use_ohlabel", True)
        self.num_branch = kwargs.get("num_branch", 1)


        self.alignment_method=kwargs.get("alignment_method","fft_sample")
        assert self.alignment_method in ["fft_sample","padding"]
        print("using [{}] to align the time  series".format(self.alignment_method))

        #增加参数 with_nosie,
        self.with_nosie=kwargs.get("with_nosie",False)
        self.noise_SNR_db=kwargs.get("noise_SNR_db",25)
        if self.with_nosie:
            print("添加随机噪声,SNR_db:{}".format(self.noise_SNR_db))

        #增加参数 mixmure,
        self.with_mixmure=kwargs.get("with_mixmure",False)
        self.mixmure_ratio=kwargs.get("mixmure_ratio",0.2)
        if self.with_mixmure:
            print("使用mixmure数据增强,ratio:{}".format(self.mixmure_ratio))


        self.with_timeshift = kwargs.get("with_timeshift", False)
        if self.with_timeshift:
            print("使用with_timeshift")

            # 增加参数 imblance sample,
        self.with_sample = kwargs.get("with_sample", False)
        self.sample_method = kwargs.get("sample_method", "ROS")
        assert self.sample_method in ["ROS","SMOTE"]
        if self.with_sample:
            print("使用采样平衡样本数量,采样方式:{}".format(self.sample_method))
        self.load_data()

    def __len__(self):
        return self.train_X.shape[0]

    def __getitem__(self, index):
        '''Generate one batch of data'''
        # Generate indexes of the batch
        ###添加mixmure功能  wslsdx


        x = self.train_X[int(index)]
        if self.with_mixmure and random.random() > 0.5:
            # 随机选择一个样本作为噪声
            noise_idx = random.choice(list(range(len(self))))
            noise = self.train_X[noise_idx] # 一个样本的 index
            x = mixture(x, noise, ratio=self.mixmure_ratio)
        if self.with_timeshift and random.random()<0.3:# 进行时间移动
            x=time_shift_spectrogram(x,axis=0) # 将数据向右移动移动若干个单位长度 环形移动，最后的数据补到最前面

        y = self.train_Y[int(index)]
        if self.use_ohlabel:
            y=one_hot(y, self.num_classes)
        return x,y





    @property
    def dim(self):
        return tuple(self.X.shape[1:])


    def load_data(self):
        train = pd.read_csv(os.path.join(CURPATH, "sensor_train.csv"))
        test = pd.read_csv(os.path.join(CURPATH, "sensor_test.csv"))
        y = np.array(train.groupby("fragment_id")["behavior_id"].min())
        fragment_ids = np.array(train.groupby("fragment_id")["fragment_id"].min())
        # print(y)
        train['mod'] = (train.acc_x ** 2 + train.acc_y ** 2 + train.acc_z ** 2) ** .5
        train['modg'] = (train.acc_xg ** 2 + train.acc_yg ** 2 + train.acc_zg ** 2) ** .5
        test['mod'] = (test.acc_x ** 2 + test.acc_y ** 2 + test.acc_z ** 2) ** .5
        test['modg'] = (test.acc_xg ** 2 + test.acc_yg ** 2 + test.acc_zg ** 2) ** .5

        HEAD = list(train.drop(['time_point', 'behavior_id'], axis=1)) + ["behavior_id"]
        # print(HEAD)
        x = np.zeros((7292, 1,60, 8))
        t = np.zeros((7500, 1,60, 8))
        for i in tqdm(range(7292)):
            tmp = train[train.fragment_id == i][:60]
            if self.alignment_method=="fft_sample":
                data=resample(tmp.drop(['fragment_id', 'time_point', 'behavior_id'],
                                              axis=1), 60, np.array(tmp.time_point))[0]
                x[i, 0, :, :] = data
            elif self.alignment_method=="padding":
                data=np.zeros((60,8))
                left=(60-tmp.shape[0])//2
                data[left:left+tmp.shape[0],:]=tmp.drop(['fragment_id', 'time_point', 'behavior_id'],axis=1)
                # print(data)
                x[i, 0, :, :] = data
        for i in tqdm(range(7500)):
            tmp = test[test.fragment_id == i][:60]
            if self.alignment_method=="fft_sample":
                data=resample(tmp.drop(['fragment_id', 'time_point'],
                                              axis=1), 60, np.array(tmp.time_point))[0]
                t[i, 0, :, :] = data
            elif self.alignment_method=="padding":
                data=np.zeros((60,8))
                left=(60-tmp.shape[0])//2
                data[left:left+tmp.shape[0],:]=tmp.drop(['fragment_id', 'time_point'],axis=1)
                t[i, 0, :, :] = data
        self.Y= np.array(train.groupby("fragment_id")["behavior_id"].min())
        self.X,self.T=standardization(x,t)
        # self.X=standardization(x)
        # self.T=standardization(t)
        self.X,self.T=self.standardization(x,t)
        if not self.channel_first:
            self.X,self.T= self.X.transpose(0,2,3,1),self.T.transpose(0,2,3,1)
        self.stratifiedKFold(self.num_folds)

    def stratifiedKFold(self,fold=5):
        kfold = StratifiedKFold(fold, shuffle=True)
        self.train_valid_idxs=[ (train_idx,valid_idx) for train_idx,valid_idx in kfold.split(self.X,self.Y) ]



    def enchance_train_data(self):
        """
        根据fold索引读训练数据，并进行数据扩充
        :return:
        """
        train_X,train_Y= self.X[self.train_idx],self.Y[self.train_idx]
        train_X_noise=jitter(train_X,snr_db=self.noise_SNR_db)
        train_X1, train_Y1 = self.sample(train_X, train_Y)  # 类别采样
        train_X2, train_Y2 = self.sample(train_X_noise, train_Y)  # 类别采样
        train_XX,train_YY=np.concatenate([train_X1,train_X2],axis=0),np.concatenate([train_Y1,train_Y2],axis=0)
        print("enchance training data...")
        self.train_X,self.train_Y=train_XX,train_YY

    def get_valid_data(self,index):
        """
        :param index:
        :return:  重新划分训练集和验证集 , 并返回验证集数据
        """
        assert index < self.num_folds
        self.train_idx,self.valid_idx= self.train_valid_idxs[index]
        self.enchance_train_data()
        self.valid_X,self.valid_Y=self.X[self.valid_idx],self.Y[self.valid_idx]

        if self.use_ohlabel:
            self.valid_Y=one_hot(self.valid_Y,self.num_classes)

        return self.valid_X,self.valid_Y
    def get_test_data(self):
        return self.T
        # test_data = [self.T[i] for i in range(self.T.shape[0])]
        # return  test_data

    def standardization(self,x,t):
        def cul(x1):
            x2 = x1.reshape(-1, x1.shape[-1])
            mean = np.mean(x2, axis=0)
            std = np.std(x2, axis=0)
            return mean,std
        mean,std=cul(np.concatenate([x, t], axis=0))
        x = ((x - mean) / (std))
        t = ((t - mean) / (std))
        return x,t

    def sample(self,X,Y):
        ##################over_sample########################
        if self.with_sample:
            shape=X.shape
            X=X.reshape((shape[0],-1))
            Y=Y
            # 定义模型，random_state相当于随机数种子的作用
            if self.sample_method == "ROS":
                sampler=RandomOverSampler(random_state=0)
            elif self.sample_method=="SMOTE":
                sampler=SMOTE(random_state=0)
            X, Y = sampler.fit_sample(X, Y)
            X = X.reshape((-1, shape[1], shape[2], shape[3]))
        return X,Y

