import numpy as np
mapping = {0: 'A_0', 1: 'A_1', 2: 'A_2', 3: 'A_3',
           4: 'D_4', 5: 'A_5', 6: 'B_1', 7: 'B_5',
           8: 'B_2', 9: 'B_3', 10: 'B_0', 11: 'A_6',
           12: 'C_1', 13: 'C_3', 14: 'C_0', 15: 'B_6',
           16: 'C_2', 17: 'C_5', 18: 'C_6'}
mapping1 = {"A": 7, "B": 8, "C": 9, "D": 10}
transfer_matrix7=np.zeros((19,7))
transfer_matrix4=np.zeros((19,4))
for key,val in mapping.items():
    alpha,num7=val.split("_")
    num4=mapping1[alpha]-7
    transfer_matrix7[key,int(num7)]=1
    transfer_matrix4[key,int(num4)]=1

def one_hot(labels, num_classes):
    labels = np.squeeze(labels).astype(np.int)
    if labels.ndim==0:
        arr = np.zeros(num_classes)
        arr[labels]=1
        return arr
    batch_size = labels.shape[0]
    idxs = np.arange(0, batch_size, 1)
    arr = np.zeros([batch_size, num_classes])
    arr[idxs, labels] = 1
    return arr

def to_categorical(cls,num_classes):
    if num_classes==19:
        return cls
    if num_classes==7:
        _,num=mapping[cls].split("_")
        return int(num)
    if num_classes==4:
        alpha,_=mapping[cls].split("_")
        num=mapping1[alpha]-7
        return int(num)


def get_acc_combo():
    def combo(y, y_pred):
        # 数值ID与行为编码的对应关系
        mapping = {0: 'A_0', 1: 'A_1', 2: 'A_2', 3: 'A_3',
            4: 'D_4', 5: 'A_5', 6: 'B_1',7: 'B_5',
            8: 'B_2', 9: 'B_3', 10: 'B_0', 11: 'A_6',
            12: 'C_1', 13: 'C_3', 14: 'C_0', 15: 'B_6',
            16: 'C_2', 17: 'C_5', 18: 'C_6'}
        # 将行为ID转为编码

        code_y, code_y_pred = mapping[int(y)], mapping[int(y_pred)]
        if code_y == code_y_pred: #编码完全相同得分1.0
            return 1.0
        elif code_y.split("_")[0] == code_y_pred.split("_")[0]: #编码仅字母部分相同得分1.0/7
            return 1.0/7
        elif code_y.split("_")[1] == code_y_pred.split("_")[1]: #编码仅数字部分相同得分1.0/3
            return 1.0/3
        else:
            return 0.0
    confusionMatrix=np.zeros((19,19))
    for i in range(19):
        for j in range(19):
            confusionMatrix[i,j]=combo(i,j)
    def acc_combo(y, y_pred):
        if y.ndim == 2:
            y=np.argmax(y,axis=1)
        if y_pred.ndim==2:
            y_pred = y_pred[:, :19]
            y_pred = np.argmax(y_pred, axis=1)
        scores=confusionMatrix[y.astype(np.int),y_pred.astype(np.int)]
        return np.mean(scores)
    return acc_combo

def get_acc_func():
    confusionMatrix=np.zeros((19,19))
    for i in range(19):
            confusionMatrix[i,i]=1
    def acc_func(y, y_pred):
        y1 ,y2,y3= y[:, :19],y[:, 19:26],y[:, 26:]
        y_pred1, y_pred2, y_pred3 = y_pred[:, :19], y_pred[:, 19:26], y_pred[:, 26:]
        scores1=confusionMatrix[np.argmax(y1,axis=1).astype(np.int),np.argmax(y_pred1,axis=1).astype(np.int)]
        scores2 = confusionMatrix[np.argmax(y2, axis=1).astype(np.int), np.argmax(y_pred2, axis=1).astype(np.int)]
        scores3 = confusionMatrix[np.argmax(y3, axis=1).astype(np.int), np.argmax(y_pred3, axis=1).astype(np.int)]
        return scores1.mean(),scores2.mean(),scores3.mean()
    return acc_func

def get_match_acc_func():
    confusionMatrix7=np.zeros((7,7))
    for i in range(7):
            confusionMatrix7[i,i]=1
    confusionMatrix4=np.zeros((4,4))
    for i in range(4):
            confusionMatrix4[i,i]=1

    def match_acc_func(y, y_pred):
        assert y_pred.ndim==2
        y_pred1=np.dot(y_pred[:, :19],transfer_matrix7).argmax(axis=1).astype(np.int)
        y_pred2 = np.dot(y_pred[:, :19], transfer_matrix4).argmax(axis=1).astype(np.int)
        y_pred11 = (y_pred[:, 19:26]).argmax(axis=1).astype(np.int)
        y_pred22 = (y_pred[:, 26:30]).argmax(axis=1).astype(np.int)
        scores7=confusionMatrix7[y_pred1,y_pred11]
        scores4 = confusionMatrix4[y_pred2, y_pred22]
        return np.mean(scores4*scores7),np.mean(scores7),np.mean(scores4)
    return match_acc_func


if __name__ == '__main__':
    y_pred=np.diag((5,5,10,6))

    # dat=np.random.random((19,200))
    # dat1=np.random.random((7,19))
    # print(np.dot(y_pred,dat))