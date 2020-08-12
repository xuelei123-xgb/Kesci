import numpy as np
import tensorflow as tf
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
    confusionMatrix=tf.convert_to_tensor(confusionMatrix)
    def acc_combo(y, y_pred):
        y=tf.argmax(y,axis=1)
        y_pred = tf.argmax(y_pred, axis=1)
        indices=tf.stack([y,y_pred],axis=1)
        # indices=tf.concat([y,y_pred],axis=1)
        scores=tf.gather_nd(confusionMatrix,tf.cast(indices,tf.int32))
        return tf.reduce_mean(scores)
    return acc_combo



