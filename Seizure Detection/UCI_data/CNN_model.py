import time
import numpy as np
import pandas as pd
from keras.optimizers import Adam
from keras.layers import Dense, Dropout, MaxPooling1D, Conv1D, Flatten
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import warnings
from collections import Counter

warnings.filterwarnings("ignore")  # 忽略警告消息

#
seed=0
# 数据加载
datapath = "UCIdata.csv"
data = pd.read_csv(datapath, header=0, index_col=0)

# 数据预处理
data["y"] = data["y"].apply(lambda x: 1 if x == 1 else 0)
x_data = data.iloc[:, :178]
y_data = data["y"]

# 数据归一化
scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)

# 划分数据为训练集（80%）和测试集（20%）
x_train_val, x_test, y_train_val, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=seed)

# 使用 SMOTE 对训练数据进行过采样
smote = SMOTE(random_state=seed)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train_val, y_train_val)

# 检查过采样后每个类别的样本数量
print("过采样前类别分布:", Counter(y_train_val))
print("过采样后类别分布:", Counter(y_train_resampled))


# 创建模型函数
def create_model():
    model = Sequential()
    model.add(Conv1D(20, 5, activation='relu', input_shape=(178, 1)))
    model.add(Dropout(0.5))
    model.add(Conv1D(20, 5, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv1D(20, 5, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# 设置 KFold 交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=seed)
test_scores = []  # 保存测试集的准确率
test_recalls = []  # 保存测试集的灵敏度
test_specificities = []  # 保存测试集的特异性
val_scores = []  # 保存验证集的准确率
val_recalls = []  # 保存验证集的灵敏度
val_specificities = []  # 保存验证集的特异性
best_val_recall = 0
best_model = None
best_model_test_results = None  # 用于保存最佳模型的测试集结果

# 早停
early_stopping = EarlyStopping(monitor='val_loss', patience=16, verbose=1)

# 进行交叉验证
for fold, (train_index, val_index) in enumerate(kf.split(x_train_resampled)):
    print(f'正在训练第 {fold + 1} 折...')
    X_train, X_val = x_train_resampled[train_index], x_train_resampled[val_index]
    y_train, y_val = y_train_resampled[train_index], y_train_resampled[val_index]

    # 将数据形状调整为 (样本数, 特征数, 通道数)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

    # 创建和训练模型
    model = create_model()
    history = model.fit(X_train, y_train, epochs=200, batch_size=256, verbose=2,
                        validation_data=(X_val, y_val), callbacks=[early_stopping])

    # 在验证集上评估模型
    y_val_pred = (model.predict(X_val) > 0.5).astype("int32")
    val_score = accuracy_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    tn_val, fp_val, fn_val, tp_val = confusion_matrix(y_val, y_val_pred).ravel()
    val_specificity = tn_val / (tn_val + fp_val) if (tn_val + fp_val) > 0 else 0

    # 保存当前折验证集的指标
    val_scores.append(val_score)
    val_recalls.append(val_recall)
    val_specificities.append(val_specificity)

    print(f'第 {fold + 1} 折的验证集准确率: {val_score}')
    print(f'第 {fold + 1} 折的验证集灵敏度: {val_recall}')
    print(f'第 {fold + 1} 折的验证集特异性: {val_specificity}')

    # 保存当前折模型在验证集上的指标
    if val_recall > best_val_recall:
        best_val_recall = val_recall
        best_model = model
        # 评估最佳模型在测试集上的结果
        y_best_model_test_pred = (model.predict(x_test.reshape((x_test.shape[0], x_test.shape[1], 1))) > 0.5).astype(
            "int32")
        best_model_test_score = accuracy_score(y_test, y_best_model_test_pred)
        best_model_test_recall = recall_score(y_test, y_best_model_test_pred)
        tn_best, fp_best, fn_best, tp_best = confusion_matrix(y_test, y_best_model_test_pred).ravel()
        best_model_test_specificity = tn_best / (tn_best + fp_best) if (tn_best + fp_best) > 0 else 0
        best_model_test_results = (best_model_test_score, best_model_test_recall, best_model_test_specificity)


    # 使用当前折的模型评估全测试集
    y_test_pred = (model.predict(x_test.reshape((x_test.shape[0], x_test.shape[1], 1))) > 0.5).astype("int32")

    # 计算测试集的准确率、灵敏度和特异性
    test_score = accuracy_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_test_pred).ravel()
    test_specificity = tn_test / (tn_test + fp_test) if (tn_test + fp_test) > 0 else 0

    # 保存当前模型的测试集指标
    test_scores.append(test_score)
    test_recalls.append(test_recall)
    test_specificities.append(test_specificity)

    print(f'第 {fold + 1} 折的测试集准确率: {test_score}')
    print(f'第 {fold + 1} 折的测试集灵敏度: {test_recall}')
    print(f'第 {fold + 1} 折的测试集特异性: {test_specificity}')

# 计算平均测试集和验证集指标
average_val_score = np.mean(val_scores)
average_val_recall = np.mean(val_recalls)
average_val_specificity = np.mean(val_specificities)
average_test_score = np.mean(test_scores)
average_test_recall = np.mean(test_recalls)
average_test_specificity = np.mean(test_specificities)

print(f'验证集的平均准确率: {average_val_score}')
print(f'验证集的平均灵敏度: {average_val_recall}')
print(f'验证集的平均特异性: {average_val_specificity}')
print(f'测试集的平均准确率: {average_test_score}')
print(f'测试集的平均灵敏度: {average_test_recall}')
print(f'测试集的平均特异性: {average_test_specificity}')

# 输出每折的测试集和验证集指标
print(f'每折的验证集准确率: {val_scores}')
print(f'每折的验证集灵敏度: {val_recalls}')
print(f'每折的验证集特异性: {val_specificities}')
print(f'每折的测试集准确率: {test_scores}')
print(f'每折的测试集灵敏度: {test_recalls}')
print(f'每折的测试集特异性: {test_specificities}')

# 输出验证集上表现最好模型的测试集结果
if best_model_test_results:
    print(f'最佳模型在测试集上的准确率: {best_model_test_results[0]}')
    print(f'最佳模型在测试集上的灵敏度: {best_model_test_results[1]}')
    print(f'最佳模型在测试集上的特异性: {best_model_test_results[2]}')
