import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from catboost import CatBoostClassifier

accuracy_list = []
precision_list = []
recall_list = []
f1_score_list = []


def classifier_eval(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    print('혼동행렬 : ', cm)
    acc = accuracy_score(y_test, y_pred)
    print('accuracy_score : ', acc)
    pre = precision_score(y_test, y_pred)
    print('precision_score : ', pre)
    recall = recall_score(y_test, y_pred)
    print('recall_score : ', recall)
    f1 = f1_score(y_test, y_pred)
    print('f1_score : ', f1)

    return acc, pre, recall, f1


# 모델 선언 예시
model = CatBoostClassifier(n_estimators=500, learning_rate=0.2, max_depth=4, random_state=32)

# 데이터 불러오기
dataset = np.loadtxt("C:/Users/AISELab/Desktop/new_airline-passenger-satisfaction.csv", delimiter=",", skiprows=1, dtype=np.float32)

# X, y 분류
X_all = dataset[:, :22]
y_all = dataset[:, 22]

X = {}
y = {}

# 교차 검증 10번 반복
kf = StratifiedKFold(n_splits=10, shuffle=False)
for train_index, test_index in kf.split(X_all, y_all):
    X_train, X_test = X_all[train_index], X_all[test_index]
    y_train, y_test = y_all[train_index], y_all[test_index]

    # SMOTE(학습데이터만 진행)
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # 정규화 - MinMaxScaler()
    minmax = MinMaxScaler()
    X_all = minmax.fit_transform(X_all)

    # 모델 학습
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 모델 평가
    acc, pre, recall, f1 = classifier_eval(y_test, y_pred)

    accuracy_list.append(acc)
    precision_list.append(pre)
    recall_list.append(recall)
    f1_score_list.append(f1)

print('avg_accuracy : {}'.format((sum(accuracy_list) / len(accuracy_list))))
print('avg_precision : {}'.format((sum(precision_list) / len(precision_list))))
print('avg_recall : {}'.format((sum(recall_list) / len(recall_list))))
print('avg_f1_score : {}'.format((sum(f1_score_list) / len(f1_score_list))))