import numpy as np
import pandas as pd
import rtdl
import scipy.special
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch
import torch.nn.functional as F
import zero
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

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


device = torch.device('cpu')
# Docs: https://yura52.github.io/zero/0.0.4/reference/api/zero.improve_reproducibility.html
zero.improve_reproducibility(seed=123456)


# 데이터 DataFrame으로 불러오기
dataset = pd.read_csv('C:/Users/AISELab/Desktop/airline-passenger-satisfaction.csv', encoding='utf-8')

# 데이터 형태 확인
print(dataset.info)

# 결측치 확인
print(dataset.isna().sum())

# 결측치 처리(평균으로 대체)
dataset['Arrival Delay in Minutes'] = dataset['Arrival Delay in Minutes'].fillna(dataset['Arrival Delay in Minutes'].mean())
print(dataset.isna().sum())

# 레이블 인코딩
le = LabelEncoder()
for col in ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']:
    dataset[col] = le.fit_transform(dataset[col])
print(dataset)

# 필요없는 feature 제거
dataset = dataset.drop(['Unnamed: 0', 'id'], axis=1)

# csv로 내보내기
dataset.to_csv('C:/Users/AISELab/Desktop/new_train.csv', index=False)

# 데이터 불러오기
dataset = np.loadtxt("C:/Users/AISELab/Desktop/new_airline-passenger-satisfaction.csv", delimiter=",", skiprows=1, dtype=np.float32)

# 이진 분류 명시
task_type = 'binclass'

assert task_type in ['binclass', 'multiclass', 'regression']

# X, y 분류
X_all = dataset[:, :22]
y_all = dataset[:, 22]

if task_type != 'regression':
    y_all = LabelEncoder().fit_transform(y_all).astype('int64')
n_classes = int(max(y_all)) + 1 if task_type == 'multiclass' else None
# n_classes = 2

X = {}
y = {}

# 교차 검증 10번 반복
kf = StratifiedKFold(n_splits=10, shuffle=False)
for train_index, test_index in kf.split(X_all, y_all):
    X['train'], X['test'] = X_all[train_index], X_all[test_index]
    y['train'], y['test'] = y_all[train_index], y_all[test_index]

    # SMOTE(학습데이터만 진행)
    smote = SMOTE(random_state=42)
    X['train'], y['train'] = smote.fit_resample(X['train'], y['train'])

    # 정규화 - MinMaxScaler()
    preprocess = MinMaxScaler()
    X = {
        k: torch.tensor(preprocess.fit_transform(v), device=device)
        for k, v in X.items()
    }
    y = {k: torch.tensor(v, device=device) for k, v in y.items()}

    if task_type == 'regression':
        y_mean = y['train'].mean().item()
        y_std = y['train'].std().item()
        y = {k: (v - y_mean) / y_std for k, v in y.items()}
    else:
        y_std = y_mean = None

    if task_type != 'multiclass':
        y = {k: v.float() for k, v in y.items()}

    d_out = n_classes or 1

    model = rtdl.FTTransformer.make_default(
        n_num_features=X_all.shape[1],
        cat_cardinalities=None,
        n_blocks=1,
        last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
        d_out=d_out,
    )
    model.to(device)
    optimizer = (
        model.make_default_optimizer()
        if isinstance(model, rtdl.FTTransformer)
        else torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    )
    loss_fn = (
        F.binary_cross_entropy_with_logits
        if task_type == 'binclass'
        else F.cross_entropy
        if task_type == 'multiclass'
        else F.mse_loss
    )

    def apply_model(x_num, x_cat=None):
        if isinstance(model, rtdl.FTTransformer):
            return model(x_num, x_cat)  # X['test']에 모델 적용한 뒤 예측한 y 반환?
        elif isinstance(model, (rtdl.MLP, rtdl.ResNet)):
            assert x_cat is None
            return model(x_num)
        else:
            raise NotImplementedError(
                f'Looks like you are using a custom model: {type(model)}.'
                ' Then you have to implement this branch first.'
            )

    @torch.no_grad()
    def evaluate(part):
        model.eval()
        prediction = []
        for batch in zero.iter_batches(X[part], 1024):
            prediction.append(apply_model(batch))
        prediction = torch.cat(prediction).squeeze(1).cpu().numpy()
        target = y[part].cpu().numpy()

        if task_type == 'binclass':
            prediction = np.round(scipy.special.expit(prediction))  # 시그모이드 함수, 음수 양수 기준으로 0과 1 분류?
            # print("round 후 prediction : ", prediction)
            score = classifier_eval(target, prediction)
        elif task_type == 'multiclass':
            prediction = prediction.argmax(1)
            score = accuracy_score(target, prediction)
        else:
            assert task_type == 'regression'
            score = mean_squared_error(target, prediction) ** 0.5 * y_std
        return score

    # Create a dataloader for batches of indices
    # Docs: https://yura52.github.io/zero/reference/api/zero.data.IndexLoader.html
    batch_size = 32
    train_loader = zero.data.IndexLoader(len(X['train']), batch_size, device=device)

    # Create a progress tracker for early stopping
    # Docs: https://yura52.github.io/zero/reference/api/zero.ProgressTracker.html
    progress = zero.ProgressTracker(patience=100)

    # 학습
    n_epochs = 50
    report_frequency = len(X['train']) // batch_size // 5
    for epoch in range(1, n_epochs + 1):
        for iteration, batch_idx in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            x_batch = X['train'][batch_idx]
            y_batch = y['train'][batch_idx]
            loss = loss_fn(apply_model(x_batch).squeeze(1), y_batch)
            loss.backward()
            optimizer.step()
            if iteration % report_frequency == 0:
                print(f'(epoch) {epoch} (batch) {iteration} (loss) {loss.item():.4f}')

        test_score = evaluate('test')

        acc, pre, recall, f1 = test_score
        accuracy_list.append(acc)
        precision_list.append(pre)
        recall_list.append(recall)
        f1_score_list.append(f1)

print(accuracy_list)
print(precision_list)
print(recall_list)
print(f1_score_list)

print('avg_accuracy : {}'.format((sum(accuracy_list) / len(accuracy_list))))
print('avg_precision : {}'.format((sum(precision_list) / len(precision_list))))
print('avg_recall : {}'.format((sum(recall_list) / len(recall_list))))
print('avg_f1_score : {}'.format((sum(f1_score_list) / len(f1_score_list))))