import numpy as np
import rtdl
import scipy.special
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
import torch
import torch.nn.functional as F
import zero
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error

pd_list = []
pf_list = []
bal_list = []
fir_list = []


def classifier_eval(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    print('혼동행렬 : ', cm)
    PD = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    print('PD : ', PD)
    PF = cm[0, 1] / (cm[0, 0] + cm[0, 1])
    print('PF : ', PF)
    balance = 1 - (((0 - PF) * (0 - PF) + (1 - PD) * (1 - PD)) / 2)
    print('balance : ', balance)
    FI = (cm[1, 1] + cm[0, 1]) / (cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1])
    FIR = (PD - FI) / PD
    print('FIR : ', FIR)

    return PD, PF, balance, FIR

device = torch.device('cpu')
# Docs: https://yura52.github.io/zero/0.0.4/reference/api/zero.improve_reproducibility.html
zero.improve_reproducibility(seed=123456)

# !!! NOTE !!! The dataset splits, preprocessing and other details are
# significantly different from those used in the
# paper "Revisiting Deep Learning Models for Tabular Data",
# so the results will be different from the reported in the paper.

dataset = np.loadtxt("C:/Users/sojeong/Desktop/revisiting-models/data/JDT.csv", delimiter=",", skiprows=1, dtype=np.float32)
task_type = 'binclass'

assert task_type in ['binclass', 'multiclass', 'regression']

X_all = dataset[:, :61]
y_all = dataset[:, 61]
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

    # SMOTE(학습 데이터만 진행)
    smote = SMOTE(random_state=42)
    X['train'], y['train'] = smote.fit_resample(X['train'], y['train'])

    # not the best way to preprocess features, but enough for the demonstration
    # 정규화 - MinMaxScaler(), 표준화 - StandardScaler()
    preprocess = MinMaxScaler()
    X = {
        k: torch.tensor(preprocess.fit_transform(v), device=device)
        for k, v in X.items()
    }
    y = {k: torch.tensor(v, device=device) for k, v in y.items()}

    # !!! CRUCIAL for neural networks when solving regression problems !!!
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
        # print("prediction : ", prediction)

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

    # print("Test score before training: ", evaluate("test"))

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

        # val_score = evaluate('val')

        test_score = evaluate('test')  # PD, PF, bal, FIR 반환
        # print(evaluate)
        # print(f'Epoch {epoch:03d} | Validation score: {val_score:.4f} | Test score: {test_score:.4f}', end='')
        # progress.update((-1 if task_type == 'regression' else 1) * val_score)
        # if progress.success:
        #     print(' <<< BEST VALIDATION EPOCH', end='')
        # print()
        # if progress.fail:
        #     break

        PD, PF, bal, FIR = test_score
        pd_list.append(PD)
        pf_list.append(PF)
        bal_list.append(bal)
        fir_list.append(FIR)

print(pd_list)
print(pf_list)
print(bal_list)
print(fir_list)

print('avg_PD: {}'.format((sum(pd_list) / len(pd_list))))
print('avg_PF: {}'.format((sum(pf_list) / len(pf_list))))
print('avg_balance: {}'.format((sum(bal_list) / len(bal_list))))
print('avg_FIR: {}'.format((sum(fir_list) / len(fir_list))))
