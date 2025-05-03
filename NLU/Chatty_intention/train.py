import os
import pickle
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


seed = 222
random.seed(seed)
np.random.seed(seed)


def load_data(data_path):
    x, y = [], []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            text, label = line.strip().split(",")
            text = " ".join(list(text.lower()))
            x.append(text)
            y.append(label)

    index = np.arange(len(x))
    np.random.shuffle(index)
    x = [x[i] for i in index]
    y = [y[i] for i in index]
    return x, y


def run(data_path, model_save_path):
    x, y = load_data(data_path)
    label_set = sorted(list(set(y)))

    label2id = {label: idx for idx, label in enumerate(label_set)}
    id2label = {idx: label for idx, label in enumerate(label_set)}

    y = [label2id[i] for i in y]
    label_names = sorted(label2id.items(), key=lambda kv: kv[1], reverse=False)
    target_names = [i[0] for i in label_names]
    labels = [i[1] for i in label_names]

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.15, random_state=42)
    # analyzer 表示按**字符（character）**级别进行分词，而不是默认的按单词。
    vec = TfidfVectorizer(ngram_range=(1, 3), min_df=0.0, max_df=0.9, analyzer="char")
    train_x = vec.fit_transform(train_x)
    print(train_x.shape)
    test_x = vec.transform(test_x)
    print(test_x.shape)

    # LR
    # C=8：该参数是正则化强度的倒数。较小的值表示更强的正则化。值为 8 表示适度的正则化，有助于防止过拟合，通过惩罚较大的系数来实现
    # n_jobs=4：指定用于并行处理的CPU核数
    # max_iter=400：模型训练的最大迭代次数
    # multi_class='ovr'：指定多分类问题的处理方式。ovr（one-vs-rest）表示对每个类分别训练一个二分类器，
    LR = LogisticRegression(C=8, n_jobs=4, max_iter=400, random_state=122)
    LR.fit(train_x, train_y)
    pred = LR.predict(test_x)
    print(pred)
    print(classification_report(test_y, pred, target_names=target_names))
    print(confusion_matrix(test_y, pred, labels=labels))
    print("-------------------------------------------------------------------------")

    # gbdt
    gbdt = GradientBoostingClassifier(n_estimators=450, learning_rate=0.01, max_depth=8, random_state=24)
    gbdt.fit(train_x, train_y)
    pred = gbdt.predict(test_x)
    print(pred)
    print(classification_report(test_y, pred, target_names=target_names))
    print(confusion_matrix(test_y, pred, labels=labels))
    print("-------------------------------------------------------------------------")

    # 融合
    pred_prob1 = LR.predict_proba(test_x)
    pred_prob2 = gbdt.predict_proba(test_x)
    pred = np.argmax((pred_prob1+pred_prob2)/2, axis=1)
    print(pred)
    print(classification_report(test_y, pred, target_names=target_names))
    print(confusion_matrix(test_y, pred, labels=labels))

    pickle.dump(id2label, open(os.path.join(model_save_path,'id2label.pkl'),'wb'))
    pickle.dump(vec, open(os.path.join(model_save_path,'vec.pkl'),'wb'))
    pickle.dump(LR, open(os.path.join(model_save_path,'LR.pkl'),'wb'))
    pickle.dump(gbdt, open(os.path.join(model_save_path,'gbdt.pkl'),'wb'))



if __name__ == '__main__':
    # x, y = load_data("./data/train.txt")
    # print(x)
    # print(y)
    run("./data/train.txt", "./model_file")
