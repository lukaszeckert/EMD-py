import pandas as pd
import os
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

def load_train_data(sub=False):
    if not os.path.exists("data/train.csv"):
       create_datasets()
    if sub:
        return pd.read_csv("data/train_under.csv")
    return pd.read_csv("data/train.csv")


def load_eval_data(sub=False):
    if not os.path.exists("data/eval.csv"):
        create_datasets()
    if sub:
        return pd.read_csv("data/train_under.csv")
    return pd.read_csv("data/eval.csv")

def under_sample(df):
    res = []

    for c in range(1,6):
        cur = df[df["score"] == c]
        res.append(cur.sample(33737, random_state=c))
    return pd.concat(res)

def create_train_eval(df):

    df.sort_values("unixReviewTime", inplace=True)
    threshold = df["unixReviewTime"].quantile(0.8)
    df_train = df[df["unixReviewTime"] < threshold]
    df_eval = df[df["unixReviewTime"] >= threshold]
    return df_train, df_eval

def create_datasets():
    df = pd.read_csv("data/reviews_train.csv")
    df_train, df_eval = create_train_eval(df)
    df_train.to_csv("data/train.csv", index=None)
    df_eval.to_csv("data/eval.csv", index=None)


    df_train, df_eval = create_train_eval(under_sample(df))
    df_train.to_csv("data/train_under.csv", index=None)
    df_eval.to_csv("data/eval_under.csv", index=None)


def plot_confusion_matrx(pred_prob, targ):
    pred = np.argmax(pred_prob, -1)+1

    print("ACC", np.mean(targ == pred))
    print("L1", np.mean(np.abs(targ - pred)))
    print("E[L1]", np.mean(np.abs(targ - pred_prob.dot([1,2,3,4,5]))))
    cm = metrics.confusion_matrix(targ, pred, labels=[1,2,3,4,5])
    cm = cm/np.sum(cm)
    sns.heatmap(cm, linewidth=0.5,  annot=True, xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5])
    plt.show()

def clean_text(text):
    res = []
    for word in text:
        word = re.sub("[^a-z]", '', word)
        if len(word) > 0:
            res.append(word)
    if len(res) == 0:
        res = ['a']
    return res
