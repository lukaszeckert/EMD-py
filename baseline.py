from utils import *
from sklearn.dummy import DummyClassifier
from preprocessing import *
import numpy as np
import pickle
import sys

def main():
    df_train: pd.DataFrame = load_train_data()
    df_eval: pd.DataFrame = load_eval_data()

    df_train = preprocessing(df_train)
    df_eval = preprocessing(df_eval)

    X_train = [0]*len(df_train)
    y_train = df_train["score"]

    X_eval = [0]*len(df_eval)
    y_eval = df_eval["score"]

    model = DummyClassifier("most_frequent")
    model.fit(X_train, y_train)

    print("Baseline eval acc:",np.mean(model.predict(X_eval) == y_eval))
    print("Baseline train acc:", np.mean(model.predict(X_train) == y_train))

    with open("models/baseline.pickle", "wb") as file:
        pickle.dump(model, file)

def test():

    df = pd.read_csv(sys.argv[1])
    with open("models/baseline.pickle", "rb") as file:
        model = pickle.load(file)
    predictions = model.predict_proba(df)
    plot_confusion_matrx(predictions, df["score"])

if __name__ == "__main__":
    if len(sys.argv) == 2:
        test()
    else:
        main()
        test()