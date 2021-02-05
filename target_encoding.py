from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC

from utils import *
from sklearn.dummy import DummyClassifier
from preprocessing import *
from sklearn.base import TransformerMixin
import numpy as np
import pickle
import category_encoders as ce
import sys






def main():
    df_train: pd.DataFrame = load_train_data()
    df_eval: pd.DataFrame = load_eval_data()

    df_train = preprocessing(df_train)
    df_eval = preprocessing(df_eval)

    X_train = df_train
    y_train = df_train["score"]

    X_eval = df_eval
    y_eval = df_eval["score"]

    model = Pipeline([
        ("features", ColumnTransformer([
            ("target_enc", ce.TargetEncoder(cols=["reviewerID", "asin"]), ["reviewerID","asin"]),
        ])),
        ("model", SGDClassifier(loss='modified_huber'))
         ])

    # model = ce.TargetEncoder(cols=["reviewerID"])

    model.fit(X_train, y_train)

    print("Target encoding eval acc:", np.mean(model.predict(X_eval) == y_eval))
    print("Target encoding acc:", np.mean(model.predict(X_train) == y_train))

    with open("models/target_encoding.pickle", "wb") as file:
        pickle.dump(model, file)



def test():
    df = pd.read_csv(sys.argv[1])
    with open("models/target_encoding.pickle", "rb") as file:
        model = pickle.load(file)

    predictions = model.predict_proba(df)
    plot_confusion_matrx(predictions, df["score"])


if __name__ == "__main__":
    if len(sys.argv) == 2:
        test()
    else:
        main()
