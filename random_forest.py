import nltk
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

nltk.download('stopwords')
nltk.download('punkt')
from utils import *
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier

from preprocessing import *
import numpy as np
import pickle
from sklearn.model_selection import TimeSeriesSplit
import sys
from nltk.stem.snowball import SnowballStemmer


def tokenize(text):

    tokens = nltk.word_tokenize(text.lower())
    stems = []
    for item in tokens:
        stems.append(SnowballStemmer("english", ignore_stopwords=True).stem(item))

    return clean_text(stems)
def preprocess(text):
    if text is np.nan:
        return "a"
    return text

def main():
    for column in ["summary", "reviewText"]:
        df_train: pd.DataFrame = load_train_data()
        df_eval: pd.DataFrame = load_eval_data()

        df_train = df_train[:50000]
        df_train = preprocessing(df_train)
        df_eval = preprocessing(df_eval)
        X_train = df_train
        y_train = df_train["score"]

        X_eval = df_eval
        y_eval = df_eval["score"]

        model = Pipeline(
            [
                ("tfidf", ColumnTransformer([
                    ("tfidf",
                     TfidfVectorizer(min_df=10, max_df=0.3,preprocessor=preprocess, tokenizer=tokenize,ngram_range=(1,2), stop_words='english'),
                     column)

                ])),
                ("SVD", TruncatedSVD(n_components=500)),
                ("forest", RandomForestClassifier(n_estimators=10, random_state=0, max_depth=4))
            ]
        )
        param_grid = {
            'SVD__n_components': [500],
            'forest__n_estimators': [100, 500, 1000],
            'forest__max_depth': [4,8]
        }

       # print(model.fit_transform(X_train[:100], y_train[:100]))
        tscv = TimeSeriesSplit(n_splits=2)
        search = GridSearchCV(model, param_grid, n_jobs=12, cv=tscv.split(X_train), verbose=True)
        search.fit(X_train, y_train)
        print("Best parameter (CV score=%0.3f):" % search.best_score_)
        print(search.best_params_)
        model = search.best_estimator_
        print(model[1].explained_variance_ratio_)


        print(f"Tfidf random forest column {column} eval acc:", np.mean(model.predict(X_eval) == y_eval))
        print(f"Tfidf random forest  column {column} train acc:", np.mean(model.predict(X_train) == y_train))

        with open(f"models/random_forest_{column}.pickle", "wb") as file:
            pickle.dump(model, file)


def test():
    df = pd.read_csv(sys.argv[1])
    for column in ["summary", "reviewText"]:
        with open(f"models/random_forest_{column}.pickle", "rb") as file:
            model = pickle.load(file)
        df = df.replace(np.nan, '', regex=True)
        predictions = model.predict_proba(df)
        print(f"Results for {column}")

        plot_confusion_matrx(predictions, df["score"])

if __name__ == "__main__":
    if len(sys.argv) == 2:
        test()
    else:
        main()
