import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
import scikitplot
import matplotlib.pyplot as plt
"""//////////////////////Dataset//////////////////////"""
imbd_reviews = pd.read_csv("IMDB Dataset.csv")

"""//////////////////////Preprocessing//////////////////////"""
# mappging from negative to 0 and 1 to positive
imbd_reviews['sentiment'] = imbd_reviews['sentiment'].apply(lambda x: 1 if x == "positive" else 0)

# data to be entered
X = imbd_reviews['review']
# data to be predicted
y = imbd_reviews['sentiment']

"""//////////////////////Splitting//////////////////////"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

"""//////////////////////Vectorization//////////////////////"""
# min_def is used to ignore terms that appear in less than 10 reviews
vectorizer = CountVectorizer(min_df=10)

x_train_sparse = vectorizer.fit_transform(X_train)

# just to visualize the matrix
# x_train_dense = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names_out())
#
# print(x_train_dense)
#
# from sys import getsizeof
# print("Megabytes of RAM memory used by the raw text format", getsizeof(X_train)/1000000)
# print("Megabytes of RAM memory used by the dense matrix", getsizeof(x_train_dense)/1000000)
# print("Megabytes of RAM memory used by the sparse matrix", getsizeof(x)/1000000)

"""//////////////////////Training & Building Prediction Pipeline//////////////////////"""

model = MultinomialNB() # naive bayes classifier

model.fit(x_train_sparse, y_train)

prediction_pipleline = make_pipeline(vectorizer, model)

"""//////////////////////Testing & Evaluation//////////////////////"""


# test_reviews = [
#     "I liked the movie so much, it was one of the best experiences of my entire life",
#     "I had high hopes regarding what I witnessed",
#     "Was expecting much more, but I will take it",
#     "It is one of my lifetime moments when I will think about it and regret so much, won't recommend it"
# ]
#
#
# print(prediction_pipleline.predict_proba(test_reviews)) # yields (negative proba, positive proba) prediciton
# print(prediction_pipleline.predict(test_reviews)) # yields predicition result

# testing the test set
prediciton_result = prediction_pipleline.predict(X_test)
print("accuracy_score:", accuracy_score(y_test, prediciton_result))


labels = ["Negative", "Positive"]


plot = scikitplot.metrics.plot_confusion_matrix(
    y_true=[labels[i] for i in y_test],
    y_pred=[labels[i] for i in prediciton_result],
    title="Confusion Matrix",
    cmap="Reds",
    figsize=(5,5)
)
plt.show()

# exporting prediction pipeline
file_name = "../pipelines/sentiment_pipeline.pkl"
joblib.dump(prediction_pipleline, file_name)
print(f"exported pipeline to: {file_name}")