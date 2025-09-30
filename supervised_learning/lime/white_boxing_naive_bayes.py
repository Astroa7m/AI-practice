import joblib
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
prediciton_pipeline = joblib.load("../pipelines/sentiment_pipeline.pkl")

class_names = ["Negative", "Positive"]
explainer = LimeTextExplainer(class_names=class_names)

def pretty_predict(texts, return_proba=True):
    """Helper function just to beautify the output"""
    result = ["Positive" if pred == 1 else "Negative" for pred in prediciton_pipeline.predict(texts)]
    result2 = ""
    if return_proba:
        result2 = [f"Poss: {round(pred[0]*100)}% Negaitve & {round(pred[1]*100)}% Positive" for pred in prediciton_pipeline.predict_proba(texts)]
    return str(result) + f"\n{str(result2)}"

test_reviews = [
    "I liked the movie so much, it was one of the best experiences of my entire life",
    "It is one of my lifetime moments when I will think about it and regret so much, won't recommend it"
]

print("Without lime")
print(pretty_predict(test_reviews))

print("/nWith Lime")
# explaining the prediction for the sample
# and making it explain the most 10 influential features
exp= explainer.explain_instance(
    test_reviews[1],
    prediciton_pipeline.predict_proba,
    num_features=10
)

# std output
for line in exp.as_list():
    print(line)

# figure

fig = exp.as_pyplot_figure()
plt.show()