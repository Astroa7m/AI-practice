import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


"""//////////////////////Dataset//////////////////////"""
medical_dataset = pd.read_csv("synthetic_symptom_dataset_4000.csv")

"""//////////////////////Splitting//////////////////////"""
train, test = train_test_split(medical_dataset, test_size=0.3, random_state=1)
# (r, c)
print(train.shape)
print(test.shape)
print("Done splitting")

"""//////////////////////Model & Training//////////////////////"""
model = DecisionTreeClassifier(random_state=1)
print("Starting training")
# drop the diseases columns to get only the symtpoms
train_symptoms = train.drop(columns=["disease"])

# get the diseases column, to be used as a classifer target
train_diseases = train["disease"]

# train the model, build a decison tree
model.fit(train_symptoms, train_diseases)
print("Done Training")

# """//////////////////////Plotting//////////////////////"""
# # printing the possible target labels
# print(model.classes_)
# print("Plotting")
# plt.figure(figsize=(12, 6)) # size in inches
#
# # plotting the tree
# plot_tree(model, max_depth=3, fontsize=10, feature_names=medical_dataset.columns[:-1])
# plt.show()

"""//////////////////////Testing & Evaluation//////////////////////"""
print("Starting test")

# dropping diagnose column
test_split = test.drop(columns=['disease'])

# get the diseases column, to be used as a classifier target
classifier_test_target = test['disease']

# guess most likely diagnose
prediction = model.predict(test_split)

print("Ending test")

print("Evaluating...")
#confusion matrix
print("Confusion Matrix")
print(confusion_matrix(classifier_test_target, prediction))
# print the achieved accuracy score
print("Accuracy", end=": ")
print(accuracy_score(classifier_test_target, prediction))