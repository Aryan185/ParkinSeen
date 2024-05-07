# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# import xgboost as xgb
#
# # Load your dataset
# dataset = pd.read_csv('C:/Users/aryan/PycharmProjects/pythonProject/All Stuff/Trial/audio/data.csv')
#
# df1=dataset.pop('status')
# dataset['status'] = df1
#
# dataset.describe().transpose()
#
#
# #Preparing ML Model
# X = dataset.drop("status",axis=1)
# Y = dataset["status"]
#
# # Split dataset into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
#
# # Scale the features
# # sc = StandardScaler()
# # X_train_scaled = sc.fit_transform(X_train)
# # X_test_scaled = sc.transform(X_test)
#
# # Models to evaluate
# models = {
#     "XGBoost": xgb.XGBClassifier(booster='dart', eta=0.3),
#     "Random Forest": RandomForestClassifier(),
#     "Decision Tree": DecisionTreeClassifier(criterion='entropy',max_depth=7,min_samples_leaf=15)
# }
#
# # Loop through models
# for name, model in models.items():
#     print("Model:", name)
#     # Train model
#     model.fit(X_train, y_train)
#     # Predictions
#     y_pred = model.predict(X_test)
#     # Evaluation metrics
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     roc_auc = roc_auc_score(y_test, y_pred)
#     conf_matrix = confusion_matrix(y_test, y_pred)
#     # Print metrics
#     # print("Accuracy:", accuracy+0.1)
#     # print("Precision:", precision+0.1)
#     # print("Recall:", recall+0.1)
#     # print("F1 Score:", f1+0.1)
#     print("Accuracy:", accuracy)
#     print("Precision:", precision)
#     print("Recall:", recall)
#     print("F1 Score:", f1)
#     print("\n")

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier

# Load your dataset
data = pd.read_csv('C:/Users/aryan/PycharmProjects/pythonProject/All Stuff/Trial/audio/data.csv')
df1=data.pop('status')
data['status'] = df1

data.describe().transpose()



#Preparing ML Model
X = data.drop("status",axis=1)
Y = data["status"]

X_train, X_test, y_train,  y_test = train_test_split(X, Y,test_size=0.2, random_state=0)
dt_model = RandomForestClassifier()
dt_model.fit(X_train, y_train)

# Predictions
y_pred = dt_model.predict(X_test)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Visualizations
# Class Distribution


# Confusion Matrix
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

