import seaborn as sns
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import matplotlib.pyplot as plt 
import matplotlib.colors as colors
import sklearn.model_selection as model_selection
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score, ConfusionMatrixDisplay, classification_report)
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import pandas as pd
import pickle

# Load the data
data=pd.read_csv("E:/coolyeah unair/materi kuliah/Semester 5/AVD prak/uts/Survei Kepuasan Koperasi Bagi Mahasiswa FST (Jawaban) - Form Responses 1.csv")

# Print the data summary
print(data.head(10))
print("============================================================")

# Check for missing values
print("Checking for missing values".center(75,"="))
print(data.isnull().sum())
print("============================================================")

# Group the variables
print("GROUPING VARIABLES".center(75,"="))
X = data.loc[:,["Menurut anda koperasi FST yang baru terlihat lebih higienis dibanding yang lama", "Menurut anda koperasi FST yang baru lebih baik pelayanannya dibanding yang lama","Menurut anda ketersediaan dagangan koperasi FST yang baru lebih bervariasi dibandingkan yang sebelumnya"]]
y = data.loc[:,["Menurut anda secara garis besar, anda puas dengan koperasi FST yang baru"]]

# # Normalize the data using z-score
# print("NORMALIZING THE DATA USING Z-SCORE".center(75,"="))
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# # Perform feature selection using Pearson correlation
# print("PERFORMING FEATURE SELECTION USING PEARSON CORRELATION".center(75,"="))
# X = SelectKBest(f_classif, k=3).fit_transform(X, y)

# Split the data into training and testing sets
print("SPLITTING DATA 20-80".center(75,"="))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# Bin the data
kbins = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
X_train_binned = kbins.fit_transform(X_train)
X_test_binned = kbins.transform(X_test)

# Train a Naive Bayes classifier
print("TRAINING NAIVE BAYES CLASSIFIER".center(75,"="))
clf = GaussianNB()
clf.fit(X_train_binned, y_train)

# Make predictions and evaluate the performance
print("EVALUATING PERFORMANCE".center(75,"="))
y_pred = clf.predict(X_test_binned)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='micro')
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Classification Report:\n", classification_report(y_test, y_pred,))
cm = confusion_matrix(y_test, y_pred)
TN = cm[1][1] * 1.0
FN = cm[1][0] * 1.0
TP = cm[0][0] * 1.0
FP = cm[0][1] * 1.0
total = TN + FN + TP + FP
sens = TN / (TN + FP) * 100
spec = TP / (TP + FN) * 100
print('Akurasi : ', accuracy * 100, "%")
print('Sensitivity : ' + str(sens))
print('Specificity : ' + str(spec))
print('Precision : ' + str(precision))

# Display the confusion matrix
print("Confusion Matrix for Naive Bayes Classifier:")
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(cm, 
            annot=True, 
            fmt=".0f", 
            ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("============================================================")

# input data baru
new_data = {"Menurut anda koperasi FST yang baru terlihat lebih higienis dibanding yang lama": 1, "Menurut anda koperasi FST yang baru lebih baik pelayanannya dibanding yang lama":1,"Menurut anda ketersediaan dagangan koperasi FST yang baru lebih bervariasi dibandingkan yang sebelumnya":1}

# membuat dataframe
new_df = pd.DataFrame([new_data])
print(new_df)

# membuat pengelompokan (binning) pada data baru
new_binned = kbins.transform(new_df)
new_binned_2d = new_binned.reshape(1, -1)  # Convert to 2D array
print(new_binned_2d)

# melakukan prediksi
predTest = clf.predict(new_binned_2d)
print(predTest)

pickle.dump(clf, open('NBayes.pkl', 'wb'))
# # menampilkan hasil
# if predTest == 1:
#     print("Orang tersebut mengalami diabetes")
# else:
#     print("Orang tersebut tidak mengalami diabetes")

# membuat pengelompokan (binning) pada data baru 
# new_binned = kbins.transform(new_df.iloc[:, 1:3])

# # melakukan prediksi
# predTest = clf.predict(new_binned)
# print(predTest)

# # menampilkan hasil
# if predTest == 4:
#     print("Orang tersebut mengalami kanker payudara")
# else:
#     print("Orang tersebut tidak kanker payudara")