import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, auc



# Ma'lumotlarni yuklash
df = pd.read_csv("./User_Data.csv")
# X va Y ni tanlash
x = df.iloc[:, [2, 3]].values
y = df.iloc[:, 4].values
# Datasetni oʻqish va sinov qatorlarga boʻlish
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# X larni standartlash
standard_x = StandardScaler()
x_train = standard_x.fit_transform(x_train)
x_test = standard_x.transform(x_test)
# Logistik regressiya modelini yaratish
model = LogisticRegression()
# Modelni oʻqitish
model.fit(x_train, y_train)
# Test qatorida bashorat qilish
y_pred = model.predict(x_test)
# Bashorat qiymatlarini koʻrsatish
print("y_pred:", y_pred)
# Confusion matrixni hisoblash
cfm = confusion_matrix(y_test, y_pred)
# Confusion matrixni koʻrsatish
print("Confusion Matrix:")
print(cfm)
# Qoʻshimcha metrikalarni hisoblash
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
# Qoʻshimcha metrikalarni koʻrsatish
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
# Confusion matrixni chizish
sns.heatmap(cfm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
# ROC kurvani chizish
y_prob = model.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
# Scatter plot chizish
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 1], Y_test, label='Test qiymatlar', color='red')
plt.scatter(X_test[:, 1], Y_pred, label='Bashorat qiymatlar', color='green')
plt.xlabel(' Taxminiy ish haqi')
plt.ylabel('Haqiqiy')
plt.title('Test va Bashorat Qilingan Qiymatlar')
plt.legend()
plt.show()
