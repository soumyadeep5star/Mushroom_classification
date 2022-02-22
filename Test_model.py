# importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# importing the dataset
data = pd.read_csv(r'C:\Users\SOUMYADEEP\PycharmProjects\Mushroom_Classification\mushrooms.csv')

# describing the dataset
x = data.describe(include='all')
data.info()

# looking for missing data
data.isnull().sum()

# converting categorical data to numerical data
from sklearn.preprocessing import LabelEncoder
df1 = data.copy()
le = LabelEncoder()
for col in df1.columns:
    df1[col] = le.fit_transform(df1[col])

#plotting histogram
df1.hist(figsize=(20,30))

#to get better view of the data
pd.crosstab(data['class'],data['cap-shape'] )
#sample output
#cap-shape    b  c     f    k   s     x
#class
#e          404  0  1596  228  32  1948
#p           48  4  1556  600   0  1708

#to plot the countplot of the dataset
sns.countplot(x="class", hue="cap-shape", data=data)

#to view dataset as pivot table
pd.pivot_table(data, index=['class', 'cap-shape'],
                     columns=[ 'cap-surface'], aggfunc=len)

#to get the mapping of encoded data
le_t = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_t)
#for this dataset for example we try to know mapping of df1['habitat'] we got
#{'d': 0, 'g': 1, 'l': 2, 'm': 3, 'p': 4, 'u': 5, 'w': 6}

# selecting features and target(X as features and Y as target)
X = df1.iloc[:,1:23].values
Y = df1.iloc[:,[0]].values

# converting 2D array to 1D column vector
Y = Y.ravel()

# splitting training and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# making logistic regression model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)

# making kneighborsclassifier model
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, Y_train)

# making support vector classifier model
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, Y_train)

# making kernel support vector classifier model
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, Y_train)

# making naive bayes classifier model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

# making decision tree classifier model
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)

# making random forest classifier model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)

# predicting the result
y_pred = classifier.predict(X_test)

# evaluating the accuracy of the model
from sklearn.metrics import accuracy_score
ace = accuracy_score(Y_test,y_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)
print(cm)

#evaluating classification model using ROC curve
from sklearn.metrics import roc_curve, auc
prob_predict = classifier.predict_proba(X_test)
prob_predict = prob_predict[:,1] #Taking probability of positive class
fpr, tpr, threshold = roc_curve(Y_test, prob_predict)
auroc = auc(fpr,tpr)
label = 'Model Classifier AUC:' + ' {0:.2f}'.format(auroc)
plt.plot([0,1], [0,1], 'r--')
plt.plot(fpr, tpr, c = 'green', label = label)
plt.title('Receiver Operating Characteristic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

#evaluating classification model using CAP curve analysis
total = len(Y_test)
class_1_count = np.sum(Y_test)
class_0_count = total - class_1_count
#Random model
plt.plot([0, total], [0, class_1_count], c = 'r', linestyle = '--', label = 'Random Model')
#Perfect Model
plt.plot([0, class_1_count, total], [0, class_1_count, class_1_count], c = 'grey', label = 'Perfect Model')
#Trained Model
probs = classifier.predict_proba(X_test)
probs = probs[:, 1]
model_y = [y for _, y in sorted(zip(probs, Y_test), reverse = True)]
y_values = np.append([0], np.cumsum(model_y))
x_values = np.arange(0, total + 1)
plt.plot(x_values, y_values, c = 'b', label = 'Model Classifier')
plt.title('Cumulative Accuracy Profile')
plt.xlabel('Total observations')
plt.ylabel('Class 1 observations')
plt.legend(loc = 'lower right')
plt.show()

#to find ratio of areas
a = auc([0, total], [0, class_1_count])
# Area between Perfect and Random Model
aP = auc([0, class_1_count, total], [0, class_1_count, class_1_count]) - a
# Area between Trained and Random Model
aR = auc(x_values, y_values) - a
print("Accuracy Rate for Model Classifier: {}".format(aR / aP))

#another way of analysing model
# Point where vertical line will cut trained model
index = int((50*total / 100))
## 50% Verticcal line from x-axis
plt.plot([index, index], [0, y_values[index]], c ='g', linestyle = '--')
## Horizontal line to y-axis from prediction model
plt.plot([0, index], [y_values[index], y_values[index]], c = 'g', linestyle = '--')
class_1_observed = y_values[index] * 100 / max(y_values)
print(class_1_observed)

#accuracy_score from sklearn.metric for classification models
#logistic regression = 0.950763
#Kneighbour classifier = 1.0
#support vector classifier = 0.987690
#kernel support vector classifier = 1.0
#naive bayes classifier = 0.91432
#decision tree classifier = 1.0
#random forest classifier = 1.0

#evaluating accuracy by AUROC curve(Area under Reciever Operating Characteristics)
#logistic regression = 0.98874(98.874%)
#Kneighbour Classifier = 1.0(100%)
#naive bayes classifier = 0.95(95%)
#decision tree classifier = 1.0(100%)
#random forest classifier = 1.0(100%)

#evaluating accuracy of the model by analysing CAP curve (Cumulative Accuracy Profile)
#firstly analysing by area ratio(aR/aP)
#logistic regression = 0.97748
#Kneighbour classifier = 1.0
#naive bayes classifier = 0.901394
#decision tree classifier = 1.0
#random forest classifier = 1.0

#secondly analysing by drawing vertical line at 50% from the x-axis till it crosses the plot formed by model
#logistic regression = 98.144%
#Kneighbour classifier = 100%
#naive bayes classifier = 92.8865%
#decision tree classifier = 100%
#random forest classifier = 100%









