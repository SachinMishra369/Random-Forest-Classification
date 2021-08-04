#import  libraries
import  pandas as pd
import numpy as np

#Reading the dataset
dataset=pd.read_csv('social_ads.csv')

#dependent variable
y=dataset.iloc[:,-1].values

#independent variable
x=dataset.iloc[:,:-1].values


#Diving the dataset into training and testing set
from sklearn.model_selection  import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size = 0.25)

#feature scaling scaling the data into a scale so that none of feature get dominant by other features
from sklearn.preprocessing import StandardScaler
sc_x= StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)

#RandomForestClassifier

'''
class sklearn.ensemble.RandomForestClassifier(n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
'''
from sklearn.ensemble import RandomForestClassifier

classifer= RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifer.fit(x_train,y_train)
#Logistic regressiomn predict the result
y_pred = classifer.predict(x_test)
print(y_pred)
print("predict FOR AGE 32 AND SALARY 120000")
age=32
salary=120000
print(classifer.predict([[age,salary]]))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

# Visualising the Training set results
# import matplotlib.pyplot  as plt
# from matplotlib.colors import ListedColormap
# X_set, y_set = x_train, y_train
# X1, X2 = np.meshgrid(np.arange(start = X_set[: , 0].min() - 1, stop = X_set[: , 0].max() + 1, step = 0.01),
#   np.arange(start = X_set[: , 1].min() - 1, stop = X_set[: , 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, classifer.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#   alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#   plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#     c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('KNN Classifier (Training set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()

# Visualising the testing set results
import matplotlib.pyplot  as plt
from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[: , 0].min() - 1, stop = X_set[: , 0].max() + 1, step = 0.01),
  np.arange(start = X_set[: , 1].min() - 1, stop = X_set[: , 1].max() + 1, step = 0.01))
print(X1,X2)
plt.contourf(X1, X2, classifer.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
  alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
  plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
    c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Gausian Naive Bayes Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()