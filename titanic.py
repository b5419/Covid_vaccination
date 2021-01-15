import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication
import seaborn as sns


def add_clipboard_to_figures():
    # use monkey-patching to replace the original plt.figure() function with
    # our own, which supports clipboard-copying
    oldfig = plt.figure

    def newfig(*args, **kwargs):
        fig = oldfig(*args, **kwargs)

        def clipboard_handler(event):
            if event.key == 'ctrl+c':
                # store the image in a buffer using savefig(), this has the
                # advantage of applying all the default savefig parameters
                # such as background color; those would be ignored if you simply
                # grab the canvas using Qt
                buf = io.BytesIO()
                fig.savefig(buf)
                QApplication.clipboard().setImage(QImage.fromData(buf.getvalue()))
                buf.close()

        fig.canvas.mpl_connect('key_press_event', clipboard_handler)
        return fig

    plt.figure = newfig


add_clipboard_to_figures()


df = pd.read_csv("train.csv")
df = df.dropna(subset=["Age", "Embarked"])

df = df.join(pd.get_dummies(df["Pclass"].astype(str), prefix="Pclass"))
df = df.join(pd.get_dummies(df[["Sex", "Embarked"]]))


df = df.drop(columns=["Cabin", "Name","Pclass", "Sex", "Embarked","PassengerId", "Ticket"])

print(df.keys())



#sns.pairplot(df, hue="Survived", vars=['Age', 'SibSp', 'Parch', 'Fare'], diag_kind="hist")
plt.show()

plt.figure(1)
width = 0.1
ind = np.arange(2)
unsurvived = np.array([len(df.loc[(df["Survived"]==0)][(df["Sex_female"]==1)]), len(df.loc[(df["Survived"]==0)][(df["Sex_male"]==1)])])
survived = np.array([len(df.loc[(df["Survived"]==1)][(df["Sex_female"]==1)]), len(df.loc[(df["Survived"]==1)][(df["Sex_male"]==1)])])
p1 = plt.bar(ind, survived)
p2 = plt.bar(ind, unsurvived, bottom=survived)

plt.ylabel('Scores')
plt.title('proportion of survivor wrt the gender')
plt.xticks(ind, ("Women", "Men"))
plt.yticks(np.arange(0, 453, 50))
plt.legend((p1[0], p2[0]),("Survived", "Unsurvived"))

for xpos, ypos, yval in zip(ind , survived/2, np.round(survived/(survived+unsurvived),3)):
    plt.text(xpos, ypos, yval, ha="center", va="center")
for xpos, ypos, yval in zip(ind, survived+unsurvived/2, np.round(unsurvived/(survived+unsurvived),3)):
    plt.text(xpos, ypos, yval, ha="center", va="center")
# add text annotation corresponding to the "total" value of each bar
for xpos, ypos, yval in zip(ind, survived+unsurvived, survived+unsurvived):
    plt.text(xpos, ypos, "N=%d"%yval, ha="center", va="bottom")

plt.close()


plt.figure(2)
width = 0.1
ind = np.arange(2)


Embarked_C = np.array([len(df.loc[(df["Survived"]==0)][(df["Embarked_C"]==1)]),
                     len(df.loc[(df["Survived"]==1)][(df["Embarked_C"]==1)])])

Embarked_Q = np.array([len(df.loc[(df["Survived"]==0)][(df["Embarked_Q"]==1)]),
                     len(df.loc[(df["Survived"]==1)][(df["Embarked_Q"]==1)])])

Embarked_S = np.array([len(df.loc[(df["Survived"]==0)][(df["Embarked_S"]==1)]),
                     len(df.loc[(df["Survived"]==1)][(df["Embarked_S"]==1)])])


plt.bar(ind- width/2, Embarked_C, label="Embarked_C")
plt.bar(ind- width/2, Embarked_Q, bottom=Embarked_C, label="Embarked_Q")
plt.bar(ind- width/2, Embarked_S, bottom=Embarked_C+Embarked_Q, label="Embarked_S")

plt.ylabel('Scores')
plt.title('proportion of survivor wrt the embarked location')
plt.xticks(ind, ('survivor', 'death'))
plt.yticks(np.arange(0, 554, 50))


for xpos, ypos, yval in zip(ind , Embarked_C/2, np.round(Embarked_C/(Embarked_C + Embarked_Q + Embarked_S),3)):
    plt.text(xpos, ypos, yval, ha="center", va="center")
for xpos, ypos, yval in zip(ind, Embarked_C+Embarked_Q/2, np.round(Embarked_Q/(Embarked_C + Embarked_Q + Embarked_S),3)):
    plt.text(xpos, ypos, yval, ha="center", va="center")
for xpos, ypos, yval in zip(ind, Embarked_C + Embarked_Q + Embarked_S / 2, np.round(Embarked_S / (Embarked_C + Embarked_Q + Embarked_S), 3)):
    plt.text(xpos, ypos, yval, ha="center", va="center")
# add text annotation corresponding to the "total" value of each bar
for xpos, ypos, yval in zip(ind, Embarked_C + Embarked_Q + Embarked_S, Embarked_C + Embarked_Q + Embarked_S):
    plt.text(xpos, ypos, "N=%d"%yval, ha="center", va="bottom")


plt.legend(bbox_to_anchor=(1.01, 0.5), loc='center left')
plt.tight_layout(rect=[0, 0, 1, 1])

plt.close()

plt.figure(3)
width = 0.1
ind = np.arange(2)


Pclass_1 = np.array([len(df.loc[(df["Survived"]==0)][(df["Pclass_1"]==1)]),
                     len(df.loc[(df["Survived"]==1)][(df["Pclass_1"]==1)])])

Pclass_2 = np.array([len(df.loc[(df["Survived"]==0)][(df["Pclass_2"]==1)]),
                     len(df.loc[(df["Survived"]==1)][(df["Pclass_2"]==1)])])

Pclass_3 = np.array([len(df.loc[(df["Survived"]==0)][(df["Pclass_3"]==1)]),
                     len(df.loc[(df["Survived"]==1)][(df["Pclass_3"]==1)])])


plt.bar(ind, Pclass_1, label="Pclass_1")
plt.bar(ind, Pclass_2, bottom= Pclass_1, label="Pclass_2")
plt.bar(ind, Pclass_3, bottom= Pclass_1+Pclass_2, label="Pclass_3")

plt.ylabel('Scores')
plt.title('proportion of survivor wrt the location in the boad')
plt.xticks(ind, ('survivor', 'death'))
plt.yticks(np.arange(0, 554, 50))


for xpos, ypos, yval in zip(ind , Pclass_1/2, np.round(Pclass_1/(Pclass_1 + Pclass_2 + Pclass_3),3)):
    plt.text(xpos, ypos, yval, ha="center", va="center")
for xpos, ypos, yval in zip(ind, Pclass_1+Pclass_2/2, np.round(Pclass_2/(Pclass_1 + Pclass_2 + Pclass_3),3)):
    plt.text(xpos, ypos, yval, ha="center", va="center")
for xpos, ypos, yval in zip(ind, Pclass_1 + Pclass_2 + Pclass_3 / 2, np.round(Pclass_3 / (Pclass_1 + Pclass_2 + Pclass_3), 3)):
    plt.text(xpos, ypos, yval, ha="center", va="center")
# add text annotation corresponding to the "total" value of each bar
for xpos, ypos, yval in zip(ind, Pclass_1 + Pclass_2 + Pclass_3, Pclass_1 + Pclass_2 + Pclass_3):
    plt.text(xpos, ypos, "N=%d"%yval, ha="center", va="bottom")


plt.legend(bbox_to_anchor=(1.01,0.5), loc='center left')
plt.tight_layout(rect=[0,0,1,1])

plt.close()

plt.figure(4)
width = 0.1
ind = np.arange(2)


Pclass_1 = np.array([len(df.loc[(df["Survived"]==0)][(df["Pclass_1"]==1)]),
                     len(df.loc[(df["Survived"]==1)][(df["Pclass_1"]==1)])])

Pclass_2 = np.array([len(df.loc[(df["Survived"]==0)][(df["Pclass_2"]==1)]),
                     len(df.loc[(df["Survived"]==1)][(df["Pclass_2"]==1)])])

Pclass_3 = np.array([len(df.loc[(df["Survived"]==0)][(df["Pclass_3"]==1)]),
                     len(df.loc[(df["Survived"]==1)][(df["Pclass_3"]==1)])])


plt.bar(ind, Pclass_1, label="Pclass_1")
plt.bar(ind, Pclass_2, bottom= Pclass_1, label="Pclass_2")
plt.bar(ind, Pclass_3, bottom= Pclass_1+Pclass_2, label="Pclass_3")

plt.ylabel('Scores')
plt.title('proportion of survivor wrt the location in the boad')
plt.xticks(ind, ('survivor', 'death'))
plt.yticks(np.arange(0, 554, 50))


for xpos, ypos, yval in zip(ind, Pclass_1/2, np.round(Pclass_1/(Pclass_1 + Pclass_2 + Pclass_3), 3)):
    plt.text(xpos, ypos, yval, ha="center", va="center")
for xpos, ypos, yval in zip(ind, Pclass_1+Pclass_2/2, np.round(Pclass_2/(Pclass_1 + Pclass_2 + Pclass_3), 3)):
    plt.text(xpos, ypos, yval, ha="center", va="center")
for xpos, ypos, yval in zip(ind, Pclass_1 + Pclass_2 + Pclass_3 / 2, np.round(Pclass_3 / (Pclass_1 + Pclass_2 + Pclass_3), 3)):
    plt.text(xpos, ypos, yval, ha="center", va="center")
# add text annotation corresponding to the "total" value of each bar
for xpos, ypos, yval in zip(ind, Pclass_1 + Pclass_2 + Pclass_3, Pclass_1 + Pclass_2 + Pclass_3):
    plt.text(xpos, ypos, "N=%d" %yval, ha="center", va="bottom")


plt.legend(bbox_to_anchor=(1.01, 0.5), loc='center left')
plt.tight_layout(rect=[0, 0, 1, 1])
plt.close()


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,7))
sns.distplot(df.loc[(df["Survived"]== 0)][(df["Sex_female"]==1)]["Age"], color='b', label='non-surviving women', ax = ax1, kde=False)
sns.distplot(df.loc[(df["Survived"]== 1)][(df["Sex_female"]==1)]["Age"], color='darkorange', label='surviving women', ax = ax1, kde=False)

sns.distplot(df.loc[(df["Survived"]== 0)][(df["Sex_male"]==1)]["Age"], color='b', kde= False, label='non-surviving men', ax = ax2)
sns.distplot(df.loc[(df["Survived"]== 1)][(df["Sex_male"]==1)]["Age"], color='orange', kde=False, label='surviving men', ax = ax2)

ax1.legend()
ax2.legend()
plt.close()






X_train = df.loc[:, df.columns != 'Survived']
y_train = df["Survived"]




from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score



"""

clf = RandomForestClassifier()
pipe = Pipeline([('clf', clf)])

param_grid = [{'bootstrap': [False, True],
              'n_estimators': [50, 100, 250, 500],
              'max_depth': [2, 5, 10, 50, 100]}]

gs = GridSearchCV( estimator= clf,
                   param_grid = param_grid,
                   scoring= "accuracy",
                   cv= 10,
                   n_jobs = -1)


gs = gs.fit(X_train, y_train)

print(gs.best_score_)
print(gs.best_params_)

score = cross_val_score(gs, X_train, y_train, cv=5)
print(score)
"""

"""
clf = RandomForestClassifier( n_estimators= 100, bootstrap= True, max_depth= 50, criterion= 'entropy',
                              random_state=0)

scores = cross_val_score(estimator = clf, X= X_train, y= y_train, cv= 10, n_jobs=2)
clf.fit(X_train, y_train)

feat_labels = df.columns[1:]
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[f], importances[indices[f]]))

"""

"""
from sklearn.neighbors import KNeighborsClassifier

parameters = {"n_neighbors": [3, 5, 8, 12], "p": [1, 2]}
knn = KNeighborsClassifier()
pipe = Pipeline([('knn', knn)])


gs = GridSearchCV( estimator= knn,
                   param_grid = parameters,
                   scoring= "accuracy",
                   cv= 10,
                   n_jobs = -1)


gs = gs.fit(X_train, y_train)

print(gs.best_score_)
print(gs.best_params_)

score = cross_val_score(gs, X_train, y_train, cv=5)
print(score)
"""
"""
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors=5, p=1, metric="minkowski")
scores = cross_val_score(estimator = KNN, X= X_train, y= y_train, cv= 10, n_jobs=2)
"""

"""
from sklearn import svm
from sklearn.preprocessing import StandardScaler


parameters = [{'clf__C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'clf__gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]}]

clf = svm.SVC(kernel='rbf')
pipe = Pipeline(steps=[('sc', StandardScaler()),('clf', svm.SVC(kernel='rbf', random_state=0))])


gs = GridSearchCV( estimator= pipe,
                   param_grid = parameters,
                   scoring= "accuracy",
                   cv= 10,
                   n_jobs = -1)


gs = gs.fit(X_train, y_train)

print(gs.best_score_)
print(gs.best_params_)

score = cross_val_score(gs, X_train, y_train, cv=5)
print(score)
"""

from sklearn import svm
from sklearn.preprocessing import StandardScaler

clf = svm.SVC(kernel='rbf', C = 1e3, gamma = 5e-3)
pipe = Pipeline(steps=[('sc', StandardScaler()),('clf', clf)])
score = cross_val_score(estimator = pipe, X= X_train, y= y_train, cv= 10, n_jobs=2)