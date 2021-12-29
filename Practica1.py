from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import neighbors
from mlxtend.data import iris_data
from sklearn.model_selection import train_test_split
from mlxtend.evaluate import combined_ftest_5x2cv

X, y = iris_data()
clf1 = LinearDiscriminantAnalysis()
clf2 = neighbors.KNeighborsClassifier(n_neighbors=5)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.4,
                     random_state=123)

score1 = clf1.fit(X_train, y_train).score(X_test, y_test)
score2 = clf2.fit(X_train, y_train).score(X_test, y_test)

print('LDA accuracy: %.2f%%' % (score1*100))
print('knn accuracy: %.2f%%' % (score2*100))


f, p = combined_ftest_5x2cv(estimator1=clf1,
                            estimator2=clf2,
                            X=X, y=y,
                            random_seed=1)

print('F statistic: %.3f' % f)
print('p value: %.3f' % p)
