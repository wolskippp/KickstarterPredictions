from sklearn.svm import LinearSVC


def run_linearSvc(X, y, X_test, y_test):
    clf = LinearSVC(C=20.0, dual=False, loss="squared_hinge", penalty="l1", tol=0.01)

    clf.fit(X, y)

    y_pred_train = clf.predict(X)
    y_pred_test = clf.predict(X_test)

    acc_train = sum(y == y_pred_train) / float(len(y_pred_train))

    print(acc_train)

    acc_test = sum(y_test == y_pred_test) / float(len(y_pred_test))
    return acc_test

