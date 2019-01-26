import xgboost as xgb
from sklearn.metrics import f1_score



def run_xgboost(X, y, X_test, y_test):
    clf = xgb.XGBClassifier(max_depth=24, learning_rate=0.05, silent=False, random_state=997, n_estimators=200)
    
    clf.fit(X, y)

    y_pred_train = clf.predict(X)
    y_pred_test = clf.predict(X_test)

    acc_train = sum(y == y_pred_train) / float(len(y_pred_train))

    print(acc_train)

    acc_test = sum(y_test == y_pred_test) / float(len(y_pred_test))
    return acc_test