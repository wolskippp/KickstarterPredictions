from data import *
import operator
from sklearn.cross_validation import train_test_split

from regression import run_regression
from tpot_result import run_linearSvc
from xgbclassifier import run_xgboost
import numpy as np


def main():
    data = load_data()
    total_data = len(data)
    total_successes = len(data[data.state == 'successful'])
    total_fails = len(data[data.state == 'failed'])
    total_other = total_data - total_successes - total_fails

    s_rate = round(total_successes / total_data * 100, 2)
    f_rate = round(total_fails / total_data * 100, 2)
    o_rate = round(total_other / total_data * 100, 2)
    print("Total number of projects: {}".format(total_data))
    print("Successes: {} ({}%), fails: {} ({}%), other: {} ({}%)".format(total_successes, s_rate, total_fails, f_rate,
                                                                         total_other, o_rate))

    # input data
    x = data.drop(['state'], 1)
    # output (state of project)
    y = data['state']

    # convert categorical data into classes
    x = preprocess_features(x)
    # print("Processed feature columns ({} total features):\n{}".format(len(x.columns), list(x.columns)))
    features = list(x.columns)
    print("Generated features: {}".format(features))

    categories = [label for label in features if
                  operator.contains(label, "category") and not operator.contains(label, "main_category")]
    print("Category count: {}".format(len(categories)))
    print(categories)

    main_categories = [label for label in features if operator.contains(label, "main_category")]
    print("Main category count: {}".format(len(main_categories)))
    print(main_categories)

    currency = [label for label in features if operator.contains(label, "currency")]
    print("Currency count: {}".format(len(currency)))
    print(currency)

    country = [label for label in features if operator.contains(label, "country")]
    print("Country count: {}".format(len(country)))
    print(country)

    # Shuffle and split the dataset into training and testing set.
    X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.1,
                                                        random_state=2,
                                                        stratify=y)
    tests = []
    for i in range(0, 20):
        acc = run_regression(X_train, y_train, X_test, y_test)
        tests.append(acc)

    print("Linear regression: {}".format(np.average(tests)))

    tests = []
    for i in range(0, 20):
        acc_xgb = run_xgboost(X_train, y_train, X_test, y_test)
        tests.append(acc_xgb)

    print("XGBoost: {}".format(np.average(tests)))

    tests = []
    for i in range(0, 20):
        acc_xgb = run_linearSvc(X_train, y_train, X_test, y_test)
        tests.append(acc_xgb)

    print("Linear Svc: {}".format(np.average(tests)))



if __name__ == '__main__':
    main()
