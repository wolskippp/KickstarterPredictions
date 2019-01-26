from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def run_regression(x, y, x_test, y_test):
	clf = LogisticRegression(random_state=42)
	clf.fit(x, y)
	y_pred = clf.predict(x_test)
	y_pred_train = clf.predict(x)
	acc = sum(y == y_pred_train) / float(len(y_pred_train))

	print('Train')
	print(acc)
	return sum(y_test == y_pred) / float(len(y_pred))