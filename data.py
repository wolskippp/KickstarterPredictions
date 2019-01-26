import pandas as pd
from config import *
from sklearn.preprocessing import scale


def load_data():
	frames = []
	for file in DATA_FILES:
		data_frame = pd.read_csv(file, encoding='ISO-8859-1', usecols=DATA_COLUMNS)
		frames.append(data_frame)

	data = pd.concat(frames)
	data.launched = pd.to_datetime(data.launched)
	data.deadline = pd.to_datetime(data.deadline)
	data['duration'] = scale((data.deadline - data.launched).dt.days)
	data['goal'] = scale(data['goal'])
	data = data.drop(['launched'], 1)
	data = data.drop(['deadline'], 1)
	data = data[(data.state=='successful') | (data.state == 'failed')]
	print(data.head())
	return data

def load_pure_data():
	frames = []
	for file in DATA_FILES:
		data_frame = pd.read_csv(file, encoding='ISO-8859-1')
		frames.append(data_frame)

	data = pd.concat(frames)
	data.launched = pd.to_datetime(data.launched)
	data.deadline = pd.to_datetime(data.deadline)
	# data['duration'] = scale((data.deadline - data.launched).dt.days)
	data['duration'] = (data.deadline - data.launched).dt.days
	# data['goal'] = scale(data['goal'])
	data['goal'] = data['goal']
	data = data.drop(['launched'], 1)
	data = data.drop(['deadline'], 1)
	data = data[(data.state=='successful') | (data.state == 'failed')]
	return data


# we want continous vars that are integers for our input data, so lets remove any categorical vars
def preprocess_features(X):
	''' Preprocesses the kickstarter data and converts catagorical variables into dummy variables. '''

	# Initialize new output DataFrame
	output = pd.DataFrame(index=X.index)

	# Investigate each feature column for the data
	for col, col_data in X.iteritems():

		# If data type is categorical, convert to dummy variables
		if col_data.dtype == object:
			col_data = pd.get_dummies(col_data, prefix=col)

		# Collect the revised columns
		output = output.join(col_data)

	return output
