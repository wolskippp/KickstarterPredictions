from scipy import stats
import numpy as np
from data import load_pure_data
import squarify
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    print("Statistical analysis:")

    data = load_pure_data()
    print(data.head())
    print(data.describe())
    
    success_projects = data[data['state'] == 'successful']['state'].count()
    fail_projects = data[data['state'] == 'failed']['state'].count()
    squarify.plot([success_projects, fail_projects], label=['successful', 'failed'], color=['green', 'red'])
    plt.show()

    print('succesful: ', success_projects)
    print('failed: ', fail_projects)
    print('total: ', data['state'].count())
    print('correlation:')
    y_bin = []
    for item in data['state']:
		if item == 'successful':
			y_bin.append(1)
		else:
			y_bin.append(0)
    print('goal to state correlation')
    print(stats.pointbiserialr(y_bin, data['goal']))
    print('duration to state correlation')
    print(stats.pointbiserialr(y_bin, data['duration']))


    print('\n\n anaylisis based on categories')

    proj_count = {}
    for category in list(set(data['main_category'])):
        count = data[data['main_category'] == category]['main_category'].count()
        proj_count[category] = count   
    kick_data = pd.Series(proj_count)
    kick_data = pd.DataFrame(kick_data)
    kick_data = kick_data.rename(columns = {0:'count'})

    success = {}
    for category in list(set(data['main_category'])):
        success_count = len(data[(data['main_category'] == category) & (data['state'] == 'successful')])
        success[category] = success_count
    kick_data['success_count'] = pd.Series(success)

    kick_data['success_rate'] = kick_data['success_count'] / kick_data['count']
    print(kick_data.head())

    kick_data = kick_data.sort_values('success_rate', ascending = False)
    ax = kick_data['success_rate'].plot(kind='bar', title ='Success percentage for categories', figsize=(15, 10), fontsize=12)
    ax.set_xlabel("main_category", fontsize=12)
    ax.set_ylabel("success rate", fontsize=12)
    plt.show()