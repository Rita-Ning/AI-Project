"""
File: titanic_level1.py
Name: Rita Tang
----------------------------------
This file builds a machine learning algorithm from scratch 
by Python codes. We'll be using 'with open' to read in dataset,
store data into a Python dict, and finally train the model and 
test it on kaggle. This model is the most flexible one among all
levels. You should do hyperparameter tuning and find the best model.
"""

import math
import util as util

TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename: str, data: dict, mode='Train', training_data=None):
    """
    :param filename: str, the filename to be processed
    :param data: dict[str: list], key is the column name, value is its data
    :param mode: str, indicating the mode we are using
    :param training_data: dict[str: list], key is the column name, value is its data
                          (You will only use this when mode == 'Test')
    :return data: dict[str: list], key is the column name, value is its data
    """
    # PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked

    with open(filename, 'r') as f:
        first = True
        for line in f:
            if first:
                lst1 = line.strip().split(',')
                if mode == 'Train':
                    for i in range(len(lst1)):
                        if i == 0 or i == 3 or i == 8 or i == 10:
                            pass
                        else:
                            data[lst1[i]] = []
                else:
                    for i in range(len(lst1)):
                        if i == 0 or i == 2 or i == 7 or i == 9:
                            pass
                        else:
                            data[lst1[i]] = []
                first = False
            else:
                lst2 = line.strip().split(',')
                if mode == 'Train':
                    if lst2[6] == '' or lst2[12] == '':  # Age, embarked
                        continue
                    data[lst1[1]].append(int(lst2[1]))
                    start = 2
                else:
                    start = 1

                for j in range(len(lst2)):
                    if j == start:  # Pclass
                        data['Pclass'].append(int(lst2[j]))
                    elif j == start+3:  # Sex
                        if lst2[j] == 'male':
                            data['Sex'].append(1)
                        else:
                            data['Sex'].append(0)
                    elif j == start+4:  # Age
                        if lst2[j] != '':
                            data['Age'].append(float(lst2[j]))
                        else:
                            mean = round((sum(training_data['Age']) / len(training_data['Age'])), 3)
                            data['Age'].append(mean)
                    elif j == start+5:  # SibSp
                        data['SibSp'].append(int(lst2[j]))
                    elif j == start+6:  # Parch
                        data['Parch'].append(int(lst2[j]))
                    elif j == start+8:  # Fare
                        if lst2[j] != '':
                            data['Fare'].append(float(lst2[j]))
                        else:
                            mean = round((sum(training_data['Fare']) / len(training_data['Fare'])), 3)
                            data['Fare'].append(mean)
                    elif j == start+10:  # Embarked
                        if lst2[j] == 'S':
                            data['Embarked'].append(0)
                        elif lst2[j] == 'C':
                            data['Embarked'].append(1)
                        elif lst2[j] == 'Q':
                            data['Embarked'].append(2)
                        else:
                            data['Embarked'].append(0)
    return data


def one_hot_encoding(data: dict, feature: str):
    """
    :param data: dict[str, list], key is the column name, value is its data
    :param feature: str, the column name of interest
    :return data: dict[str, list], remove the feature column and add its one-hot encoding features
    """
    if feature == 'Sex':
        data['Sex_0'] = []
        data['Sex_1'] = []
        for sex in data[feature]:
            data['Sex_0'].append(1) if sex == 0 else data['Sex_0'].append(0)
            data['Sex_1'].append(1) if sex == 1 else data['Sex_1'].append(0)
    elif feature == 'Pclass':
        data['Pclass_0'] = []
        data['Pclass_1'] = []
        data['Pclass_2'] = []
        for pcl in data[feature]:
            data['Pclass_0'].append(1) if pcl == 1 else data['Pclass_0'].append(0)
            data['Pclass_1'].append(1) if pcl == 2 else data['Pclass_1'].append(0)
            data['Pclass_2'].append(1) if pcl == 3 else data['Pclass_2'].append(0)
    elif feature == 'Embarked':
        data['Embarked_0'] = []
        data['Embarked_1'] = []
        data['Embarked_2'] = []
        for emb in data[feature]:
            data['Embarked_0'].append(1) if emb == 0 else data['Embarked_0'].append(0)
            data['Embarked_1'].append(1) if emb == 1 else data['Embarked_1'].append(0)
            data['Embarked_2'].append(1) if emb == 2 else data['Embarked_2'].append(0)
    data.pop(feature)
    return data


def normalize(data: dict):
    """
     :param data: dict[str, list], key is the column name, value is its data
     :return data: dict[str, list], key is the column name, value is its normalized data
    """

    for key, value in data.items():
        max_v = max(data[key])
        min_v = min(data[key])
        for i in range(len(value)):
            new_val = (value[i] - min_v) / (max_v - min_v)
            value[i] = new_val

    return data


def learnPredictor(inputs: dict, labels: list, degree: int, num_epochs: int, alpha: float):
    """
    :param inputs: dict[str, list], key is the column name, value is its data
    :param labels: list[int], indicating the true label for each data
    :param degree: int, degree of polynomial features
    :param num_epochs: int, the number of epochs for training
    :param alpha: float, known as step size or learning rate
    :return weights: dict[str, float], feature name and its weight
    """
    # Step 1 : Initialize weights
    weights = {}  # feature => weight
    keys = list(inputs.keys())
    if degree == 1:
        for i in range(len(keys)):
            weights[keys[i]] = 0
    elif degree == 2:
        for i in range(len(keys)):
            weights[keys[i]] = 0
        for i in range(len(keys)):
            for j in range(i, len(keys)):
                weights[keys[i] + keys[j]] = 0
    # Step 2 : Start training
    for epoch in range(num_epochs):
    # Step 3 : Feature Extract
        for i in range(len(labels)):
            feature_v = {}
            if degree == 1:
                for j in range(len(keys)):
                    feature_v[keys[j]] = inputs[keys[j]][i]
            else:
                for j in range(len(keys)):
                    feature_v[keys[j]] = inputs[keys[j]][i]
                    for k in range(j, len(keys)):
                        feature_v[keys[j] + keys[k]] = inputs[keys[j]][i] * inputs[keys[k]][i]
    # Step 4 : Update weights
            y = labels[i]
            k = util.dotProduct(feature_v, weights)
            h = 1/(1 + math.exp(-k))
            util.increment(weights, -alpha * (h - y), feature_v)

    return weights
