import pandas as pd
import numpy as np
from numpy import log2 as log
import pprint

eps = np.finfo(float).eps

dataset = {
    'Age': ['36-55', '18-35', '36-55', '18-35', '<18', '18-35', '36-55', '>55', '36-55', '>55', '36-55', '>55', '<18',
            '36-55', '36-55', '<18', '18-35', '>55', '>55', '36-55'],
    'Education': ['Masters', 'High School', 'Masters', 'Bachelors', 'High School', 'Bachelors', 'Bachelors',
                  'Bachelors', 'Masters', 'Masters', 'Masters', 'Masters', 'High School', 'Masters', 'High School',
                  'High School', 'Bachelors', 'High School', 'Bachelors', 'High School'],
    'Income': ['High', 'Low', 'Low', 'High', 'Low', 'High', 'Low', 'High', 'Low', 'Low', 'High', 'High', 'High', 'Low',
               'Low', 'Low', 'High', 'High', 'Low', 'High'],
    'Marital Status': ['Single', 'Single', 'Single', 'Single', 'Single', 'Married', 'Married', 'Single', 'Married',
                       'Married', 'Single', 'Single', 'Single', 'Single', 'Single', 'Married', 'Married', 'Married',
                       'Single', 'Married'],
    'Buy Computer': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes',
                     'Yes', 'No', 'Yes', 'Yes', 'Yes']
}

df = pd.DataFrame(dataset, columns=['Age', 'Education', 'Income', 'Marital Status', 'Buy Computer'])


def find_entropy(df):
    Class = df.keys()[-1]
    entropy = 0
    values = df[Class].unique()
    for value in values:
        fraction = float(df[Class].value_counts()[value]) / len(df[Class])
        entropy += -fraction * np.log2(fraction)
    return entropy


def find_entropy_attribute(df, attribute):
    Class = df.keys()[-1]
    values = df[Class].unique()
    variables = df[attribute].unique()
    for variable in variables:
        entropy = 0
        entropy2= 0
        for target_variable in values:
            num = len(df[attribute][df[attribute] == variable][df[Class] == target_variable])
            dem = len(df[attribute][df[attribute] == variable])
            fraction = float(num) / (dem + eps)
            entropy += -fraction * log(fraction + eps)
        fraction2 = dem / len(df)
        entropy2 += fraction2 * entropy
    return abs(entropy2)


def find_winner(df):
    IG = []
    for key in df.keys()[:-1]:
        IG.append(find_entropy(df) - find_entropy_attribute(df, key))
    return df.keys()[:-1][np.argmax(IG)]


def get_subtable(df, node, value):
    return df[df[node] == value].reset_index(drop=True)


def buildTree(df, tree=None):
    Class = df.keys()[-1]
    node = find_winner(df)
    attValue = np.unique(df[node])
    if tree is None:
        tree = {}
        tree[node] = {}
        for value in attValue:
            subtable = get_subtable(df, node, value)
            clValue, counts = np.unique(subtable['Buy Computer'], return_counts=True)
            if len(counts) == 1:
                tree[node][value] = clValue[0]
            else:
                tree[node][value] = buildTree(subtable)
    return tree


tree = buildTree(df)

pprint.pprint(tree)
