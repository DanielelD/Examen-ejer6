# -*- coding: utf-8 -*-
"""
Created on Mon May 13 20:07:49 2024

@author: ddiaz
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz

df = pd.read_csv("FIFA2022.csv", sep=",")

X = df.drop('Team', axis=1) 
y = df['Team']

clf = DecisionTreeClassifier()
clf = clf.fit(X, y)

plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=X.columns, class_names=clf.classes_, filled=True)
plt.show()

dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=X.columns,  
                                class_names=clf.classes_,  
                                filled=True, rounded=True,  
                                special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.view() 
