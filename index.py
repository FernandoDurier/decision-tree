#loading the iris dataset
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
dtree = tree.DecisionTreeClassifier()

#Training the decision tree model
dtree = dtree.fit(iris.data, iris.target)

#creating visualizations from our decision tree
import graphviz 
dot_data = tree.export_graphviz(dtree, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris-simple") 
dot_data = tree.export_graphviz(dtree, out_file=None, 
                     feature_names=iris.feature_names,  
                     class_names=iris.target_names,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.render("iris-complex")
