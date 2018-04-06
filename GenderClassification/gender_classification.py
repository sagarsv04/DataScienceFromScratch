from sklearn import tree
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Data
# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
# labels
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

X_Test = [[180, 80, 42], [177, 72, 43], [160, 60, 35], [154, 50, 35], [166, 65, 45]]

Y_Test = ['male', 'male', 'female', 'female', 'male']

# Classifiers
# using the default values for all the hyperparameters
clf_tree = tree.DecisionTreeClassifier()
clf_svm = SVC()
clf_perceptron = Perceptron()
clf_KNN = KNeighborsClassifier()

# list of classifiers with respective names
classifiers = [["DecisionTree",clf_tree], ["SVM",clf_svm], ["Perceptron", clf_perceptron], ["KNN", clf_KNN]]


def train_models(clf, data, label):
    # here we train our model
    # clf is dynamic based on different classifiers passed
    clf.fit(data, label)
    return clf


def clf_predict(clf, data):
    # here we predict using our trained model
    # clf is dynamic based on different classifiers passed
    prediction = clf.predict(data)
    return prediction


def get_accuracy_score(label, prediction):
    # clf is dynamic based on different classifiers passed
    accuracy = accuracy_score(label, prediction) * 100
    return accuracy


def run_classification():
    # The best classifier list
    best_clf = []

    for clf in classifiers:
        # clf = classifiers[0]
        clf_name = clf[0]
        clf = clf[1]
        clf = train_models(clf, X, Y)
        # Testing using the same or different data
        prediction = clf_predict(clf, X_Test)
        accuracy = get_accuracy_score(Y_Test, prediction)
        best_clf.append([clf_name, accuracy])

    return best_clf


def print_best_clf(best_clf):

    for clf_list in best_clf:
        # clf_list = best_clf[0]
        print('Accuracy for {0}: {1}'.format(clf_list[0], clf_list[1]))

    # The best classifier from list
    index = np.argmax([acc[1] for acc in best_clf])
    classifiers = {index:item[0] for index, item in enumerate(best_clf)}
    print('Best gender classifier is {}'.format(classifiers[index]))

    return 0


def main():

    best_clf = run_classification()
    print_best_clf(best_clf)

    return 0


if __name__ == '__main__':
    main()
