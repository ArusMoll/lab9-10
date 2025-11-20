import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from utilities import visualize_classifier


classifier_type = "rf"



input_file = 'data_random_forests.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

class_0 = np.array(X[y == 0])
class_1 = np.array(X[y == 1])
class_2 = np.array(X[y == 2])

plt.figure()
if class_0.size:
    plt.scatter(class_0[:, 0], class_0[:, 1], s=75,
                facecolors='white', edgecolors='black',
                linewidth=1, marker='s')
if class_1.size:
    plt.scatter(class_1[:, 0], class_1[:, 1], s=75,
                facecolors='white', edgecolors='black',
                linewidth=1, marker='o')
if class_2.size:
    plt.scatter(class_2[:, 0], class_2[:, 1], s=75,
                facecolors='white', edgecolors='black',
                linewidth=1, marker='^')
plt.title('Вхідні дані')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5
)

params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}

if classifier_type == "rf":
    classifier = RandomForestClassifier(**params)
else:
    classifier = ExtraTreesClassifier(**params)

classifier.fit(X_train, y_train)
visualize_classifier(classifier, X_train, y_train)

y_test_pred = classifier.predict(X_test)
visualize_classifier(classifier, X_test, y_test)

class_names = ["Class-0", "Class-1", "Class-2"]
print("\n" + "#" * 40)
print("\nTraining report\n")
print(classification_report(y_train, classifier.predict(X_train),
                            target_names=class_names))
print("#" * 40 + "\n")

print("\nTest report\n")
print(classification_report(y_test, y_test_pred,
                            target_names=class_names))
print("#" * 40 + "\n")

# Confidence for 6 points
test_datapoints = np.array([[5, 5], [3, 6], [6, 4],
                            [7, 2], [4, 4], [5, 2]])

print("\nConfidence measure:")
for datapoint in test_datapoints:
    probs = classifier.predict_proba([datapoint])[0]
    pred_class = "Class-" + str(np.argmax(probs))
    print("\nDatapoint:", datapoint)
    print("Predicted class:", pred_class)
    print("Probabilities:", probs)

visualize_classifier(classifier, test_datapoints, np.zeros(len(test_datapoints)))
plt.show()
