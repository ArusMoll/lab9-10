import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utilities import visualize_classifier


def main():
    # Завантажуємо дані з файлу
    input_file = 'data_imbalance.txt'
    data = np.loadtxt(input_file, delimiter=',')

    # X – ознаки, y – класи
    X, y = data[:, :-1], data[:, -1]

    # Розділяємо точки двох класів для візуалізації
    class_0 = np.array(X[y == 0])
    class_1 = np.array(X[y == 1])

    # Візуалізація вхідних даних
    plt.figure()
    if class_0.size:
        plt.scatter(class_0[:, 0], class_0[:, 1], s=75,
                    facecolors='black', edgecolors='black',
                    linewidth=1, marker='o')
    if class_1.size:
        plt.scatter(class_1[:, 0], class_1[:, 1], s=75,
                    facecolors='white', edgecolors='black',
                    linewidth=1, marker='o')
    plt.title('Вхідні дані')
    plt.show()

    # Розділяємо вибірку на тренувальну та тестову
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=5
    )

    # Базові параметри Extra Trees
    params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}

    # Обробка аргументів командного рядка
    # "balance" → вмикає class_weight='balanced'
    # "ignore" → ігнорує попередження
    if len(sys.argv) > 1:
        if sys.argv[1] == 'balance':
            params = {'n_estimators': 100, 'max_depth': 4,
                      'random_state': 0, 'class_weight': 'balanced'}
        elif sys.argv[1] == 'ignore':
            import warnings
            warnings.filterwarnings("ignore")
        else:
            raise TypeError("Invalid input argument; should be 'balance' or 'ignore'")

    # Створюємо та навчаємо модель
    classifier = ExtraTreesClassifier(**params)
    classifier.fit(X_train, y_train)

    # Візуалізація меж класифікації на тренувальних даних
    visualize_classifier(classifier, X_train, y_train)

    # Прогноз на тестових даних + візуалізація меж
    y_test_pred = classifier.predict(X_test)
    visualize_classifier(classifier, X_test, y_test)

    # Імена класів для звіту
    class_names = ['Class-0', 'Class-1']

    # Виводимо якість на тренувальних даних
    print("\n" + "#" * 40)
    print("\nClassifier performance on training dataset\n")
    print(classification_report(
        y_train,
        classifier.predict(X_train),
        target_names=class_names
    ))
    print("#" * 40 + "\n")

    # Виводимо якість на тестових даних
    print("\nClassifier performance on test dataset\n")
    print(classification_report(
        y_test,
        y_test_pred,
        target_names=class_names
    ))
    print("#" * 40 + "\n")

    plt.show()


# Точка входу
if __name__ == '__main__':
    main()
