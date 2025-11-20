import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def load_boston_compat():
    try:
        # scikit-learn older versions
        return datasets.load_boston()
    except Exception:
        # try openml fallback
        try:
            from sklearn.datasets import fetch_openml
            boston = fetch_openml(name='boston', version=1, as_frame=False)
            data = boston['data']
            target = boston['target'].astype(float)
            class B:
                pass
            b = B()
            b.data = data
            b.target = target
            b.feature_names = np.array(boston['feature_names'])
            return b
        except Exception:
            # fallback to california housing (structure differs)
            ch = datasets.fetch_california_housing()
            class C:
                pass
            c = C()
            c.data = ch.data
            c.target = ch.target
            c.feature_names = ch.feature_names
            return c

def main():
    housing_data = load_boston_compat()
    X, y = shuffle(housing_data.data, housing_data.target, random_state=7)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    regressor = AdaBoostRegressor(
        DecisionTreeRegressor(max_depth=4),
        n_estimators=400, random_state=7)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)

    print("\nADABOOST REGRESSOR")
    print("Mean squared error =", round(mse, 2))
    print("Explained variance score =", round(evs, 2))

    feature_importances = regressor.feature_importances_
    feature_names = getattr(housing_data, 'feature_names', np.arange(len(feature_importances)))

    feature_importances = 100.0 * (feature_importances / max(feature_importances))
    index_sorted = np.flipud(np.argsort(feature_importances))
    pos = np.arange(index_sorted.shape[0]) + 0.5

    plt.bar(pos, feature_importances[index_sorted])
    plt.xticks(pos, feature_names[index_sorted], rotation=90)
    plt.ylabel('Relative importance')
    plt.title('Feature importances (AdaBoost)')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
