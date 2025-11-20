import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import preprocessing

def try_parse_number(s):
    try:
        return float(s)
    except Exception:
        return None

def main():
    input_file = 'traffic_data.txt'
    raw = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            items = [it.strip() for it in line.strip().split(',') if it.strip() != '']
            if items:
                raw.append(items)
    data = np.array(raw, dtype=object)

    # Build encoders for non-numeric columns
    label_encoders = []
    X_encoded = np.empty(data.shape, dtype=float)

    for i in range(data.shape[1]):
        col = data[:, i]
        # try parse numeric
        parsed = [try_parse_number(v) for v in col]
        if all(v is not None for v in parsed):
            X_encoded[:, i] = [float(v) for v in col]
        else:
            le = preprocessing.LabelEncoder()
            X_encoded[:, i] = le.fit_transform(col)
            label_encoders.append((i, le))

    # assume last column is target (vehicles count)
    X = X_encoded[:, :-1].astype(int)
    y = X_encoded[:, -1].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

    params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
    regressor = ExtraTreesRegressor(**params)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    print("Mean absolute error:", round(mean_absolute_error(y_test, y_pred), 2))

    # Example single test datapoint (update values to match your file's columns)
    test_datapoint = ['Saturday', '10:20', 'Atlanta', 'no']  # adjust length if necessary
    test_encoded = []
    le_index = 0
    for i, val in enumerate(test_datapoint):
        num = try_parse_number(val)
        if num is not None:
            test_encoded.append(int(num))
        else:
            # find encoder for column i
            found = None
            for col_index, le in label_encoders:
                if col_index == i:
                    found = le
                    break
            if found is None:
                # if encoder not found because column was numeric in training, try numeric fallback
                parsed = try_parse_number(val)
                test_encoded.append(int(parsed) if parsed is not None else 0)
            else:
                try:
                    test_encoded.append(int(found.transform([val])[0]))
                except Exception:
                    # unseen label -> try to append 0
                    test_encoded.append(0)

    test_encoded = np.array(test_encoded).astype(int)
    if test_encoded.shape[0] != X.shape[1]:
        # pad or trim to match
        if test_encoded.shape[0] < X.shape[1]:
            pad = np.zeros(X.shape[1] - test_encoded.shape[0], dtype=int)
            test_encoded = np.concatenate([test_encoded, pad])
        else:
            test_encoded = test_encoded[:X.shape[1]]

    pred = regressor.predict([test_encoded])[0]
    print("Predicted traffic:", int(pred))

if __name__ == '__main__':
    main()
