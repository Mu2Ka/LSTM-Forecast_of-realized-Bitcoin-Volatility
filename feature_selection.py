"""Последовательный отбор признаков с временной кросс-валидацией."""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def forward_selection(
    X,
    y,
    time_series_split,
    max_features=87,
    min_improvement=0.0001,
):
    n_features = X.shape[1]
    remaining_features = list(range(n_features))
    selected_features = []
    scores = []
    best_overall_score = -np.inf
    prepared_folds = []
    for train_index, valid_index in time_series_split.split(X):
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_train_imputed = imputer.fit_transform(X.iloc[train_index])
        X_valid_imputed = imputer.transform(X.iloc[valid_index])

        X_train_processed = scaler.fit_transform(X_train_imputed)
        X_valid_processed = scaler.transform(X_valid_imputed)

        prepared_folds.append(
            (
                X_train_processed,
                y.iloc[train_index].to_numpy(),
                X_valid_processed,
                y.iloc[valid_index].to_numpy(),
            )
        )

    for step in range(min(n_features, max_features)):
        best_score = -np.inf
        best_feature = None

        for feature in remaining_features:
            current_features = selected_features + [feature]
            fold_scores = []

            for (
                X_train_processed,
                y_train_fold,
                X_valid_processed,
                y_valid_fold,
            ) in prepared_folds:
                model = LinearRegression()
                model.fit(
                    X_train_processed[:, current_features],
                    y_train_fold,
                )
                fold_scores.append(
                    model.score(
                        X_valid_processed[:, current_features],
                        y_valid_fold,
                    )
                )

            score = float(np.mean(fold_scores))
            if score > best_score:
                best_score = score
                best_feature = feature

        if best_score <= best_overall_score + min_improvement:
            print("Отбор остановлен: средний validation R² больше не улучшается")
            break

        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        scores.append(best_score)
        best_overall_score = best_score

        print(
            f"Шаг {step + 1}: "
            f"{X.columns[best_feature]}, "
            f"средний R² = {best_score:.4f}"
        )

    return selected_features, scores
