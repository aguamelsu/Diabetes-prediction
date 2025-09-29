import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


def load_data():
    df = pd.read_csv('/workspaces/Diabetes-prediction/data/processed/diabetes_clean.csv')
    return df


def clean_data(df):
    return df


def features(df_train_or_test):
    X = df_train_or_test.drop(columns=['outcome'])
    y = df_train_or_test['outcome']

    
    return X, y


def split_data(df):
    # Split data into train and test sets
    df_train, df_test = train_test_split(df, test_size=0.1, stratify=df['outcome'], random_state=2025)

    # Reset index
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    X_train, y_train = features(df_train)
    X_test, y_test = features(df_test)

    # Encode outcome labels to numerical values
    encoder = LabelEncoder()

    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test)

    return X_train, X_test, y_train_enc, y_test_enc


def train_tuned_model(X_train, y_train):
    tree_clf = DecisionTreeClassifier(random_state=2025)

    param_grid = {
        "max_depth": [1, 2, 3, 4, 5, 6, 7],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 2, 5],
        "criterion": ["gini", "entropy"]
    }

    tree_cv = GridSearchCV(
        tree_clf,
        param_grid=param_grid,
        scoring="recall", 
        cv=5
    )

    tree_cv.fit(X_train, y_train)
    
    return tree_cv, tree_cv.best_params_


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print(f'Accuracy: {accuracy:.3f}')
    print('Classification Report:')
    print(class_report)


def main():
    # Load and preprocess data
    df = load_data()
    df = clean_data(df)

    # Feature engineering and data splitting
    X_train, X_test, y_train, y_test = split_data(df)

    # Hyperparameter tuning
    model, best_params = train_tuned_model(X_train, y_train)
    print(f'Best Parameters: {best_params}')
    
    # Model evaluation
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
