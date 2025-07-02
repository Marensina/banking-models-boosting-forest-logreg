import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (make_scorer, f1_score, precision_score, recall_score, accuracy_score,
                             confusion_matrix, classification_report, roc_curve, auc, mean_absolute_error)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb
from scipy.stats import chi2_contingency

# Load data
df_train = pd.read_csv('data/Train.csv')

X = df_train.drop(['bank_account'], axis=1)
y = df_train['bank_account']

# Split dataset stratified by target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# Encode target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Cramér's V function for categorical correlation
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

# Calculate Cramér's V for categorical features
categorical_features = X_train.select_dtypes(include=['object']).columns
print("Cramér's V matrix for categorical features:")
cramers_results = pd.DataFrame(index=categorical_features, columns=categorical_features)
for col1 in categorical_features:
    for col2 in categorical_features:
        cramers_results.loc[col1, col2] = cramers_v(X_train[col1], X_train[col2])
print(cramers_results.astype(float))

# Preprocessing pipelines for numeric and categorical data
numeric_features = X_train.select_dtypes(include=['int', 'float']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline([
    ('scaler', MinMaxScaler())
])

categorical_transformer = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Confusion matrix plot helper
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Generalized training & evaluation function
def train_and_evaluate_model(model_name, classifier, param_grid):
    print(f"\n--- {model_name} ---")

    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', classifier)
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train_encoded)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test_encoded, y_pred)
    precision = precision_score(y_test_encoded, y_pred)
    recall = recall_score(y_test_encoded, y_pred)
    f1 = f1_score(y_test_encoded, y_pred)
    confusion = confusion_matrix(y_test_encoded, y_pred)
    mae = mean_absolute_error(y_test_encoded, y_pred)

    print("Best parameters found:")
    print(grid_search.best_params_)
    print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")
    print("\nTest set evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Confusion Matrix:\n{confusion}")
    print("\nClassification Report:")
    print(classification_report(y_test_encoded, y_pred, target_names=['No', 'Yes']))

    plot_confusion_matrix(confusion, classes=['No', 'Yes'], title=f'{model_name} Confusion matrix')
    plot_confusion_matrix(confusion, classes=['No', 'Yes'], normalize=True, title=f'{model_name} Normalized confusion matrix')

    # ROC Curve plot if classifier supports predict_proba
    clf = best_model.named_steps['classifier']
    if hasattr(clf, "predict_proba"):
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test_encoded, clf.predict_proba(best_model.named_steps['preprocessor'].transform(X_test))[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} ROC Curve')
        plt.legend(loc='lower right')
        plt.show()

    # Feature importance if available
    if hasattr(clf, 'feature_importances_'):
        feat_imp = clf.feature_importances_
        cat_features_ohe = best_model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features)
        all_features = list(numeric_features) + list(cat_features_ohe)
        feat_imp_df = pd.DataFrame({'Feature': all_features, 'Importance': feat_imp}).sort_values(by='Importance', ascending=False)
        print(f"\n{model_name} Feature Importances:")
        print(feat_imp_df)
    else:
        print(f"{model_name} feature importances not available.")

    return best_model

# Models & parameter grids

models_params = {
    'Logistic Regression': (
        LogisticRegression(random_state=42, max_iter=1000),
        {'classifier__C': [0.01, 0.1, 1, 10],
         'classifier__penalty': ['l2'],
         'classifier__solver': ['lbfgs']}
    ),

    'AdaBoost': (
        AdaBoostClassifier(random_state=42),
        {'classifier__n_estimators': [50, 100, 200],
         'classifier__learning_rate': [0.01, 0.1, 1.0]}
    ),

    'XGBoost': (
        xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        {'classifier__n_estimators': [50, 100, 200],
         'classifier__learning_rate': [0.01, 0.1, 1.0],
         'classifier__max_depth': [3, 5, 7]}
    ),

    'Gradient Boosting': (
        GradientBoostingClassifier(random_state=42),
        {'classifier__n_estimators': [50, 100, 200],
         'classifier__learning_rate': [0.01, 0.1, 1.0],
         'classifier__max_depth': [3, 5, 7]}
    ),

    'Random Forest': (
        RandomForestClassifier(random_state=42),
        {'classifier__n_estimators': [100, 200, 300],
         'classifier__max_depth': [None, 10, 20, 30],
         'classifier__min_samples_split': [2, 5, 10],
         'classifier__min_samples_leaf': [1, 2, 4]}
    )
}

# Run all models
best_models = {}
for name, (model, params) in models_params.items():
    best_models[name] = train_and_evaluate_model(name, model, params)
