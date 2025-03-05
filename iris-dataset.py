#--------------------NN
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

iris = load_iris()
df = pd.DataFrame(np.c_[iris['data'], iris['target']],
                  columns=np.append(iris['feature_names'], ['target']))

X = df.drop(columns=['target'])
y = df['target']

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the standardized dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'hidden_layer_sizes': [(50, 50), (100, 100), (50, 100, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    # TODO: change batch size to 16, 32, 64 for better performance
    # total items = 150
    # train = .8(150) = 120
    # batch = 128 which is > 120
    # must be in increments of powers of 2
    'batch_size': [16, 32, 64],
}

# Define the MLPClassifier model
model = MLPClassifier(random_state=42, max_iter=2000, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10)

# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Make predictions on the testing data
y_pred = best_model.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
# TODO: change average to 'macro' for multiclass classification instead of binary classification
# 'macro': Computes the metric for each class and averages them equally.
# 'weighted': Averages them based on the number of samples in each class.
# 'micro': Computes the metric globally (good for imbalance but loses class-specific info).
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro') 
cm = confusion_matrix(y_test, y_pred)

# Print the best hyperparameters and performance metrics
print("Best NN hyperparameters found:")
print(grid_search.best_params_)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print('Confusion matrix:\n', cm)









#--------------------SVM
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

iris = load_iris()
df = pd.DataFrame(np.c_[iris['data'], iris['target']],
                  columns=np.append(iris['feature_names'], ['target']))

X = df.drop(columns=['target'])
y = df['target']

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameters to tune
hyperparams = {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100], 
                'kernel': ['linear', 'rbf'], 
                'gamma': ['scale', 'auto']}

svc = SVC(probability=True)
# Perform hyperparameter tuning using k-fold cross-validation
grid_search = GridSearchCV(svc, hyperparams, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_svc = grid_search.best_estimator_

# Make predictions on test set
y_pred = best_svc.predict(X_test)
y_prob = best_svc.predict_proba(X_test)[:, 1]

# Calculate performance metrics
acc = accuracy_score(y_test, y_pred)
# TODO: change average to 'macro' for multiclass classification instead of binary classification
# 'macro': Computes the metric for each class and averages them equally.
# 'weighted': Averages them based on the number of samples in each class.
# 'micro': Computes the metric globally (good for imbalance but loses class-specific info).
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
cm = confusion_matrix(y_test, y_pred)

# Print SVM performance metrics
print("Best SVM hyperparameters found:")
print('Accuracy:', acc)
print('Precision:', prec)
print('Recall:', rec)
print('F1 score:', f1)
print('Confusion matrix:\n', cm)
