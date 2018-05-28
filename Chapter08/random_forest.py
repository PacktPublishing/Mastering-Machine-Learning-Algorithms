import matplotlib.pyplot as plt
import numpy as np

from multiprocessing import cpu_count

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    # Load the dataset
    X, Y = load_wine(return_X_y=True)

    # Test Logistic regression
    lr = LogisticRegression(max_iter=1000, random_state=1000)
    print('Logistic Regression CV score: {}'.format(np.mean(cross_val_score(lr, X, Y, cv=10))))

    # Test Decision Tree
    dt = DecisionTreeClassifier(criterion='entropy', random_state=1000)
    print('Decistion Tree CV score: {}'.format(np.mean(cross_val_score(dt, X, Y, cv=10))))

    # Test Polynomial SVM
    svm = SVC(kernel='poly', random_state=1000)
    print('Polynomial SVM CV score: {}'.format(np.mean(cross_val_score(svm, X, Y, cv=10))))

    # Test Random Forest
    rf = RandomForestClassifier(n_estimators=50, n_jobs=cpu_count(), random_state=1000)
    scores = cross_val_score(rf, X, Y, cv=10)
    print('Random Forest CV score: {}'.format(np.mean(scores)))

    # Plot CV scores
    fig, ax = plt.subplots(figsize=(15, 7))

    ax.plot(scores)
    ax.set_xlabel('Number of Trees (x10)')
    ax.set_ylabel('10-fold Cross-Validation Accuracy')
    ax.grid()
    plt.show()

    # Show feature importances
    rf.fit(X, Y)

    wine = load_wine()
    features = [wine['feature_names'][x] for x in np.argsort(rf.feature_importances_)][::-1]

    fig, ax = plt.subplots(figsize=(15, 8))

    ax.bar([i for i in range(13)], np.sort(rf.feature_importances_)[::-1], align='center')
    ax.set_ylabel('Feature Importance')
    plt.xticks([i for i in range(13)], features, rotation=60)
    plt.show()

    # Select the most important features
    sfm = SelectFromModel(estimator=rf, prefit=True, threshold=0.02)
    X_sfm = sfm.transform(X)

    print('Feature selection shape: {}'.format(X_sfm.shape))

