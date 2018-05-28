import numpy as np

from sklearn.datasets import load_digits
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# Set random seed for reproducibility
np.random.seed(1000)


if __name__ == '__main__':
    X, Y = load_digits(return_X_y=True)
    X /= np.max(X)

    # Test Decision Tree
    dt = DecisionTreeClassifier(criterion='entropy', random_state=1000)
    print('Decision Tree score: {}'.format(np.mean(cross_val_score(dt, X, Y, cv=10))))

    # Test Logistic Regression
    lr = LogisticRegression(C=2.0, random_state=1000)
    print('Logistic Regression score: {}'.format(np.mean(cross_val_score(lr, X, Y, cv=10))))

    # Create a soft voting classifier
    vc = VotingClassifier(estimators=[
        ('LR', LogisticRegression(C=2.0, random_state=1000)),
        ('DT', DecisionTreeClassifier(criterion='entropy', random_state=1000))],
        voting='soft', weights=(0.9, 0.1))

    print('Voting classifier score: {}'.format(np.mean(cross_val_score(vc, X, Y, cv=10))))