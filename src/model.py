from sklearn.linear_model import LogisticRegression

def train_base_model(X_train_t, y_train):
    model = LogisticRegression(max_iter=3000)
    model.fit(X_train_t, y_train)
    return model
