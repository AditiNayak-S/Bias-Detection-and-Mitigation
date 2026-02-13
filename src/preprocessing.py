from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    X = df.drop("income", axis=1)
    y = df["income"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    categorical = X.select_dtypes(include="object").columns
    numerical = X.select_dtypes(exclude="object").columns

    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("num", StandardScaler(), numerical)
    ])

    preprocess.fit(X_train)

    X_train_t = preprocess.transform(X_train)
    X_test_t = preprocess.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_t, X_test_t
