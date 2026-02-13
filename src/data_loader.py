import pandas as pd

def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
    ]

    df = pd.read_csv(url, names=columns, sep=", ", engine="python")
    df = df.dropna()
    df["income"] = df["income"].apply(lambda x: 1 if x == ">50K" else 0)

    return df
