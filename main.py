from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model import train_base_model
from src.fairness import evaluate_fairness, mitigate_bias

from sklearn.metrics import accuracy_score


def main():

    print("Loading dataset...")
    df = load_data()

    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, X_train_t, X_test_t = preprocess_data(df)

    print("Training base model...")
    model = train_base_model(X_train_t, y_train)
    preds = model.predict(X_test_t)

    print("\n=== MODEL PERFORMANCE ===")
    print("Accuracy Before Mitigation:", accuracy_score(y_test, preds))

    print("\n=== GENDER FAIRNESS (Before Mitigation) ===")
    fairness_before = evaluate_fairness(y_test, preds, X_test["sex"])
    print(fairness_before.by_group)

    print("\nApplying Bias Mitigation...")
    mitigator = mitigate_bias(X_train_t, y_train, X_train["sex"])
    new_preds = mitigator.predict(X_test_t)

    print("\nAccuracy After Mitigation:", accuracy_score(y_test, new_preds))

    print("\n=== GENDER FAIRNESS (After Mitigation) ===")
    fairness_after = evaluate_fairness(y_test, new_preds, X_test["sex"])
    print(fairness_after.by_group)


if __name__ == "__main__":
    main()
