from sklearn.metrics import confusion_matrix
from fairlearn.metrics import MetricFrame, selection_rate
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.linear_model import LogisticRegression


# --------- Custom Fairness Metrics ---------

def false_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn)


def true_positive_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn)


# --------- Fairness Evaluation ---------

def evaluate_fairness(y_true, y_pred, sensitive_feature):
    metric = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "TPR": true_positive_rate,
            "FPR": false_positive_rate
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_feature
    )
    return metric


# --------- Bias Mitigation ---------

def mitigate_bias(X_train_t, y_train, sensitive_feature):
    mitigator = ExponentiatedGradient(
        LogisticRegression(max_iter=3000),
        constraints=DemographicParity()
    )

    mitigator.fit(
        X_train_t,
        y_train,
        sensitive_features=sensitive_feature
    )

    return mitigator
