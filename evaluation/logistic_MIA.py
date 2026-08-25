import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

"""
MIA implementation using a logistic regression classifier trained directly on the model's
full output probability distribution (rather than a single scalar feature).
"""


def logistic_regression_MIA(train_member_probs, train_non_member_probs, forget_probs, audit_non_member_probs):
    """
    A logistic regression attack classifier is fit on `train_member_probs`/`train_non_member_probs`
    (e.g. subsets of retain/test), using the full softmax output distribution as its features.

    The attack is then audited against `forget_probs` (true members, label 1) vs
    `audit_non_member_probs` (true non-members, label 0) - a successful unlearning should make
    the attack fail on the forget set.
    """
    X_train_member = train_member_probs.detach().cpu().numpy()
    X_train_non_member = train_non_member_probs.detach().cpu().numpy()
    X_train = np.concatenate([X_train_member, X_train_non_member])
    y_train = np.concatenate([np.ones(len(X_train_member)), np.zeros(len(X_train_non_member))])

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    X_forget = forget_probs.detach().cpu().numpy()
    X_audit_non_member = audit_non_member_probs.detach().cpu().numpy()
    X_audit = np.concatenate([X_forget, X_audit_non_member])
    y_audit = np.concatenate([np.ones(len(X_forget)), np.zeros(len(X_audit_non_member))])

    audit_member_prob = clf.predict_proba(X_audit)[:, 1]
    audit_preds = clf.predict(X_audit)

    n_forget = len(X_forget)
    forget_preds = audit_preds[:n_forget]
    forget_member_prob = audit_member_prob[:n_forget]

    tn, fp, fn, tp = confusion_matrix(y_audit, audit_preds, labels=[0, 1]).ravel()

    return {
        "efficacy": np.mean(forget_preds == 0),
        "attack_accuracy": np.mean(audit_preds == y_audit),
        "auc": roc_auc_score(y_audit, audit_member_prob),
        "f1": f1_score(y_audit, audit_preds),
        "avg_member_probability_forget": np.mean(forget_member_prob),
        "n_forget_pred_member": int(np.sum(forget_preds == 1)),
        "n_forget_pred_non_member": int(np.sum(forget_preds == 0)),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
    }
