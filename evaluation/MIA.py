import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, confusion_matrix
from evaluation.entropy import entropy

class black_box_benchmarks(object):
    def __init__(
        self,
        shadow_train_performance,
        shadow_test_performance,
        target_train_performance,
        target_test_performance,
        num_classes,
    ):
        """
        each input contains both model predictions (shape: num_data*num_classes) and ground-truth labels.
        """
        self.num_classes = num_classes

        # separate the output distributions and labels for each shadow (retain + test) and target models (retain_test + forget)
        self.s_tr_outputs, self.s_tr_labels = shadow_train_performance
        self.s_te_outputs, self.s_te_labels = shadow_test_performance
        self.t_tr_outputs, self.t_tr_labels = target_train_performance
        self.t_te_outputs, self.t_te_labels = target_test_performance

        # whether or not the max output equals the correct label
        self.s_tr_corr = (
            np.argmax(self.s_tr_outputs, axis=1) == self.s_tr_labels
        ).astype(int)
        self.s_te_corr = (
            np.argmax(self.s_te_outputs, axis=1) == self.s_te_labels
        ).astype(int)
        self.t_tr_corr = (
            np.argmax(self.t_tr_outputs, axis=1) == self.t_tr_labels
        ).astype(int)
        self.t_te_corr = (
            np.argmax(self.t_te_outputs, axis=1) == self.t_te_labels
        ).astype(int)

        # the confidence of the predicted label
        self.s_tr_conf = np.take_along_axis(
            self.s_tr_outputs, self.s_tr_labels[:, None], axis=1
        )
        self.s_te_conf = np.take_along_axis(
            self.s_te_outputs, self.s_te_labels[:, None], axis=1
        )
        self.t_tr_conf = np.take_along_axis(
            self.t_tr_outputs, self.t_tr_labels[:, None], axis=1
        )
        self.t_te_conf = np.take_along_axis(
            self.t_te_outputs, self.t_te_labels[:, None], axis=1
        )

        # the entropy of the output distribution
        self.s_tr_entr = self._entr_comp(self.s_tr_outputs)
        self.s_te_entr = self._entr_comp(self.s_te_outputs)
        self.t_tr_entr = self._entr_comp(self.t_tr_outputs)
        self.t_te_entr = self._entr_comp(self.t_te_outputs)

        # the modified entropy of the output distribution
        self.s_tr_m_entr = self._m_entr_comp(self.s_tr_outputs, self.s_tr_labels)
        self.s_te_m_entr = self._m_entr_comp(self.s_te_outputs, self.s_te_labels)
        self.t_tr_m_entr = self._m_entr_comp(self.t_tr_outputs, self.t_tr_labels)
        self.t_te_m_entr = self._m_entr_comp(self.t_te_outputs, self.t_te_labels)

    # a handy function to prevent numerical underflow
    def _log_value(self, probs, eps=1e-30):
        return -np.log(np.maximum(probs, eps))

    # helper function for calculating entropy of the output distribution
    def _entr_comp(self, probs):
        return np.sum(np.multiply(probs, self._log_value(probs)), axis=1)

    # helper function for calculating modified entropy
    def _m_entr_comp(self, probs, true_labels):
        log_probs = self._log_value(probs)
        reverse_probs = 1 - probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels] = reverse_probs[
            range(true_labels.size), true_labels
        ]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels] = log_probs[
            range(true_labels.size), true_labels
        ]
        return np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)


    # here is where we actually calculate the threshold value which maximizes accuracy on the shadow inputs - this is akin to "training" our attack model
    # def _thre_setting(self, tr_values, te_values):
    #     value_list = np.concatenate((tr_values, te_values))
    #     thre, max_acc = 0, 0
    #     for value in value_list:
    #         tr_ratio = np.sum(tr_values >= value) / (len(tr_values) + 0.0)
    #         te_ratio = np.sum(te_values < value) / (len(te_values) + 0.0)
    #         acc = 0.5 * (tr_ratio + te_ratio)
    #         if acc > max_acc:
    #             thre, max_acc = value, acc
    #     return thre

    # SUPPOSEDLY FASTER, EQUIVALENT VERSION OF THE ABOVE
    def _thre_setting(self, tr_values, te_values):
        # If no member samples exist for this class (e.g. class-based unlearning where the
        # forgotten class is absent from retain), we cannot calibrate a threshold. Return -inf
        # so all samples of this class are treated as members (the conservative default).F
        # Should only kick in on class forgetting
        if len(tr_values.flatten()) == 0 or len(te_values.flatten()) == 0:
            return -np.inf

        # Create labels: 1 for train (members), 0 for test (non-members)
        y_true = np.concatenate([np.ones_like(tr_values), np.zeros_like(te_values)])
        y_scores = np.concatenate([tr_values, te_values])
        
        # roc_curve sorts the scores and calculates rates in O(N log N) time
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        # tr_ratio is the True Positive Rate (tpr)
        # te_ratio is the True Negative Rate (1 - fpr)
        acc = 0.5 * (tpr + (1 - fpr))
        
        # Find the index of the maximum accuracy and return the corresponding threshold
        # max_idx = np.argmax(acc)
        
        # Skip index 0: roc_curve prepends a sentinel (threshold=max+1, tpr=0, fpr=0, acc=0.5)
        # which np.argmax would return whenever no real threshold achieves acc > 0.5
        max_idx = np.argmax(acc[1:]) + 1
        
        return thresholds[max_idx]

    # perform membership inference attack based on whether the input is correctly classified or not
    # you dont actually have to train a threshold value for this
    def _mem_inf_via_corr(self):
        t_tr_acc = np.sum(self.t_tr_corr) / (len(self.t_tr_corr) + 0.0)
        t_te_acc = 1 - np.sum(self.t_te_corr) / (len(self.t_te_corr) + 0.0)
        mem_inf_acc = 0.5 * (t_tr_acc + t_te_acc)
        print(
            "For membership inference attack via correctness, the attack acc is {acc1:.3f}, with train acc {acc2:.3f} and test acc {acc3:.3f}".format(
                acc1=mem_inf_acc, acc2=t_tr_acc, acc3=t_te_acc
            )
        )
        return t_tr_acc, t_te_acc

    # perform membership inference attack by thresholding feature values 
    # --- this picks a different threshold for each class
    # the feature can be prediction confidence, (negative) prediction entropy, and (negative) modified entropy
    def _mem_inf_thre_class(self, v_name, s_tr_values, s_te_values, t_tr_values, t_te_values):
        t_tr_mem, t_te_non_mem = 0, 0
        for num in range(self.num_classes):
            thre = self._thre_setting(
                s_tr_values[self.s_tr_labels == num],
                s_te_values[self.s_te_labels == num],
            )
            t_tr_mem += np.sum(t_tr_values[self.t_tr_labels == num] >= thre)
            t_te_non_mem += np.sum(t_te_values[self.t_te_labels == num] < thre)
        t_tr_acc = t_tr_mem / (len(self.t_tr_labels) + 0.0)
        t_te_acc = t_te_non_mem / (len(self.t_te_labels) + 0.0)
        mem_inf_acc = 0.5 * (t_tr_acc + t_te_acc)
        print(
            "For MIA via {n}, with different thresholds per class: the attack acc is {acc1:.3f}, with train acc {acc2:.3f} and test acc {acc3:.3f}".format(
                n=v_name, acc1=mem_inf_acc, acc2=t_tr_acc, acc3=t_te_acc
            )
        )
        return t_tr_acc, t_te_acc
    
    # picks one threshold for ALL classes
    def _mem_inf_thre_no_class(self, v_name, s_tr_values, s_te_values, t_tr_values, t_te_values):
        
        t_tr_mem, t_te_non_mem = 0, 0
        
        thre = self._thre_setting(s_tr_values, s_te_values)
        t_tr_mem += np.sum(t_tr_values >= thre)
        t_te_non_mem += np.sum(t_te_values < thre)
        
        t_tr_acc = t_tr_mem / (len(self.t_tr_labels) + 0.0)
        t_te_acc = t_te_non_mem / (len(self.t_te_labels) + 0.0)
        mem_inf_acc = 0.5 * (t_tr_acc + t_te_acc)
        print(
            "For MIA via {n}, with one threshold across all classes: the attack acc is {acc1:.3f}, with train acc {acc2:.3f} and test acc {acc3:.3f}".format(
                n=v_name, acc1=mem_inf_acc, acc2=t_tr_acc, acc3=t_te_acc
            )
        )
        return t_tr_acc, t_te_acc


    # gathering MIA accuracies for all methods in one function
    def _mem_inf_benchmarks(self, all_methods=True, benchmark_methods=[]):

        # preinitialize sub-dictinaries
        ret = {"class": {}, "no_class": {}}

        if (all_methods) or ("correctness" in benchmark_methods):
            ret["correctness"] = dict(
                zip(
                    ["retain_test_member", "forget_non_member"], 
                    self._mem_inf_via_corr()
                )
            )
        if (all_methods) or ("confidence" in benchmark_methods):
            ret["class"]["confidence"] = dict(
                zip(
                    ["retain_test_member", "forget_non_member"], 
                    self._mem_inf_thre_class(
                        "confidence",
                        self.s_tr_conf,
                        self.s_te_conf,
                        self.t_tr_conf,
                        self.t_te_conf,
                    )
                )
            )
            ret["no_class"]["confidence"] = dict(
                zip(
                    ["retain_test_member", "forget_non_member"], 
                    self._mem_inf_thre_no_class(
                        "confidence",
                        self.s_tr_conf,
                        self.s_te_conf,
                        self.t_tr_conf,
                        self.t_te_conf,
                    )
                )
            )
        if (all_methods) or ("entropy" in benchmark_methods):
            ret['class']["entropy"] = dict(
                zip(
                    ["retain_test_member", "forget_non_member"], 
                    self._mem_inf_thre_class(
                        "entropy",
                        -self.s_tr_entr,
                        -self.s_te_entr,
                        -self.t_tr_entr,
                        -self.t_te_entr,
                    )
                )
            )
            ret["no_class"]["entropy"] = dict(
                zip(
                    ["retain_test_member", "forget_non_member"], 
                    self._mem_inf_thre_no_class(
                        "entropy",
                        -self.s_tr_entr,
                        -self.s_te_entr,
                        -self.t_tr_entr,
                        -self.t_te_entr,
                    )
                )
            )
        if (all_methods) or ("modified entropy" in benchmark_methods):
            ret["class"]["m_entropy"] = dict(
                zip(
                    ["retain_test_member", "forget_non_member"], 
                    self._mem_inf_thre_class(
                        "modified entropy",
                        -self.s_tr_m_entr,
                        -self.s_te_m_entr,
                        -self.t_tr_m_entr,
                        -self.t_te_m_entr,
                    )
                )
            )
            ret["no_class"]["m_entropy"] = dict(
                zip(
                    ["retain_test_member", "forget_non_member"], 
                    self._mem_inf_thre_no_class(
                        "modified entropy",
                        -self.s_tr_m_entr,
                        -self.s_te_m_entr,
                        -self.t_tr_m_entr,
                        -self.t_te_m_entr,
                    )
                )
            )

        # we insert a print new line here, since the above functions print a bunch of text, and we want to separate from the next block (whatever it is)
        print("\n")
        return ret


# def output_dists_and_labels(data_loader, model, device):
#     probs = []
#     labels = []
#     model.eval()

#     for data, target in data_loader:
#         data = data.to(device)
#         target = target.to(device)
#         with torch.no_grad():
#             output = model(data)
#             prob = F.softmax(output, dim=-1)

#         probs.append(prob)
#         labels.append(target)

#     return torch.cat(probs).cpu().numpy(), torch.cat(labels).cpu().numpy()


def MIA(
    shadow_train_inputs, shadow_test_inputs, target_train_inputs, target_test_inputs, model, device
):
    # shadow_train_performance = output_dists_and_labels(shadow_train, model, device)
    # shadow_test_performance = output_dists_and_labels(shadow_test, model, device)
    # target_train_performance = output_dists_and_labels(target_train, model, device)
    # target_test_performance = output_dists_and_labels(target_test, model, device)

    BBB = black_box_benchmarks(
        shadow_train_inputs,
        shadow_test_inputs,
        target_train_inputs,
        target_test_inputs,
        num_classes=10,
    )
    return BBB._mem_inf_benchmarks()


def entropy_threshold_MIA(train_member_probs, train_non_member_probs, forget_probs, audit_non_member_probs):
    """
    Threshold-based MIA using (negative) prediction entropy as the sole attack feature.

    A single global threshold is calibrated on `train_member_probs`/`train_non_member_probs`
    (e.g. subsets of retain/test). The attack is then audited against `forget_probs` (true
    members, label 1) vs `audit_non_member_probs` (true non-members, label 0) - a successful
    unlearning should make the attack fail on the forget set.
    """
    train_member_score = -entropy(train_member_probs).detach().cpu().numpy()
    train_non_member_score = -entropy(train_non_member_probs).detach().cpu().numpy()
    forget_score = -entropy(forget_probs).detach().cpu().numpy()
    audit_non_member_score = -entropy(audit_non_member_probs).detach().cpu().numpy()

    # calibrate the threshold that maximizes balanced accuracy on the training split
    y_train = np.concatenate([np.ones_like(train_member_score), np.zeros_like(train_non_member_score)])
    scores_train = np.concatenate([train_member_score, train_non_member_score])
    fpr, tpr, thresholds = roc_curve(y_train, scores_train)
    balanced_acc = 0.5 * (tpr + (1 - fpr))
    # skip index 0: roc_curve prepends a sentinel threshold that argmax would return by default
    thre = thresholds[np.argmax(balanced_acc[1:]) + 1]

    y_audit = np.concatenate([np.ones_like(forget_score), np.zeros_like(audit_non_member_score)])
    audit_scores = np.concatenate([forget_score, audit_non_member_score])
    audit_preds = (audit_scores >= thre).astype(int)

    # min-max normalize scores against the training split's range, to use as a pseudo membership-probability
    score_min, score_max = scores_train.min(), scores_train.max()
    audit_member_prob = np.clip((audit_scores - score_min) / (score_max - score_min + 1e-12), 0, 1)

    n_forget = len(forget_score)
    forget_preds = audit_preds[:n_forget]
    forget_member_prob = audit_member_prob[:n_forget]

    tn, fp, fn, tp = confusion_matrix(y_audit, audit_preds, labels=[0, 1]).ravel()

    return {
        "efficacy": np.mean(forget_preds == 0).item(),
        "attack_accuracy": np.mean(audit_preds == y_audit).item(),
        "auc": roc_auc_score(y_audit, audit_scores),
        "f1": f1_score(y_audit, audit_preds),
        "avg_member_probability_forget": np.mean(forget_member_prob).item(),
        "n_forget_pred_member": int(np.sum(forget_preds == 1)),
        "n_forget_pred_non_member": int(np.sum(forget_preds == 0)),
        "TP": (tp / (tp + fn)).item() if (tp + fn) > 0 else 0.0,
        "TN": (tn / (tn + fp)).item() if (tn + fp) > 0 else 0.0,
        "FP": (fp / (fp + tn)).item() if (fp + tn) > 0 else 0.0,
        "FN": (fn / (fn + tp)).item() if (fn + tp) > 0 else 0.0,
    }
