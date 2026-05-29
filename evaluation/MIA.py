import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve

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
        # Create labels: 1 for train (members), 0 for test (non-members)
        y_true = np.concatenate([np.ones_like(tr_values), np.zeros_like(te_values)])
        y_scores = np.concatenate([tr_values, te_values])
        
        # roc_curve sorts the scores and calculates rates in O(N log N) time
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        # tr_ratio is the True Positive Rate (tpr)
        # te_ratio is the True Negative Rate (1 - fpr)
        acc = 0.5 * (tpr + (1 - fpr))
        
        # Find the index of the maximum accuracy and return the corresponding threshold
        max_idx = np.argmax(acc)
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
    def _mem_inf_thre(self, v_name, s_tr_values, s_te_values, t_tr_values, t_te_values):
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
            "For membership inference attack via {n}, the attack acc is {acc1:.3f}, with train acc {acc2:.3f} and test acc {acc3:.3f}".format(
                n=v_name, acc1=mem_inf_acc, acc2=t_tr_acc, acc3=t_te_acc
            )
        )
        return t_tr_acc, t_te_acc


    # gathering MIA accuracies for all methods in one function
    def _mem_inf_benchmarks(self, all_methods=True, benchmark_methods=[]):
        ret = {}
        if (all_methods) or ("correctness" in benchmark_methods):
            ret["correctness"] = dict(
                zip(
                    ["retain_test_member", "forget_non_member"], 
                    self._mem_inf_via_corr()
                )
            )
        if (all_methods) or ("confidence" in benchmark_methods):
            ret["confidence"] = dict(
                zip(
                    ["retain_test_member", "forget_non_member"], 
                    self._mem_inf_thre(
                        "confidence",
                        self.s_tr_conf,
                        self.s_te_conf,
                        self.t_tr_conf,
                        self.t_te_conf,
                    )
                )
            )
        if (all_methods) or ("entropy" in benchmark_methods):
            ret["entropy"] = dict(
                zip(
                    ["retain_test_member", "forget_non_member"], 
                    self._mem_inf_thre(
                        "entropy",
                        -self.s_tr_entr,
                        -self.s_te_entr,
                        -self.t_tr_entr,
                        -self.t_te_entr,
                    )
                )
            )
        if (all_methods) or ("modified entropy" in benchmark_methods):
            ret["m_entropy"] = dict(
                zip(
                    ["retain_test_member", "forget_non_member"], 
                    self._mem_inf_thre(
                        "modified entropy",
                        -self.s_tr_m_entr,
                        -self.s_te_m_entr,
                        -self.t_tr_m_entr,
                        -self.t_te_m_entr,
                    )
                )
            )
        return ret


def output_dists_and_labels(data_loader, model, device):
    probs = []
    labels = []
    model.eval()

    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        with torch.no_grad():
            output = model(data)
            prob = F.softmax(output, dim=-1)

        probs.append(prob)
        labels.append(target)

    return torch.cat(probs).cpu().numpy(), torch.cat(labels).cpu().numpy()


def MIA(
    retain_loader_train, test_loader, retain_loader_test, forget_loader, model, device
):
    shadow_train_performance = output_dists_and_labels(retain_loader_train, model, device)
    shadow_test_performance = output_dists_and_labels(test_loader, model, device)
    target_train_performance = output_dists_and_labels(retain_loader_test, model, device)
    target_test_performance = output_dists_and_labels(forget_loader, model, device)

    BBB = black_box_benchmarks(
        shadow_train_performance,
        shadow_test_performance,
        target_train_performance,
        target_test_performance,
        num_classes=10,
    )
    return BBB._mem_inf_benchmarks()
