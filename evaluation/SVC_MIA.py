import numpy as np
import torch
import torch.nn.functional as F
# from imagenet import get_x_y_from_data_dict
from sklearn.svm import SVC
from evaluation.entropy import entropy, m_entropy

import warnings


"""
MIA implementations using a support vector machine

pulls the same attack features as the threshold-function MIA (corrrectness, confidence, entropy, m-entropy)

By default, SVC is done with RBK kernel, c = 3 (might need to be tuned)

"""

# def collect_prob(data_loader, model):
    
#     # if data_loader is None:
#     #     return torch.zeros([0, 10]), torch.zeros([0])

#     prob = []
#     targets = []

#     model.eval()
#     with torch.no_grad():
#         for batch in data_loader:
#             # try:
            
#             # send to the appropriate device
#             batch = [tensor.to(next(model.parameters()).device) for tensor in batch]
#             data, target = batch

#             # except:
#             #     device = (
#             #         torch.device("cuda:0")
#             #         if torch.cuda.is_available()
#             #         else torch.device("cpu")
#             #     )
#             #     data, target = get_x_y_from_data_dict(batch, device)
            
#             # with torch.no_grad():

#             output = model(data)
#             prob.append( F.softmax(output, dim=-1).detach().cpu())
#             targets.append(target)

#     return torch.cat(prob), torch.cat(targets)


def SVC_fit_predict(shadow_train, shadow_test, target_train, target_test):
    n_shadow_train = shadow_train.shape[0]
    n_shadow_test = shadow_test.shape[0]
    n_target_train = target_train.shape[0]
    n_target_test = target_test.shape[0]

    X_shadow = (
        torch.cat([shadow_train, shadow_test])
        .cpu()
        .numpy()
        .reshape(n_shadow_train + n_shadow_test, -1)
    )
    Y_shadow = np.concatenate([np.ones(n_shadow_train), np.zeros(n_shadow_test)])

    clf = SVC(C=3, gamma="auto", kernel="rbf")
    clf.fit(X_shadow, Y_shadow)

    accs = {}

    if n_target_train > 0:
        X_target_train = target_train.cpu().numpy().reshape(n_target_train, -1)
        acc_train = clf.predict(X_target_train).mean()
        accs.update({"train_member": acc_train})

    if n_target_test > 0:
        X_target_test = target_test.cpu().numpy().reshape(n_target_test, -1)
        acc_test = 1 - clf.predict(X_target_test).mean()
        accs.update({"test_non_member": acc_test})

    return accs


def SVC_MIA(shadow_train_inputs, shadow_test_inputs, target_train_inputs, target_test_inputs, model):
    """
    The inputs here are the probability values

    each is a tuple, (probs, labels)
    """
    shadow_train_probs, shadow_train_labels = shadow_train_inputs
    shadow_test_probs, shadow_test_labels = shadow_test_inputs

    target_train_probs, target_train_labels = target_train_inputs
    target_test_probs, target_test_labels = target_test_inputs

    shadow_train_corr = (
        torch.argmax(shadow_train_probs, axis=1) == shadow_train_labels
    ).int()
    shadow_test_corr = (
        torch.argmax(shadow_test_probs, axis=1) == shadow_test_labels
    ).int()
    target_train_corr = (
        torch.argmax(target_train_probs, axis=1) == target_train_labels
    ).int()
    target_test_corr = (
        torch.argmax(target_test_probs, axis=1) == target_test_labels
    ).int()

    shadow_train_conf = torch.gather(shadow_train_probs, 1, shadow_train_labels[:, None])
    shadow_test_conf = torch.gather(shadow_test_probs, 1, shadow_test_labels[:, None])
    target_train_conf = torch.gather(target_train_probs, 1, target_train_labels[:, None])
    target_test_conf = torch.gather(target_test_probs, 1, target_test_labels[:, None])

    shadow_train_entr = entropy(shadow_train_probs)
    shadow_test_entr = entropy(shadow_test_probs)
    target_train_entr = entropy(target_train_probs)
    target_test_entr = entropy(target_test_probs)

    shadow_train_m_entr = m_entropy(shadow_train_probs, shadow_train_labels)
    shadow_test_m_entr = m_entropy(shadow_test_probs, shadow_test_labels)
    target_train_m_entr = m_entropy(target_train_probs, target_train_labels)
    target_test_m_entr = m_entropy(target_test_probs, target_test_labels)
    # if target_train is not None:
        
    # else:
    #     warnings.warn("`m-entropy` being explicitly set to `entropy` for target_train")
    #     target_train_m_entr = target_train_entr
    # if target_test is not None:
        
    # else:
    #     warnings.warn("`m-entropy` being explicitly set to `entropy` for target_test")
    #     target_test_m_entr = target_test_entr

    acc_corr = SVC_fit_predict(shadow_train_corr, shadow_test_corr, target_train_corr, target_test_corr)
    print(f"Correctess: Train_member = {acc_corr['train_member']:.2f}, Test_non_member = {acc_corr['test_non_member']:.2f}")

    acc_conf = SVC_fit_predict(shadow_train_conf, shadow_test_conf, target_train_conf, target_test_conf)
    print(f"Confidence: Train_member = {acc_corr['train_member']:.2f}, Test_non_member = {acc_corr['test_non_member']:.2f}")
    
    acc_entr = SVC_fit_predict(shadow_train_entr, shadow_test_entr, target_train_entr, target_test_entr)
    print(f"Entropy: Train_member = {acc_corr['train_member']:.2f}, Test_non_member = {acc_corr['test_non_member']:.2f}")
    
    acc_m_entr = SVC_fit_predict(shadow_train_m_entr, shadow_test_m_entr, target_train_m_entr, target_test_m_entr)
    print(f"M-Entropy: Train_member = {acc_corr['train_member']:.2f}, Test_non_member = {acc_corr['test_non_member']:.2f}")
    
    acc_prob = SVC_fit_predict(shadow_train_probs, shadow_test_probs, target_train_probs, target_test_probs)
    print(f"Full Probability Dist: Train_member = {acc_corr['train_member']:.2f}, Test_non_member = {acc_corr['test_non_member']:.2f}")
    
    m = {
        "correctness": acc_corr,
        "confidence": acc_conf,
        "entropy": acc_entr,
        "m_entropy": acc_m_entr,
        "prob": acc_prob,
    }
    
    return m
