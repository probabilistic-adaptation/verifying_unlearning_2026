import torch


def fisher_information_bound(w_o, sigma_o, retrained_model, retrained_fisher_diag):
    """
    Upper bound on the amount of information remaining about the forgotten data after unlearning
    (Golatkar et al. 2020a, "Eternal Sunshine of the Spotless Net"), using the full Gaussian KL
    divergence between the unlearned model's weight distribution N(w_o, Sigma_o) and the
    hypothetical retrain-from-scratch weight distribution N(w_r, Sigma_r):

        KL(N(w_o, Sigma_o) || N(w_r, Sigma_r)) =
            1/2 ( tr(Sigma_r^-1 Sigma_o) + (w_r - w_o)^T Sigma_r^-1 (w_r - w_o)
                  - k + log(|Sigma_r| / |Sigma_o|) )

    w_o (dict of {param_name: tensor}) is the unlearned model's own weights. Sigma_o (same dict
    shape) is the per-parameter variance describing the spread of the distribution w_o was drawn
    from: for Fisher-scrubbed models this is the *exact* injected noise variance
    (unlearn.fisher.fisher()'s bound_info["noise_variance"]); for any other unlearning method there
    is no explicit noise term, so the caller should instead pass the same Laplace/local-curvature
    approximation used for Sigma_r below -- i.e. 1 / (retain-set Fisher evaluated at w_o), e.g. from
    trainer.val.validate(retain_loader, model, ..., compute_fisher=True)["fisher_diag"].

    w_r is `retrained_model`'s weights. Sigma_r is approximated, as in the paper, by the inverse
    Fisher information of the retain set evaluated at the retrained model, since Sigma_r describes
    the spread of the hypothetical distribution of weights obtained by retraining on the retain set
    from scratch -- passed in precomputed as `retrained_fisher_diag` (same format as returned by
    unlearn.fisher.fisher_information_matrix(retrained_model, retain_loader, device)).
    """
    retrained_named_params = list(retrained_model.named_parameters())

    epsilon = 1e-8
    sigma_r = {
        name: 1.0 / (retrained_fisher_diag[i] + epsilon)
        for i, (name, _) in enumerate(retrained_named_params)
    }
    w_r = {name: param.detach() for name, param in retrained_named_params}

    trace_term = 0.0
    quad_term = 0.0
    log_det_term = 0.0
    k = 0

    for name, w_o_i in w_o.items():
        s_o = sigma_o[name]
        s_r = sigma_r[name].to(w_o_i.device)
        w_r_i = w_r[name].to(w_o_i.device)

        trace_term += (s_o / s_r).sum()
        quad_term += ((w_r_i - w_o_i).pow(2) / s_r).sum()
        log_det_term += (torch.log(s_r) - torch.log(s_o)).sum()
        k += w_o_i.numel()

    bound = 0.5 * (trace_term + quad_term - k + log_det_term)
    return bound.item()




# def information_score(model, x, y, training=False, logistic_regression=False):
#     """
#     Compute Fisher-based information score for the given model and data.
#     The training argument determines if the resulting tensor requires grad and also if the computational graph should be created for the gradient.
#     """
#     # get model prediction

#     predictions = torch.log_softmax(model(x), dim=-1)

#     # if x is just a single data point, expand the tensor by an additional dimension, such that x.shape = [1, n], expand y and predictions accordingly
#     # y = y if len(x.shape) > 1 else y[None, :]
#     # # guarantee that y is one-hot encoded
#     # y = y if len(y.shape) > 1 and not logistic_regression else torch.tensor(onehot(y, 10))
#     # predictions = predictions if len(x.shape) > 1 else predictions[None, :]
#     # x = x if len(x.shape) > 1 else x[None, :]
#     num_data_points = x.shape[0]

#     information = torch.tensor([0.], requires_grad=training)
#     # accumulate information score for all data points
#     for i in range(num_data_points):
#         label = torch.argmax(y[i])
#         prediction = predictions[i][label]
#         # gradient of model prediction w.r.t. the model parameters
#         gradient = grad(prediction, model.parameters(), create_graph=training, retain_graph=True)
#         for derivative in gradient:
#             information = information + torch.sum(derivative**2)

#     # "convert" single-valued tensor to float value
#     information = information[0]
#     # return averaged information score
#     return information / num_data_points


# def efficacy(model, x, y, logistic_regression=False):
#     """Return forgetting score (efficacy)."""
#     information_target_data = information_score(model, x, y, logistic_regression=logistic_regression)
#     eff = torch.inf if information_target_data == 0 else 1. / information_target_data
#     return eff.tolist()


def information_score(fisher_diag):
    """
    Fisher-based information score ι(θ;D) (Becker & Liebig, "Evaluating Machine Unlearning via
    Epistemic Uncertainty", Eq. 7), i.e. the trace of the diagonal Fisher Information Matrix.

    Takes an already-computed FIM diagonal (e.g. from
    unlearn.fisher.fisher_information_matrix(model, forget_loader, device)) rather than looping
    over data points and calling autograd per sample -- summing every entry of the diagonal gives
    the same quantity the old per-sample-loop version computed.
    """
    return sum(t.sum() for t in fisher_diag).item()


def efficacy(fisher_diag):
    """Return forgetting score (efficacy) from a precomputed Fisher information diagonal over the target (forget) data."""
    information_target_data = information_score(fisher_diag)
    return torch.inf if information_target_data == 0 else 1. / information_target_data


def efficacy_upper_bound(grad_norm):
    """
    Return upper bound for forgetting score (efficacy), from a precomputed gradient norm of the
    mean cross-entropy loss over the target (forget) data (Becker & Liebig, "Evaluating Machine
    Unlearning via Epistemic Uncertainty", Lemma 1 / Eq. 13) -- e.g. trainer.val.validate(...,
    compute_fisher=True)["grad_norm"].
    """
    squared_norm = grad_norm ** 2
    return torch.inf if squared_norm == 0 else 1. / squared_norm

