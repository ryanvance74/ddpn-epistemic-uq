from losses import double_poisson_nll
import posteriors
def double_poisson_log_posterior(params, batch, model_func, n_data=None, prior_sd=0.5, beta=None):
    """
    A combined log posterior function.
    
    Args:
        params: Model parameters.
        batch: (x, y) batch.
        model_func: Functional wrapper for the model forward pass.
        n_data: If provided, scales likelihood to the total dataset size (for Laplace).
                If None, uses batch-level scaling (for VI).
        prior_sd: Standard deviation of the Gaussian prior.
        beta: Beta parameter for the Double Poisson.
    """
    x, y = batch
    output = model_func(params, x)
    
    # 1. Base Negative Log Likelihood (Mean)
    batch_nll_mean = double_poisson_nll(output, y, beta=beta)
    
    # 2. Scaling Likelihood
    if n_data is not None:
        # LAPLACE MODE: Scale mean NLL to the total dataset
        log_likelihood = -batch_nll_mean * n_data
    else:
        # VI MODE: Sum NLL for this batch
        log_likelihood = -batch_nll_mean * y.shape[0]

    # 3. Log Prior
    log_prior = posteriors.utils.diag_normal_log_prob(params, mean=0.0, sd_diag=prior_sd)
    
    return log_likelihood + log_prior, torch.tensor([])