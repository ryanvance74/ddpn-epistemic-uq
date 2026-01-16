from functools import partial
from typing import Type

import torch
import torch.func
import posteriors
from deep_uncertainty.enums import BetaSchedulerType
from deep_uncertainty.enums import LRSchedulerType
from deep_uncertainty.enums import OptimizerType
from deep_uncertainty.models.backbones import Backbone
from deep_uncertainty.models.double_poisson_nn import DoublePoissonNN 



class DoublePoissonBayesianNN(DoublePoissonNN):
    """
    Base class for Bayesian Neural Networks estimating Double Poisson NN parameters.
    
    """
    def __init__(
        self,
        num_mc_samples: int = 10,
        **kwargs
    ):
        """
        Args:
            num_mc_samples (int): Number of Monte Carlo samples to draw during inference 
                                  to approximate the posterior predictive distribution.
            **kwargs: Arguments passed to DoublePoissonNN.
        """
        super().__init__(**kwargs)
        self.num_mc_samples = num_mc_samples
        self.save_hyperparameters()

    def functional(self, params, x):
        """
        A stateless functional call to the model, required by `posteriors`.
        
        Args:
            params: A dictionary of model parameters.
            x: Input tensor.
        """
        return torch.func.functional_call(self, params, (x,))

    def _sample_parameters(self) -> dict | None:
        """
        Abstract method to sample a single set of parameters from the posterior.
        
        Returns:
            dict: A state_dict of parameters for the functional call.
            None: 
        """
        raise NotImplementedError("Subclasses must implement _sample_parameters")

    def _predict_impl(self, x: torch.Tensor) -> torch.Tensor:
        """
        Bayesian prediction implementation using Monte Carlo averaging.
        
        It draws `self.num_mc_samples` from the posterior, computes the outputs,
        and averages the parameters (mu, phi) to form the expected predictive distribution.
        """

        self.backbone.eval() 
        
        outputs = []
        
        for _ in range(self.num_mc_samples):
            params = self._sample_parameters()
            
            if params is not None:
                # This is how prediction is handled with the posteriors library.
                y_hat = self.functional_model(params, x)
            else:
                # If using Monte Carlo dropout, we do not need to use a stateless functional call.
                y_hat = self._forward_impl(x)
                
            outputs.append(torch.exp(y_hat)) # Convert log params to (mu, phi)


        # Shape: (Num_Samples, Batch, 2)
        stacked_outputs = torch.stack(outputs)
        
        mean_prediction = stacked_outputs.mean(dim=0)
        
        self.backbone.train()
        return mean_prediction