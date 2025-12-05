import tempfile
from collections.abc import Iterable, Sequence, Iterator
import numpy.typing as npt
from typing import Callable, Literal, Optional, Union

from enum import Enum
from enum import auto

import pyro
import pyro.distributions as dist
import scvi
import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.distributions import Distribution, Bernoulli, Categorical, MixtureSameFamily, Independent, Multinomial, RelaxedBernoulli

from scvi import REGISTRY_KEYS, settings
from scvi.module.base import (
    BaseModuleClass,
    LossOutput,
    PyroBaseModuleClass,
    auto_move_data,
)
from scvi.model._utils import _get_batch_code_from_category
from scvi.distributions._utils import DistributionConcatenator
from scvi.nn import FCLayers
from scvi._types import Number
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
    ObsmField
)
from scvi.module._constants import MODULE_KEYS

from scvi.model.base import BaseModelClass, UnsupervisedTrainingMixin, VAEMixin

from anndata import AnnData
from scvi.data import AnnDataManager

from torch.distributions import NegativeBinomial, Normal, Beta
from torch.distributions import kl_divergence as kl

import sys
from enums import AggMethod, D

import warnings


MAX_MODALITIES = 16
REGISTRY_MODALITY_KEYS = [f'modality_{i}' for i in range(MAX_MODALITIES)]


class SoftplusEps(nn.Module):
    def __init__(self, eps):
        self.eps = eps
        super().__init__()
    
    def forward(self, x):
        return F.softplus(x) + self.eps


class ExpEps(nn.Module):
    def __init__(self, eps):
        self.eps = eps
        super().__init__()

    def forward(self, x):
        return torch.exp(x) + self.eps


class Sqrt(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.sqrt(x)


class Decoder(nn.Module):
    """Decodes data from latent space of ``n_input`` dimensions into ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_layers`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        distr: D,
        dispersion: Literal["gene", "gene-cell"] = 'gene-cell',
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate=0,
        inject_covariates: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        decoder_param_eps: float = 1e-12,
        decoder_activation: Literal["exp", "softplus"] = 'softplus',
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        self.decoder_param_eps = decoder_param_eps

        self.decoder_activation = {
            'exp': ExpEps(decoder_param_eps),
            'softplus': SoftplusEps(decoder_param_eps),
        }[decoder_activation]

        self.distr = distr

        # if distr == D.Normal:
        #     self.param_decoders = nn.ModuleDict(
        #         {
        #             'loc': nn.Linear(n_hidden, n_output),
        #             'scale': nn.Sequential(nn.Linear(n_hidden, n_output), self.decoder_activation),
        #         }
        #     )
        
        if distr.name == 'Normal':
            self.param_decoders = nn.ModuleDict(
                {
                    'loc': nn.Linear(n_hidden, n_output),
                    'scale': nn.Sequential(nn.Linear(n_hidden, n_output), self.decoder_activation),
                }
            )
        
        
        if distr == D.Beta:
            # alpha, beta decoders based on torch.distributions conventions (alpha = concentration1)
            
            self.param_decoders = nn.ModuleDict(
                {
                    'concentration1': nn.Sequential(nn.Linear(n_hidden, n_output), self.decoder_activation),
                    'concentration0': nn.Sequential(nn.Linear(n_hidden, n_output), self.decoder_activation),
                }
            )

            # Scaling factor multiplying both alpha and beta, to allow for easier control over variance while maintaining mean
            self.scaling_factor_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), ExpEps(self.decoder_param_eps))

            # Allow a probability of predicting a constant distribution (not easily modelled with Beta)
            self.constant_value_prob_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Sigmoid())
            self.constant_value_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), nn.Sigmoid())
            

    def forward(
        self,
        z: torch.Tensor,
        *cat_list: int,
    ):
        px = self.px_decoder(z, *cat_list)

        params = {}
        info = {}

        if self.distr == D.Beta:
            info['scaling_factor'] = scaling_factor = self.scaling_factor_decoder(px)
            info['constant_value_prob'] = constant_value_prob = self.constant_value_prob_decoder(px)
            info['constant_value'] = constant_value = self.constant_value_decoder(px)

        for param_name, param_decoder in self.param_decoders.items():
            params[param_name] = param_decoder(px)
            if self.distr == D.Beta:
                params[param_name] *= scaling_factor

        return params, info



class Encoder(nn.Module):
    """Encode data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_layers`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (data space)
    n_output
        The dimensionality of the output (latent space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    distribution
        Distribution of z
    var_eps
        Minimum value for the variance;
        used for numerical stability
    var_activation
        Callable used to ensure positivity of the variance.
        Defaults to :meth:`torch.exp`.
    return_dist
        Return directly the distribution of z instead of its parameters.
    **kwargs
        Keyword args for :class:`~scvi.nn.FCLayers`
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        var_eps: float = 1e-12,
        var_activation: Literal["exp", "softplus"] = 'softplus',
        **kwargs,
    ):
        super().__init__()

        self.var_eps = var_eps
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)

        self.var_encoder = nn.Sequential(
            nn.Linear(n_hidden, n_output), 
            {'softplus': SoftplusEps(var_eps), 'exp': ExpEps(var_eps)}[var_activation]
        )


    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. multivariate normal
            \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)

        Parameters
        ----------
        x
            tensor with shape (n_input,)
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample

        """
        q = self.encoder(x, *cat_list)
        qzm = self.mean_encoder(q)
        qzv = self.var_encoder(q)
        qz = Normal(qzm, qzv.sqrt())
        return {
            'q': q, 'qzm': qzm, 'qzv': qzv, 'qz': qz,
        }


class MultiModalVAE(BaseModuleClass):
    """
    Basic VAE with Isotropic Gaussian prior, varying number of modalities and options for normal or beta distributions.
    There are multiple options for aggregating the modalities, see AggMethod.
    


    """

    def __init__(
        self, 
        input_ds, 
        n_latent, 
        n_hidden, 
        n_layers, 
        n_batch, 
        distrs: Sequence[D] = [D.Normal],
        dropout_rate=0.05, 
        use_batch_norm=True, 
        use_layer_norm=False,
        unimodal_kl=True, 
        joint_kl=False, 
        agg_method: AggMethod = AggMethod.AOE_GLOBAL_WEIGHTS,
        experts_method='POE',
        fixed_latent_weights: Sequence[float] | None = None,
        external_kl_weight: float = 1.0, 
        loss_weights : None | Sequence[float] | Literal['auto'] = None,
        encoder_kwargs = {},
        decoder_kwargs = {'dropout_rate': 0,},
        allow_beta_constant = True,
        n_obs: int = 0,
    ):
        '''
        param distrs: list of distributions for the modalities (including main)
        param loss_weights: 
            None (default) -    each feature is given weight 1, regardless of modality, so that in practice each modality is weighted
                                by the number of features
            Sequence - custom weights given to the loss of each modality.
            'auto' - Equivalent to passing a list where each modality is weighted by d_main / (n_modalities * d_modality). 
                    That is, all modalities have the same total weight for their features (d_main / n_modalities), so that 
                    the total sum of weights is d_main, same as if the main modality was alone
        param external_kl_weight: Multiply KL term(s) by external_kl_weight (good for debugging)
        param encoder_kwargs: kwargs to init of encoder. Will take precedence over arguments passed outside encoder_kwargs.
        param decoder_kwargs: kwargs to init of decoder. Will take precedence over arguments passed outside decoder_kwargs.
        param n_obs: number of observations 
        '''
        
        #TODO - allow passing kwargs to the decoders
        super().__init__()

        n_modalities = self.n_modalities = len(distrs)
    
        assert isinstance(input_ds, Sequence) and len(input_ds) == n_modalities, "Internal error, illegal input_ds"

        self.agg_method = agg_method
        self.experts_method=experts_method
        if fixed_latent_weights is not None:
            assert self.agg_method == AggMethod.AOE_FIXED_WEIGHTS
            weights = torch.tensor(fixed_latent_weights, requires_grad=False)
            assert weights.shape == (n_modalities, ), "Number of modalities must match shape of weights list"
            self.fixed_weights = weights
        else:
            self.fixed_weights = None

        if loss_weights is None:
            loss_weights = [1]*n_modalities
        elif loss_weights == 'auto':
            loss_weights = [input_ds[0] / (self.n_modalities * input_ds[i]) for i in range(len(input_ds))]
        else:
            assert len(loss_weights) == n_modalities

        self._loss_weights = torch.tensor(loss_weights, requires_grad=False)

        self.n_latent = n_latent
        self.input_ds = input_ds
        self.distrs = distrs
        self.distr_classes = [d.to_torch() for d in self.distrs]
        self.external_kl_weight = external_kl_weight
        self.unimodal_kl = unimodal_kl
        self.joint_kl = joint_kl
        self.allow_beta_constant = allow_beta_constant
        if self.n_modalities == 1:
            # Eliminate double KL
            self.unimodal_kl = True
            self.joint_kl = False

        common_kwargs = dict(n_layers=n_layers, n_hidden=n_hidden, use_batch_norm=use_batch_norm, use_layer_norm=use_layer_norm, dropout_rate=dropout_rate)
        encoder_kwargs_ = common_kwargs.copy()
        encoder_kwargs_.update(encoder_kwargs)
        decoder_kwargs_ = common_kwargs.copy()
        decoder_kwargs_.update(decoder_kwargs)


        # if self.agg_method == AggMethod.SHARED_ENCODER:
        #     assert (not self.unimodal_kl) and self.joint_kl, "For shared encoder, only joint kl is possible"
        #     self.encoders = nn.ModuleList([
        #         Encoder(n_input=sum(input_ds), n_output=n_latent, **encoder_kwargs_)
        #     ])

        # else:
        #     self.encoders = nn.ModuleList(
        #         [
        #             Encoder(n_input=input_d, n_output=n_latent, **encoder_kwargs_)
        #             for input_d in self.input_ds
        #         ]
        #     )
            
        if self.agg_method.name =='SHARED_ENCODER':
            assert (not self.unimodal_kl) and self.joint_kl, "For shared encoder, only joint kl is possible"
            self.encoders = nn.ModuleList([
                Encoder(n_input=sum(input_ds), n_output=n_latent, **encoder_kwargs_)
            ])

        else:
            self.encoders = nn.ModuleList(
                [
                    Encoder(n_input=input_d, n_output=n_latent, **encoder_kwargs_)
                    for input_d in self.input_ds
                ]
            )

        self.decoders = nn.ModuleList(
            [
                Decoder(n_input=n_latent, n_output=input_d, distr=distr, n_cat_list=[n_batch], **decoder_kwargs_)
                for input_d, distr in zip(self.input_ds, self.distrs)            
            ]
        )

        if self.agg_method == AggMethod.AOE_GLOBAL_WEIGHTS:
            self.learned_weights = nn.Parameter(torch.ones((self.n_modalities, )))
        elif self.agg_method == AggMethod.AOE_PER_CELL_WEIGHTS:
            self.weight_encoder = FCLayers(n_in=sum(self.input_ds), n_out=self.n_modalities, n_layers=1)

    
    def _weighted_sum(self, tensors, weights):
        '''
        tensors: sequence of length n of tensors of shape (batch_size, *(feature_dims))
        weights: tensor of shape (n,) or (batch_size, n) where n is length of tensors
        '''
        n = weights.shape[-1]
        batch_size = tensors[0].shape[0]
        dtype, device = tensors[0].dtype, tensors[0].device
        feature_shape = tensors[0].shape[1:]

        if weights.dim() == 1:
            new_shape = (n, *([1]*len(tensors[0].shape)))
        elif weights.dim() == 2:
            assert weights.shape == (batch_size, n)
            new_shape = (n, batch_size, *([1]*len(feature_shape)))
        weights = weights.reshape(new_shape).to(dtype).to(device)
        return torch.sum(torch.stack(tensors, dim=0)*weights, dim=0)

    def _get_inference_input(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Parse the dictionary to get appropriate args"""
        return {
            "inputs": [tensors[REGISTRY_KEYS.X_KEY]] + [tensors[REGISTRY_MODALITY_KEYS[i]] for i in range(1, self.n_modalities)],
        }

    def get_global_weights(self, numpy=True):
        if self.agg_method == AggMethod.AOE_GLOBAL_WEIGHTS:
            weights = F.softmax(self.learned_weights, dim=0)
        elif self.agg_method == AggMethod.AOE_FIXED_WEIGHTS:
            weights = self.fixed_weights.numpy(force=True)
        else:
            return None
        return weights if not numpy else weights.numpy(force=True)

    def get_per_cell_weights(self, inputs, grad=False, numpy=True):
        if self.agg_method != AggMethod.AOE_PER_CELL_WEIGHTS:
            return None
        with torch.set_grad_enabled(grad):
            concat_input = torch.concat(inputs, dim=-1).to(device=self.device)
            scores = self.weight_encoder(concat_input)
            weights = F.softmax(scores, dim=1)
            return weights if not numpy else weights.numpy(force=True)


    @auto_move_data
    def inference(self, inputs, n_samples=1) -> dict[str, torch.Tensor]:
        """
        High level inference method.
        Runs the inference (encoder) model.
        """

        sample_shape = (n_samples, ) if n_samples > 1 else torch.Size([])

        
        
        
        if self.agg_method.name =='SHARED_ENCODER':
            inference_results = [
                self.encoders[0](torch.cat(inputs, dim=-1))
            ]
        else:
            inference_results = [
                encoder(input) for input, encoder in zip(inputs, self.encoders)
            ]

        for encoder_result in inference_results:
            encoder_result['z'] = encoder_result['qz'].rsample(sample_shape=sample_shape)
    
        qs, means, vars, zs = zip(*[(res['q'], res['qzm'], res['qzv'], res['z']) for res in inference_results])
        batch_size = means[0].shape[0]
        device = means[0].device

        if self.agg_method.name == 'SHARED_ENCODER':
            shared_qzm, shared_qzv, shared_z = means[0], vars[0], zs[0]
            inference_results = None
            weights = None
        
        elif self.agg_method in (AggMethod.AOE_FIXED_WEIGHTS, AggMethod.AOE_GLOBAL_WEIGHTS, AggMethod.AOE_PER_CELL_WEIGHTS):
            if self.agg_method in (AggMethod.AOE_FIXED_WEIGHTS, AggMethod.AOE_GLOBAL_WEIGHTS):
                
                if self.experts_method=='POE':
                
                    eps = 1e-6
                    precisions, prec_mu = [], []
                    for modality_num, (mu_m, var_m, input_m) in enumerate(zip(means, vars, inputs)):
                        var_m = torch.clamp(var_m, min=eps)
                        lambda_m = 1.0 / var_m

                        if modality_num == 1:  
                            missing_mask = (input_m == 0).all(dim=1) | (input_m == 1000).all(dim=1)
                            lambda_m[missing_mask, :] = 0.0
                            mu_m[missing_mask, :] = 0.0

                        precisions.append(lambda_m)
                        prec_mu.append(lambda_m * mu_m)

                    Lambda = torch.stack(precisions, dim=0).sum(dim=0)
                    eta = torch.stack(prec_mu, dim=0).sum(dim=0)
                    shared_qzv = torch.clamp(1.0 / (Lambda + eps), min=eps)
                    shared_qzm = shared_qzv * eta
                    weights = self.get_global_weights(numpy=False)
                    assert weights.shape == (self.n_modalities, )
                    
                elif self.experts_method=='MOE':
                    weights = self.get_global_weights(numpy=False)
                    assert weights.shape == (self.n_modalities, )
                    shared_qzm = self._weighted_sum(means, weights)
                    shared_qzv = self._weighted_sum(vars, torch.square(weights))
                
                
            elif self.agg_method in (AggMethod.AOE_PER_CELL_WEIGHTS,):
                weights = self.get_per_cell_weights(inputs, grad=True, numpy=False)
                assert weights.shape == (batch_size, self.n_modalities)
            
            

        shared_qz = Normal(shared_qzm, shared_qzv.sqrt())
        shared_z = shared_qz.rsample(sample_shape=sample_shape)


        return {"qzm": shared_qzm, "qzv": shared_qzv, "z": shared_z, 'qz': shared_qz, "modalities": inference_results, 
                'inputs': inputs, 'weights': weights }


    def _get_generative_input(
        self, tensors: dict[str, torch.Tensor], inference_outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        return {
            "z": inference_outputs["z"],
            MODULE_KEYS.BATCH_INDEX_KEY: tensors[REGISTRY_KEYS.BATCH_KEY],
        }

    @auto_move_data
    def generative(self, z: torch.Tensor, batch_index: torch.Tensor) -> dict[str, torch.Tensor]:
        """Runs the generative model."""
        px_params = []
        px_distrs = []
        px_infos = []
        for i, decoder in enumerate(self.decoders):
            px_params_i, info = decoder(z, batch_index)
            if decoder.distr == D.Beta and self.allow_beta_constant:
                constant_prob, constant_value = info['constant_value_prob'], info['constant_value']
                alpha, beta = px_params_i['concentration1'], px_params_i['concentration0']
                px_distr = MixtureSameFamily(
                    Categorical(probs=torch.stack((constant_prob, 1-constant_prob), dim=-1)),
                    Beta(
                        concentration1=torch.stack((
                                constant_value / (1 - constant_value) *1e6*torch.ones_like(alpha),
                                alpha,
                            ), dim=-1
                        ),
                        concentration0=torch.stack((
                                1e6*torch.ones_like(beta),
                                beta,
                            ), dim=-1
                        )
                    ),
                )
            else:
                px_distr = self.distr_classes[i](**px_params_i)
            px_params.append(px_params_i)
            px_distrs.append(px_distr)
            px_infos.append(info)
        return {
            'px_params': px_params,
            'px_distrs': px_distrs,
            'px_info': px_infos,
        }


    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor],
        generative_outputs: dict[str, torch.Tensor],
        kl_weight: float=1.0
    ) -> LossOutput:
        
        # KL loss
        joint_qzm = inference_outputs["qzm"]
        joint_qzv = inference_outputs["qzv"]

        kl_divergence = 0

        prior_dist = Normal(torch.zeros_like(joint_qzm), torch.ones_like(joint_qzv))

        if self.joint_kl:
            var_post_dist = inference_outputs['qz']
            kl_divergence += kl(var_post_dist, prior_dist).sum(dim=-1)

        spatial_mask = None
        if "spatial_masked" in tensors:
            spatial_mask = tensors["spatial_masked"].to(joint_qzm.device).float().squeeze()

        if self.unimodal_kl:
            for i, latent_param_dict in enumerate(inference_outputs['modalities']):
                mod_post_dist = Normal(latent_param_dict['qzm'], torch.sqrt(latent_param_dict['qzv']))
                mod_kl=kl(mod_post_dist, prior_dist).sum(dim=-1)
                if i==1 and spatial_mask in tensors:
                    kl_divergence += (mod_kl * (1.0 - spatial_mask))
                else:
                    kl_divergence += mod_kl

    
        inputs = inference_outputs['inputs']
        log_lik = 0
        for i, (input, distr, loss_weight) in enumerate(zip(inputs, generative_outputs["px_distrs"], self._loss_weights)):
            mod_log_lik = distr.log_prob(input).sum(dim=-1)
            
            if i == 1 and "spatial_masked" in tensors:                
                mod_log_lik = mod_log_lik * (1.0 - spatial_mask)                    
                   
            log_lik += loss_weight*mod_log_lik

        elbo = log_lik - kl_weight*self.external_kl_weight*kl_divergence
        loss = torch.mean(-elbo)

        return LossOutput(
            loss=loss,
            reconstruction_loss=-log_lik,
            kl_local=kl_divergence,
            kl_global=0.0,
        )
