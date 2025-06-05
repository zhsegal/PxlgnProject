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
from torch.distributions import Distribution, Bernoulli, Categorical, MixtureSameFamily, Independent, Multinomial

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

import warnings


MAX_MODALITIES = 16
REGISTRY_MODALITY_KEYS = [f'modality_{i}' for i in range(MAX_MODALITIES)]


def _identity(x):
    return x

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

class Dispersion(Enum):
    GENE = auto()
    GENE_CELL = auto()

    @classmethod
    def from_str(cls, str):
        return {'gene': cls.GENE, 'gene-cell': cls.GENE_CELL}[str]


class Decoder(nn.Module):
    """Decodes data from latent space of ``n_input`` dimensions into ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

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
        distr: Literal["normal", "beta"],
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

        self.dispersion = Dispersion.from_str(dispersion)

        if distr == 'normal':
            # loc, scale decoders
            # self.param_decoders = nn.ModuleDict(
            #     {
            #         'loc': nn.Linear(n_hidden, n_output),
            #     }
            # )
            # if self.dispersion == Dispersion.GENE_CELL:
            #     self.log_scale_decoder = nn.Linear(n_hidden, n_output)
            # elif self.dispersion == Dispersion.GENE:
            #     self.log_scale_param = nn.Parameter(torch.zeros(n_output))
            self.param_decoders = nn.ModuleDict(
                {
                    'loc': nn.Linear(n_hidden, n_output),
                    'scale': nn.Sequential(nn.Linear(n_hidden, n_output), SoftplusEps(decoder_param_eps)),
                }
            )
            # self.param_decoders.add_module('loc_decoder', nn.Linear(n_hidden, n_output))
            # self.param_decoders.add_module('scale_decoder', nn.Sequential(nn.Linear(n_hidden, n_output), SoftplusEps(self.decoder_param_eps)))
        if distr == 'beta':
            # alpha, beta decoders based on torch.distributions conventions
            # self.param_decoders = nn.ModuleDict(
            #     {
            #         'con1': nn.Sequential(nn.Linear(n_hidden, n_output), self.decoder_activation),
            #         'con0': nn.Sequential(nn.Linear(n_hidden, n_output), self.decoder_activation),
            #     }
            # )   
            # if self.dispersion == Dispersion.GENE_CELL:
            #     self.log_scaling_factor_decoder = nn.Linear(n_hidden, n_output)
            # elif self.dispersion == Dispersion.GENE:
            #     self.log_scaling_factor_param = nn.Parameter(torch.zeros(n_output))
            
            self.param_decoders = nn.ModuleDict(
                {
                    'concentration1': nn.Sequential(nn.Linear(n_hidden, n_output), SoftplusEps(self.decoder_param_eps)),
                    'concentration0': nn.Sequential(nn.Linear(n_hidden, n_output), SoftplusEps(self.decoder_param_eps)),
                }
            )
            self.scaling_factor_decoder = nn.Sequential(nn.Linear(n_hidden, n_output), SoftplusEps(self.decoder_param_eps))
            

            # self.param_decoders.add_module('alpha_decoder', nn.Sequential(nn.Linear(n_hidden, n_output), SoftplusEps(self.decoder_param_eps)))
            # self.param_decoders.add_module('beta_decoder', nn.Sequential(nn.Linear(n_hidden, n_output), SoftplusEps(self.decoder_param_eps)))

    def forward(
        self,
        z: torch.Tensor,
        *cat_list: int,
    ):
        px = self.px_decoder(z, *cat_list)

        params = {}

        # def _act(t):
        #     return self.decoder_activation(t)

        # def _expand_1d(t, other):
        #     view_shape = [1] * (other.dim() - 1) + [t.size(0)]
        #     return t.view(*view_shape).expand_as(other)

        # for param_name, param_decoder in self.param_decoders.items():
        #     params[param_name] = param_decoder(px)

        # if self.distr == 'normal':
        #     if self.dispersion == Dispersion.GENE:
        #         log_scale = _expand_1d(self.log_scale_param, params['loc'])
        #     elif self.dispersion == Dispersion.GENE_CELL:
        #         log_scale = self.log_scale_decoder(px)
        #     params['scale'] = _act(log_scale)

        # if self.distr == 'beta':
        #     if self.dispersion == Dispersion.GENE:
        #         log_scaling_factor = _expand_1d(self.log_scaling_factor_param, params['con1'])
        #     elif self.dispersion == Dispersion.GENE_CELL:
        #         log_scaling_factor = self.log_scaling_factor_decoder(px)
        #     scaling_factor = _act(log_scaling_factor)
        #     for param_name in params.keys():
        #         params[param_name] *= scaling_factor


        if self.distr == 'beta':
            scaling_factor = self.scaling_factor_decoder(px)

        for param_name, param_decoder in self.param_decoders.items():
            params[param_name] = param_decoder(px)
            if self.distr == 'beta':
                params[param_name] *= scaling_factor

        return params



class Encoder(nn.Module):
    """Encode data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

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
        var_activation: Callable | None = None,
        return_dist: bool = False,
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
        self.var_encoder = nn.Sequential(nn.Linear(n_hidden, n_output), SoftplusEps(var_eps))
        self.return_dist = return_dist
        # self.var_activation = SoftplusEps(var_eps)

    def forward(self, x: torch.Tensor, n_samples=0, *cat_list: int):
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
        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)
        qzm = self.mean_encoder(q)
        qzv = self.var_encoder(q)
        qz = Normal(qzm, qzv.sqrt())
        # dist = Normal(qzm, qzv.sqrt())
        # z = dist.rsample()
        # if self.return_dist:
        #     return dist, z
        return {
            'q': q, 'qzm': qzm, 'qzv': qzv, 'qz': qz,
        }


class MultiModalVAE(BaseModuleClass):
    """
    Basic VAE with Isotropic Gaussian prior, varying number of modalities and options for normal or beta distributions
    Modality integration is achieved by modelling the joint latent as a weighted sum of the unimodal latents, which are each a Gaussian (hence the 
    sum too is a Gaussian)

    agg_method: one of 'learned_local_avg', 'learned_global_avg', 'fixed_avg', 'moe', 'shared_encoder',
    loss_weights: 
        None (default) - each feature is given weight 1, regardless of modality, so that in practice each modality is weighted
                         by the number of features
        Iterable - custom weights given to the loss of each modality.
        'auto' - Equivalent to passing a list where each modality is weighted by d_X / d_modality. 
                 That is, the extra modalities are weighted by the number of features in each one, so that the total weight of each modality
                 is equal to the total weight (e.g. number of features) of the main modality. 

    """

    def __init__(self, input_ds, latent_d, hidden_d, n_hidden, n_batch, n_modalities=1, weights=None, distrs=['normal',], 
                 dropout_rate=0.05, use_batch_norm=True, unimodal_kl=True, joint_kl=False, agg_method='learned_local_avg',
                 external_kl_weight=1, loss_weights=None,
                 ):
        
        #TODO - allow passing kwargs to the decoders
        super().__init__()


        assert isinstance(input_ds, Sequence) and torch.all(torch.tensor([len(input_ds), len(distrs)]) == n_modalities)
        assert agg_method in ['learned_local_avg', 'learned_global_avg', 'fixed_avg', 'moe', 'shared_encoder']

        self.agg_method = agg_method
        self.learn_agg = agg_method not in ['moe', 'fixed_avg']
    
        if weights is not None:
            assert len(weights) == n_modalities
            assert not self.learn_agg
            self._weights = torch.tensor(weights, requires_grad=False)
        else:
            assert self.learn_agg
            self._weights = None
        assert all([d in ('normal', 'beta') for d in distrs])

        if loss_weights is None:
            loss_weights = torch.ones((n_modalities,))
        elif loss_weights == 'auto':
            loss_weights = [input_ds[0] / input_ds[i] for i in range(len(input_ds))]
        else:
            assert len(loss_weights) == n_modalities

        self._loss_weights = torch.tensor(loss_weights, requires_grad=False)

        self.latent_d = latent_d
        self.n_modalities = n_modalities
        self.input_ds = input_ds
        self.distrs = distrs
        self.distr_classes = [{'normal': Normal, 'beta': Beta}[d] for d in self.distrs]
        self.external_kl_weight = external_kl_weight
        self.unimodal_kl = unimodal_kl
        self.joint_kl = joint_kl
        if self.n_modalities == 1:
            # Eliminate double KL
            self.unimodal_kl = True
            self.joint_kl = False

        # if n_extra_modalities > 0:
        #     assert isinstance(input_d, Sequence) and torch.all(torch.tensor([len(weights), len(input_d)]) == n_extra_modalities + 1)
        #     self.input_ds = input_d
        #     assert isinstance(weights, torch.Tensor)
        #     self.weights = weights
        # else:
        #     self.input_ds = [input_d]
        #     self.weights = torch.tensor([1,], dtype=float)


        common_kwargs = dict(n_hidden=hidden_d, n_layers=n_hidden, use_batch_norm=use_batch_norm)

        if self.agg_method == 'shared_encoder':
            assert (not self.unimodal_kl) and self.joint_kl
            self.encoders = nn.ModuleList([
                Encoder(n_input=sum(input_ds), n_output=latent_d, dropout_rate=dropout_rate, **common_kwargs)
            ])

        else:
            self.encoders = nn.ModuleList(
                [
                    Encoder(n_input=input_d, n_output=latent_d, dropout_rate=dropout_rate, **common_kwargs)
                    for input_d in self.input_ds
                ]
            )

        self.decoders = nn.ModuleList(
            [
                Decoder(n_input=latent_d, n_output=input_d, distr=distr, n_cat_list=[n_batch], dropout_rate=0, **common_kwargs)
                for input_d, distr in zip(self.input_ds, self.distrs)            
            ]
        )

        if self.learn_agg:
            # if self.learn_agg == 'layer':
            #     # Pooling layer takes as input the output of the encoders before mean and var prediction
            #     self.pooling_layer = FCLayers(n_in=hidden_d*self.n_modalities, n_out=hidden_d, n_layers=1, 
            #                                 use_batch_norm=use_batch_norm, dropout_rate=dropout_rate)
            #     self.shared_mean_encoder = nn.Linear(hidden_d, self.latent_d)
            #     self.shared_var_encoder = nn.Sequential(nn.Linear(hidden_d, self.latent_d), SoftplusEps(eps=1e-12))

            # elif self.learn_agg == 'weights':
            if self.agg_method == 'learned_global_avg':
                self.learned_weights = nn.Parameter(torch.ones((self.n_modalities, )))
            elif self.agg_method == 'learned_local_avg':
                # self.weight_encoder = FCLayers(n_in=hidden_d*self.n_modalities, n_out=self.n_modalities, n_layers=2,
                #                             n_hidden=hidden_d, 
                #                             use_batch_norm=use_batch_norm, dropout_rate=dropout_rate)
                self.weight_encoder = nn.Linear(in_features=hidden_d*self.n_modalities, out_features=self.n_modalities)
                # self.weight_encoder = FCLayers(n_in=hidden_d*self.n_modalities, n_out=self.n_modalities, n_layers=1, 
                #                             use_batch_norm=use_batch_norm, dropout_rate=dropout_rate)
                # self.learned_weights = nn.Parameter(torch.ones((self.n_modalities, )))
            # self.pooling_layer = nn.Linear(n_in=self.latent_d*self.n_modalities, n_out=self.latent_d)



    
    def _weighted_sum(self, tensors, weights):
        '''
        tensors: sequence of length n of tensors of shape (batch_size, *(feature_dims))
        weights: tensor of shape (n,) or (batch_size, n) where n is length of tensors
        '''
        n = weights.shape[-1]
        batch_size = tensors[0].shape[0]
        dtype, device = tensors[0].dtype, tensors[0].device
        feature_shape = tensors[0].shape[1:]
        # if not isinstance(weights, torch.Tensor):
        #     weights = torch.tensor(weights, dtype=dtype, requires_grad=False)
        if weights.dim() == 1:
            new_shape = (n, *([1]*len(tensors[0].shape)))
        elif weights.dim() == 2:
            assert weights.shape == (batch_size, n)
            new_shape = (n, batch_size, *([1]*len(tensors[0].shape[1:])))
        weights = weights.reshape(new_shape).to(dtype).to(device)
        return torch.sum(torch.stack(tensors, dim=0)*weights, dim=0)
    
    # def _weighted_sum_1(self, tensors, weights, expand_method):
    #     '''
    #     Computes weighted sum of tensors according to weights.
    #     tensors: sequence of length n of tensors of shape (batch_size, *(feature_dims))
    #     weights: tensor of shape (n,)
    #     expand_method: method of expanding weights to the shape of tensors
    #     '''
    #     n = weights.shape[-1]
    #     batch_size = tensors[0].shape[0]
    #     dtype, device = tensors[0].dtype, tensors[0].device
    #     if expand_method == '1':
    #         weights = weights.reshape((1, n)).expand(batch_size, n)
    #         new_shape = (n, batch_size, *([1]*len(tensors[0].shape[1:])))
    #     elif expand_method == '2':
    #         new_shape = (n, *([1]*len(tensors[0].shape)))
    #     weights = weights.reshape(new_shape).to(dtype).to(device)
    #     return torch.sum(torch.stack(tensors, dim=0)*weights, dim=0)

    def _get_inference_input(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Parse the dictionary to get appropriate args"""
        # let us fetch the data, and add it to the dictionary
        return {
                "inputs": [tensors[REGISTRY_KEYS.X_KEY]] + [tensors[REGISTRY_MODALITY_KEYS[i]] for i in range(1, self.n_modalities)],
                # "x": tensors[REGISTRY_KEYS.X_KEY], 
                # "inputs": [tensors[REGISTRY_KEYS.X_KEY]] + [tensors[REGISTRY_OBSM_MODALITY_KEYS[i]] for i in range(self.n_extra_modalities)],
                # **{f'{self._extra_mod_s(i)}': tensors[REGISTRY_OBSM_MODALITY_KEYS[i]] for i in range(1, self.n_modalities + 1)},
                }

    def _get_weights(self):
        if self.agg_method in ('learned_global_avg', ):
            return F.softmax(self.learned_weights, dim=0)
        else:
            return self._weights

    @property
    def weights(self):
        _weights = self._get_weights()
        if _weights is None:
            return None
        return _weights.numpy(force=True)


    @auto_move_data
    def inference(self, inputs, n_samples=1) -> dict[str, torch.Tensor]:
        """
        High level inference method.

        Runs the inference (encoder) model.
        """

        sample_shape = (n_samples, ) if n_samples > 1 else torch.Size([])

        if self.agg_method == 'shared_encoder':
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


        if self.agg_method in ('fixed_avg', 'learned_global_avg', 'learned_local_avg'):

            if self.agg_method in ('fixed_avg', 'learned_global_avg'):
                # weights = self._get_weights().reshape((1, self.n_modalities)).expand(batch_size, self.n_modalities)
                weights = self._get_weights()
                # print(weights.requires_grad)
                # assert weights.shape == (batch_size, self.n_modalities)
                # weights = self._get_weights()
                # assert weights.shape == (self.n_modalities,)
        
            elif self.agg_method in ('learned_local_avg',):
                # concat_inputs = torch.cat(inputs, dim=1)
                # assert concat_inputs.shape == (batch_size, sum(self.input_ds))
                # weights = self.weight_encoder(concat_inputs)
                concat_reps = torch.cat(qs, dim=1)
                weights = self.weight_encoder(concat_reps)
                assert weights.shape == (batch_size, self.n_modalities)
                weights = F.softmax(weights, dim=1)
            
            shared_qzm = self._weighted_sum(means, weights)
            shared_qzv = self._weighted_sum(vars, torch.square(weights))
            shared_qz = Normal(shared_qzm, shared_qzv.sqrt())
            shared_z = shared_qz.rsample(sample_shape=sample_shape)
            
        
        elif self.agg_method == 'moe':
            weights = self._get_weights().to(device)
            mix  = Categorical(probs=weights)
            comp = Independent(Normal(torch.stack(means, dim=-2), torch.stack(vars, dim=-2).sqrt()), 1)
            shared_qz  = MixtureSameFamily(mix, comp)
            shared_qzm = shared_qz.mean
            shared_qzv = shared_qz.variance
            multinomial = Multinomial(total_count=n_samples, probs=weights).sample().to(torch.int)
            if n_samples == 1:
                chosen_mod_index = list(multinomial).index(1)
                shared_z = zs[chosen_mod_index]
            else:
                raise NotImplementedError()
        
        elif self.agg_method == 'shared_encoder':
            shared_qzm, shared_qzv, shared_z = means[0], vars[0], zs[0],
            shared_qz = Normal(shared_qzm, shared_qzv.sqrt())
            inference_results = None
            weights = None


            # shared_qzm = shared_qz.mean
            # shared_qzv = shared_qz.variance
            # shared_z = shared_qz.rsample(sample_shape=sample_shape)

        # elif self.learn_agg == 'layer':
        #     stacked_q = torch.concat([res['q'] for res in inference_results], dim=1)
        #     shared_q = self.pooling_layer(stacked_q)
        #     shared_qzm = self.shared_mean_encoder(shared_q)
        #     shared_qzv = self.shared_var_encoder(shared_q)
        #     shared_z = Normal(shared_qzm, torch.sqrt(shared_qzv)).rsample((n_samples,))


        return {"qzm": shared_qzm, "qzv": shared_qzv, "z": shared_z, 'qz': shared_qz, "modalities": inference_results, 
                'inputs': inputs, 'weights': weights }


        # get variational parameters via the encoder networks
        # qzm = self.mean_encoder(x)
        # qzv = torch.exp(self.log_var_encoder(x))
        # # get one sample to feed to the generative model
        # # under the hood here is the Reparametrization trick (Rsample)
        # z = Normal(qzm, torch.sqrt(qzv)).rsample()

        # return {"qzm": qzm, "qzv": qzv, "z": z}

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
        # px_scale = self.decoder(z, batch_index)
        # theta = torch.exp(self.log_theta)

        # sampled = None
        # if sample:
        #     sampled = Normal(loc=px_scale, scale=torch.sqrt(theta)).sample()

        # return {
        #     "px_scale": px_scale,
        #     "theta": theta,
        #     "sample": sampled,
        # }

        px_params = []
        px_distrs = []
        for i, decoder in enumerate(self.decoders):
            px_params_i = decoder(z, batch_index)
            px_params.append(px_params_i)
            px_distrs.append(self.distr_classes[i](**px_params_i))
        return {
            'px_params': px_params,
            'px_distrs': px_distrs,
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

        if self.unimodal_kl:
            for latent_param_dict, loss_weight in zip(inference_outputs['modalities'], self._loss_weights):
                mod_post_dist = Normal(latent_param_dict['qzm'], torch.sqrt(latent_param_dict['qzv']))
                kl_divergence += loss_weight*(kl(mod_post_dist, prior_dist).sum(dim=-1))

    
        inputs = inference_outputs['inputs']
        log_lik = 0
        for input, distr, loss_weight in zip(inputs, generative_outputs['px_distrs'], self._loss_weights):
            # x = tensors[REGISTRY_KEYS.X_KEY]
            mod_log_lik = distr.log_prob(input).sum(dim=-1)
            log_lik += loss_weight*mod_log_lik

        elbo = log_lik - kl_weight*self.external_kl_weight*kl_divergence
        loss = torch.mean(-elbo)

        return LossOutput(
            loss=loss,
            reconstruction_loss=-log_lik,
            kl_local=kl_divergence,
            kl_global=0.0,
        )


class MultiModalSCVI(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """single-cell Variational Inference [Lopez18]_."""

    def __init__(
        self,
        adata: AnnData,
        n_latent: int = 20,
        n_hidden: int = 128,
        n_layers: int = 1,
        distrs : Sequence[str] = ['normal'],
        weights = None,
        **model_kwargs,
    ):
        super().__init__(adata)

        self.obsm_extra_modality_keys = self.get_setup_arg('obsm_extra_modality_keys')
        self.n_modalities = self.get_setup_arg('n_modalities')


        assert self.n_modalities == len(distrs), "Number of modalities must match length of distrs list"


        if weights:
            assert self.n_modalities == len(weights), "Number of modalities must match length of weights list"

        self.distrs = distrs

        self.n_modalities = self.n_modalities

        self.modality_key_to_index = {
            'X': 0,
            **{key: i+1 for i, key in enumerate(self.obsm_extra_modality_keys)}
        }

        self.modality_key_to_registry_key = {
            'X': REGISTRY_KEYS.X_KEY,
            **{key: REGISTRY_MODALITY_KEYS[i+1] for i, key in enumerate(self.obsm_extra_modality_keys)}
        }

        self.modality_key_to_df = {
            'X': adata.to_df(self.registered_layer),
            **{key: adata.obsm[key] for key in self.obsm_extra_modality_keys}
        }

        self.modalities = self.modality_key_to_index.keys()



        if self.n_modalities == 1:
            self.input_ds = [self.summary_stats['n_vars'], ]
        else:
            self.input_ds = [self.summary_stats['n_vars']] + [adata.obsm[k].shape[1] for k in self.obsm_extra_modality_keys]


        self.module = MultiModalVAE(
            input_ds=self.input_ds,
            latent_d=n_latent,
            hidden_d=n_hidden,
            n_hidden=n_layers,
            distrs=distrs,
            n_modalities=self.n_modalities,
            weights=weights,
            n_batch=self.summary_stats["n_batch"],
            **model_kwargs,
        )
        self._model_summary_string = (
            f"Normal SCVI Model with the following params: \nn_latent: {n_latent}\nn_modalities: {self.n_modalities}\nmodalities: {self.modalities}\nweights: {self.module.weights}\nagg_method: {self.module.agg_method}\ninput_ds: {self.input_ds}\nmodality indices: {self.modality_key_to_index}\ndistributions: {self.distrs}"
        )
        self.init_params_ = self._get_init_params(locals())


    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        batch_key: str | None = None,
        layer: str | None = None,
        n_modalities = 1,
        obsm_extra_modality_keys = [],
        is_count_data: bool = False,
        **kwargs,
    ) -> AnnData | None:
        setup_method_args = cls._get_setup_method_args(**locals())

        assert n_modalities - 1 == len(obsm_extra_modality_keys), "Number of modalities - 1 must match length of obsm extra modality keys list"

        for key in obsm_extra_modality_keys:
            if not isinstance(adata.obsm[key], pd.DataFrame):
                raise RuntimeError('Each obsm key must be a dataframe')
        

        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=is_count_data),
            # LayerField(REGISTRY_MODALITY_KEYS[0], layer, is_count_data=is_count_data),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            # Dummy fields required for VAE class.
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, None),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, None, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, None),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, None),
        ] + [
            ObsmField(REGISTRY_MODALITY_KEYS[i+1], key) for i, key in enumerate(obsm_extra_modality_keys)
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
    
    def get_setup_arg(self, arg):
        manager = self.get_anndata_manager(self.adata)
        return manager._get_setup_method_args()['setup_args'][arg]
    
    @property
    def registered_layer(self):
        return self.get_setup_arg('layer')
    
    @torch.inference_mode()
    def get_weights(        
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        dataloader: Iterator[dict[str, Tensor | None]] = None,
    ):
        '''
        Returns numpy array of shape (n_points, n_modalities) with weights per point per modality
        '''
        self._check_if_trained(warn=False)
        if adata is not None and dataloader is not None:
            raise ValueError("Only one of `adata` or `dataloader` can be provided.")

        if dataloader is None:
            adata = self._validate_anndata(adata)
            dataloader = self._make_data_loader(
                adata=adata, indices=indices, batch_size=batch_size
            )
        
        n = sum([len(t) for t in dataloader])
        
        if self.module.weights is not None:
            return np.repeat(self.module.weights[np.newaxis, :], repeats=n, axis=0)

        weights = []
        for tensors in dataloader:
            outputs: dict[str, Tensor | Distribution | None] = self.module.inference(
                **self.module._get_inference_input(tensors)
            )
            weights.append(outputs['weights'].cpu())

        return torch.cat(weights).numpy()
        



    @torch.inference_mode()
    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        give_mean: bool = True,
        modality: str = None,
        mc_samples: int = 5_000,
        batch_size: int | None = None,
        return_dist: bool = False,
        dataloader: Iterator[dict[str, Tensor | None]] = None,
    ) -> npt.NDArray | tuple[npt.NDArray, npt.NDArray]:
        """Compute the latent representation of the data.

        This implementation adds the modality argument. Default (None) or 'joint' is the combined latent. "X" means the main modality.

        This is typically denoted as :math:`z_n`.

        Parameters
        ----------
        adata
            :class:`~anndata.AnnData` object with :attr:`~anndata.AnnData.var_names` in the same
            order as the ones used to train the model. If ``None`` and ``dataloader`` is also
            ``None``, it defaults to the object used to initialize the model.
        indices
            Indices of observations in ``adata`` to use. If ``None``, defaults to all observations.
            Ignored if ``dataloader`` is not ``None``
        give_mean
            If ``True``, returns the mean of the latent distribution. If ``False``, returns an
            estimate of the mean using ``mc_samples`` Monte Carlo samples.
        mc_samples
            Number of Monte Carlo samples to use for the estimator for distributions with no
            closed-form mean (e.g., the logistic normal distribution). Not used if ``give_mean`` is
            ``True`` or if ``return_dist`` is ``True``.
        batch_size
            Minibatch size for the forward pass. If ``None``, defaults to
            ``scvi.settings.batch_size``. Ignored if ``dataloader`` is not ``None``
        return_dist
            If ``True``, returns the mean and variance of the latent distribution. Otherwise,
            returns the mean of the latent distribution.
        dataloader
            An iterator over minibatches of data on which to compute the metric. The minibatches
            should be formatted as a dictionary of :class:`~torch.Tensor` with keys as expected by
            the model. If ``None``, a dataloader is created from ``adata``.

        Returns
        -------
        An array of shape ``(n_obs, n_latent)`` if ``return_dist`` is ``False``. Otherwise, returns
        a tuple of arrays ``(n_obs, n_latent)`` with the mean and variance of the latent
        distribution.
        """
        from torch.distributions import Normal
        from torch.nn.functional import softmax

        from scvi.module._constants import MODULE_KEYS

        self._check_if_trained(warn=False)
        if adata is not None and dataloader is not None:
            raise ValueError("Only one of `adata` or `dataloader` can be provided.")

        if dataloader is None:
            adata = self._validate_anndata(adata)
            dataloader = self._make_data_loader(
                adata=adata, indices=indices, batch_size=batch_size
            )

        zs: list[Tensor] = []
        qz_means: list[Tensor] = []
        qz_vars: list[Tensor] = []
        for tensors in dataloader:
            outputs: dict[str, Tensor | Distribution | None] = self.module.inference(
                **self.module._get_inference_input(tensors)
            )

            if modality is not None and modality != 'joint':
                outputs = outputs['modalities'][self.modality_key_to_index[modality]]

            if MODULE_KEYS.QZ_KEY in outputs:
                qz: Distribution = outputs.get(MODULE_KEYS.QZ_KEY)
                qzm: Tensor = qz.loc
                qzv: Tensor = qz.scale.square()
            else:
                qzm: Tensor = outputs.get(MODULE_KEYS.QZM_KEY)
                qzv: Tensor = outputs.get(MODULE_KEYS.QZV_KEY)
                qz: Distribution = Normal(qzm, qzv.sqrt())

            if return_dist:
                qz_means.append(qzm.cpu())
                qz_vars.append(qzv.cpu())
                continue

            z: Tensor = qzm if give_mean else outputs.get(MODULE_KEYS.Z_KEY)

            if give_mean and getattr(self.module, "latent_distribution", None) == "ln":
                samples = qz.sample([mc_samples])
                z = softmax(samples, dim=-1).mean(dim=0)

            zs.append(z.cpu())

        if return_dist:
            return torch.cat(qz_means).numpy(), torch.cat(qz_vars).numpy()
        else:
            return torch.cat(zs).numpy()

    
    def get_normalized_expression(self, adata=None, indices = None, batch_size = None, n_samples=1, return_mean=True, return_px_distrs=True, return_l2_error=True, return_numpy=False):
        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")
        
        if adata is None:
            adata = self.adata
        
        if n_samples > 1 and not return_mean and not return_numpy:
            raise RuntimeError('If generating with n_samples > 1 and return_mean=False, return type must be numpy')

        adata = self._validate_anndata(adata)
        dataloader = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        ret = {'exprs': {key: [] for key in self.modalities}}
        if return_l2_error:
            ret['errors'] = {key: [] for key in self.modalities}
        if return_px_distrs:
            ret['distrs'] = {key: DistributionConcatenator() for key in self.modalities}
    
        with torch.no_grad():
            for tensors in dataloader:
                inference_inputs = self.module._get_inference_input(tensors)
                inference_outputs = self.module.inference(**inference_inputs, n_samples=n_samples)
                # if use_mean_latent:
                #     inference_outputs['z'] = inference_outputs['qzm']
                generative_inputs = self.module._get_generative_input(tensors=tensors, inference_outputs=inference_outputs)
                generative_outputs = self.module.generative(**generative_inputs)
                for mod, i in self.modality_key_to_index.items():
                    distr = generative_outputs['px_distrs'][i]

                    # if return_mean_expression:
                    #     generated_tensor = distr.mean
                    # else:
                    generated_tensor = distr.sample()

                    ret['exprs'][mod] += [generated_tensor.cpu()]

                    if return_l2_error:
                        input_tensor = inference_inputs['inputs'][self.modality_key_to_index[mod]]
                        if n_samples > 1:
                            input_tensor = input_tensor.unsqueeze(0).repeat_interleave(n_samples, dim=0)
                        ret['errors'][mod] += [torch.square(generated_tensor.cpu() - input_tensor.cpu())]
                    if return_px_distrs:
                        ret['distrs'][mod].store_distribution(distr)

        mean_for = ['exprs', 'errors']
        pd_for = ['exprs', 'errors']
        for mod in self.modalities:
            for ret_key, ret_dict in ret.items():
                # -2 is cell dimension (whether or not n_samples > 1)
                if ret_key == 'distrs':
                    ret_dict[mod] = ret_dict[mod].get_concatenated_distributions(axis=-2)
                else:
                    ret_dict[mod] = torch.cat(ret_dict[mod], dim=-2).numpy()
    
                if return_mean and n_samples > 1 and ret_key in mean_for:
                    ret_dict[mod] = ret_dict[mod].mean(axis=0)
                
                if not return_numpy and ret_key in pd_for:
                    df = self.modality_key_to_df[mod]
                    ret_dict[mod] = pd.DataFrame(ret_dict[mod], columns=df.columns, index=df.index)

        return ret


    # @torch.inference_mode()
    # def get_normalized_expression(
    #     self,
    #     adata: Optional[AnnData] = None,
    #     indices: Optional[Sequence[int]] = None,
    #     modalities=None,
    #     transform_batch: Optional[Sequence[Union[Number, str]]] = None,
    #     n_samples: int = 1,
    #     n_samples_overall: int = None,
    #     weights: Union[Literal["uniform", "importance"], None] = None,
    #     batch_size: Optional[int] = None,
    #     return_mean: bool = True,
    #     return_numpy: Optional[bool] = None,
    #     nan_warning: Optional[bool] = True,
    #     **importance_weighting_kwargs,
    # ) -> dict:
    #     r"""Returns the normalized (decoded) protein expression.

    #     This is denoted as :math:`\rho_n` in the scVI paper.

    #     Parameters
    #     ----------
    #     adata
    #         AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
    #         AnnData object used to initialize the model.
    #     indices
    #         Indices of cells in adata to use. If `None`, all cells are used.
    #     modalities
    #         List of modalities to return. If None, all modalities are used
    #     transform_batch
    #         Batch to condition on.
    #         If transform_batch is:

    #         - 'all', then the mean across batches is used
    #         - None, then real observed batch is used.
    #         - int, then batch transform_batch is used.
    #         This behaviour affects only proteins that are detected across multiple batches.
    #         Unobserved proteins are decoded in the batch(es), in which they were measured.
    #     n_samples
    #         Number of posterior samples to use for estimation.
    #     n_samples_overall
    #         Number of posterior samples to use for estimation. Overrides `n_samples`.
    #     weights
    #         Weights to use for sampling. If `None`, defaults to `"uniform"`.
    #     batch_size
    #         Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
    #     return_mean
    #         Whether to return the mean of the samples.
    #     return_numpy
    #         Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
    #         gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
    #         Otherwise, it defaults to `True`.

    #     Returns
    #     -------
    #     If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
    #     Otherwise, shape is `(cells, genes)`. In this case, return type is dict of str (modality) to :class:`~pandas.DataFrame` unless `return_numpy` is True.
    #     """
    #     adata = self._validate_anndata(adata)
    #     all_batches = list(np.unique(self.adata_manager.get_from_registry("batch")))

    #     if indices is None:
    #         indices = np.arange(adata.n_obs)
    #     if n_samples_overall is not None:
    #         assert n_samples == 1  # default value
    #         n_samples = n_samples_overall // len(indices) + 1
    #     scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

    #     if n_samples > 1 and return_mean is False:
    #         if return_numpy is False:
    #             msg = "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray"
    #             warnings.warn(msg, UserWarning, stacklevel=settings.warnings_stacklevel)
    #         return_numpy = True
        
    #     if transform_batch == "all":
    #         transform_batch = all_batches
    #     else:
    #         transform_batch = _get_batch_code_from_category(
    #             self.get_anndata_manager(adata, required=True), transform_batch
    #         )
        

    #     qz_store = DistributionConcatenator()
    #     zs = []

    #     modality_epxrs_zs = {
    #         key: {'exprs': [], 'zs': [], 'px_store': DistributionConcatenator()} 
    #         for key in self.modalities
    #     }

    #     # exprs = []
    #     # zs = []
    #     # qz_store = DistributionConcatenator()
    #     # px_store = DistributionConcatenator()
    #     for tensors in scdl:
    #         per_batch_exprs = {key: [] for key in self.modalities}
    #         for batch in transform_batch:
    #             generative_kwargs = self._get_transform_batch_gen_kwargs(batch)
    #             inference_kwargs = {"n_samples": n_samples}
    #             inference_outputs, generative_outputs = self.module.forward(
    #                 tensors=tensors,
    #                 inference_kwargs=inference_kwargs,
    #                 generative_kwargs=generative_kwargs,
    #                 compute_loss=False,
    #             )

    #             for i, px_distrs in enumerate(generative_outputs['px_distrs']):
    #                 output = px_distrs.mean
    #                 output = output.cpu().numpy()

    #                 per_batch_exprs[self.modalities[i]].append(output)
    #                 modality_epxrs_zs['px_store'].store_distribution(px_distrs)

    #             qz_store.store_distribution(inference_outputs["qz"])
    #         zs.append(inference_outputs["z"].cpu())
    #         per_batch_exprs = {key: np.stack(mod_expr) for key, mod_expr in per_batch_exprs.items()}

    #         exprs += [per_batch_exprs.mean(axis=0)]

    #     if n_samples > 1:
    #         # The -2 axis correspond to cells.
    #         exprs = np.concatenate(exprs, axis=-2)
    #         zs = torch.concat(zs, dim=-2)
    #     else:
    #         exprs = np.concatenate(exprs, axis=0)
    #         zs = torch.concat(zs, dim=0)

    #     if n_samples_overall is not None:
    #         # Converts the 3d tensor to a 2d tensor
    #         exprs = exprs.reshape(-1, exprs.shape[-1])
    #         n_samples_ = exprs.shape[0]
    #         if (weights is None) or weights == "uniform":
    #             p = None
    #         else:
    #             qz = qz_store.get_concatenated_distributions(axis=0)
    #             x_axis = 0 if n_samples == 1 else 1
    #             px = px_store.get_concatenated_distributions(axis=x_axis)
    #             p = self._get_importance_weights(
    #                 adata,
    #                 indices,
    #                 qz=qz,
    #                 px=px,
    #                 zs=zs,
    #                 **importance_weighting_kwargs,
    #             )
    #         ind_ = np.random.choice(n_samples_, n_samples_overall, p=p, replace=True)
    #         exprs = exprs[ind_]
    #     elif n_samples > 1 and return_mean:
    #         exprs = exprs.mean(axis=0)

    #     if return_numpy is None or return_numpy is False:
    #         return pd.DataFrame(
    #             exprs,
    #             columns=adata.var_names,
    #             index=adata.obs_names[indices],
    #         )
    #     else:
    #         return exprs
    
    # def get_l2_error(self, adata=None, **normalized_expression_args):
    #     normalized_expression = self.get_normalized_expression(adata, **normalized_expression_args)

    #     if adata is None:
    #         adata = self.adata
    #     l2_errors = {}
    #     inputs = self._module.get_inference_inputs()
    #     for mod, generated in normalized_expression.items():
    #         l2_errors



# class NormalVAEPyro(PyroBaseModuleClass):
#     def __init__(self, input_d, latent_d, hidden_d, n_hidden,):
#         super().__init__()
#         self.latent_d = latent_d
#         self.input_d = input_d
#         # in the init, we create the parameters of our elementary stochastic computation unit.

#         # First, we setup the parameters of the generative model
#         self.decoder = LinearNetWithTransform(input_d=latent_d, output_d=input_d, hidden_d=hidden_d, n_hidden=n_hidden, transform="none")
#         self.log_theta = torch.nn.Parameter(torch.randn(input_d))

#         # Second, we setup the parameters of the variational distribution
#         self.mean_encoder = LinearNetWithTransform(input_d=input_d, output_d=latent_d, hidden_d=hidden_d, n_hidden=n_hidden, transform="none")
#         self.var_encoder = LinearNetWithTransform(input_d=input_d, output_d=latent_d, hidden_d=hidden_d, n_hidden=n_hidden, transform="exp")

#     @staticmethod
#     def _get_fn_args_from_batch(tensor_dict):
#         x = tensor_dict[REGISTRY_KEYS.X_KEY]
#         return (x, ), {}

#     def model(self, x, library):
#         # register PyTorch module `decoder` with Pyro
#         pyro.module("NormalVAE", self)
#         with pyro.plate("data", x.shape[0]):
#             # setup hyperparameters for prior p(z)
#             z_loc = x.new_zeros(torch.Size((x.shape[0], self.latent_d)))
#             z_scale = x.new_ones(torch.Size((x.shape[0], self.latent_d)))
#             # sample from prior (value will be sampled by guide when computing the ELBO)
#             z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
#             # get the "normalized" mean of the negative binomial

#             decoded_loc = self.decoder(z)


#             # get the mean of the negative binomial
#             # px_rate = library * px_scale
#             # get the dispersion parameter
#             theta = torch.exp(self.log_theta)
#             # build count distribution
#             # nb_logits = (px_rate + 1e-4).log() - (theta + 1e-4).log()
#             # x_dist = dist.NegativeBinomial(total_count=theta, logits=nb_logits)

#             # score against actual counts
#             x_dist = dist.Normal(loc=decoded_loc, scale=z_scale),
#             pyro.sample("obs", x_dist.to_event(1), obs=x)

#     def guide(self, x):
#         # define the guide (i.e. variational distribution) q(z|x)
#         pyro.module("NormalVAE", self)
#         with pyro.plate("data", x.shape[0]):
#             # use the encoder to get the parameters used to define q(z|x)
#             qz_m = self.mean_encoder(x)
#             qz_v = self.var_encoder(x)
#             # sample the latent code z
#             pyro.sample("latent", dist.Normal(qz_m, torch.sqrt(qz_v)).to_event(1))




# class LinearNetWithTransform(torch.nn.Module):
#     def __init__(
#         self,
#         input_d: int,
#         hidden_d: int,
#         output_d: int,
#         n_hidden: int,
#         transform: Literal["exp", "none", "softmax"],
#     ):
#         """Encodes data of ``n_input`` dimensions into a space of ``n_output`` dimensions.

#         Uses a one layer fully-connected neural network with 128 hidden nodes.

#         Parameters
#         ----------
#         n_input
#             The dimensionality of the input.
#         n_output
#             The dimensionality of the output.
#         link_var
#             The final non-linearity.
#         """
#         super().__init__()
#         self.neural_net = torch.nn.Sequential(
#             torch.nn.Linear(input_d, hidden_d),
#             torch.nn.ReLU(),
#             *(  [
#                     torch.nn.Linear(hidden_d, hidden_d),
#                     torch.nn.ReLU(),
#                 ]*(n_hidden - 1)
#             ),
#             torch.nn.Linear(hidden_d, output_d),
#         )
#         self.transformation = None
#         if transform == "softmax":
#             self.transformation = torch.nn.Softmax(dim=-1)
#         elif transform == "exp":
#             self.transformation = torch.exp

#     def forward(self, x: torch.Tensor):
#         output = self.neural_net(x)
#         if self.transformation:
#             output = self.transformation(output)
#         return output


