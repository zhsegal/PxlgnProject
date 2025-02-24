import tempfile
from collections.abc import Iterable, Sequence
from typing import Callable, Literal, Optional

import pyro
import pyro.distributions as dist
import scvi
import torch
from torch import nn
import torch.nn.functional as F

from scvi import REGISTRY_KEYS
from scvi.module.base import (
    BaseModuleClass,
    LossOutput,
    PyroBaseModuleClass,
    auto_move_data,
)
from scvi.nn import FCLayers, Encoder
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

from torch.distributions import NegativeBinomial, Normal
from torch.distributions import kl_divergence as kl

MAX_OBSM_MODALITIES = 16
REGISTRY_OBSM_MODALITY_KEYS = [f'modality_{i}' for i in range(MAX_OBSM_MODALITIES)]


def _identity(x):
    return x

class LinearNetWithTransform(torch.nn.Module):
    def __init__(
        self,
        input_d: int,
        hidden_d: int,
        output_d: int,
        n_hidden: int,
        transform: Literal["exp", "none", "softmax"],
    ):
        """Encodes data of ``n_input`` dimensions into a space of ``n_output`` dimensions.

        Uses a one layer fully-connected neural network with 128 hidden nodes.

        Parameters
        ----------
        n_input
            The dimensionality of the input.
        n_output
            The dimensionality of the output.
        link_var
            The final non-linearity.
        """
        super().__init__()
        self.neural_net = torch.nn.Sequential(
            torch.nn.Linear(input_d, hidden_d),
            torch.nn.ReLU(),
            *(  [
                    torch.nn.Linear(hidden_d, hidden_d),
                    torch.nn.ReLU(),
                ]*(n_hidden - 1)
            ),
            torch.nn.Linear(hidden_d, output_d),
        )
        self.transformation = None
        if transform == "softmax":
            self.transformation = torch.nn.Softmax(dim=-1)
        elif transform == "exp":
            self.transformation = torch.exp

    def forward(self, x: torch.Tensor):
        output = self.neural_net(x)
        if self.transformation:
            output = self.transformation(output)
        return output


class DecoderNormalVAE(nn.Module):
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
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate=0,
        inject_covariates: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        decoder_param_eps: float = 1e-6,
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

        self.loc_decoder = nn.Linear(n_hidden, n_output)
        self.scale_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self,
        z: torch.Tensor,
        *cat_list: int,
    ):
        # The decoder returns values for the parameters of the emission distribution
        px = self.px_decoder(z, *cat_list)

        loc = self.loc_decoder(px)
        scale = self.scale_decoder(px)

        # Make scale positive
        scale = F.softplus(scale) + self.decoder_param_eps

        return loc, scale



class EncoderNormalVAE(nn.Module):
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
        var_eps: float = 1e-4,
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
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.return_dist = return_dist
        self.var_activation = torch.exp if var_activation is None else var_activation

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
        # Parameters for latent distribution
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
        dist = Normal(q_m, q_v.sqrt())
        latent = dist.rsample()
        if self.return_dist:
            return dist, latent
        return q_m, q_v, latent


class NormalVAE(BaseModuleClass):
    """
    Basic NormalVAE with varying number of modalities
    """

    def __init__(self, input_d, latent_d, hidden_d, n_hidden, n_batch, n_extra_modalities=0, weights=[1,], dropout_rate=0.05, use_batch_norm=True):
        super().__init__()
        self.latent_d = latent_d
        self.n_extra_modalities = n_extra_modalities

        if n_extra_modalities > 0:
            assert isinstance(input_d, Sequence) and torch.all(torch.tensor([len(weights), len(input_d)]) == n_extra_modalities + 1)
            self.input_ds = input_d
            assert isinstance(weights, torch.Tensor)
            self.weights = weights
        else:
            self.input_ds = [input_d]
            self.weights = torch.tensor([1,], dtype=float)


        common_kwargs = dict(n_hidden=hidden_d, n_layers=n_hidden, use_batch_norm=use_batch_norm)

        self.encoders = nn.ModuleList(
            [
                EncoderNormalVAE(n_input=input_d, n_output=latent_d, dropout_rate=dropout_rate, **common_kwargs)
                for input_d in self.input_ds
            ]
        )
        self.decoders = nn.ModuleList(
            [
            DecoderNormalVAE(n_input=latent_d, n_output=input_d, n_cat_list=[n_batch], dropout_rate=0, **common_kwargs)
            for input_d in self.input_ds
            ]
        )


    def _extra_mod_s(self, i):
        return f'extra_mod_{i}'
    
    def _weighted_sum(self, tensors, weights):
        n = len(weights)
        device = tensors[0].device
        dtype = tensors[0].dtype
        weights = weights.view(n, *([1]*len(tensors[0].shape))).to(dtype=dtype).to(device)
        return torch.sum(torch.stack(tensors, dim=0)*weights, dim=0)

    def _get_inference_input(self, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Parse the dictionary to get appropriate args"""
        # let us fetch the data, and add it to the dictionary
        return {
                "x": tensors[REGISTRY_KEYS.X_KEY], 
                "inputs": [tensors[REGISTRY_KEYS.X_KEY]] + [tensors[REGISTRY_OBSM_MODALITY_KEYS[i]] for i in range(self.n_extra_modalities)],
                # **{f'{self._extra_mod_s(i)}': tensors[REGISTRY_OBSM_MODALITY_KEYS[i]] for i in range(1, self.n_modalities + 1)},
                }

    @auto_move_data
    def inference(self, x: torch.Tensor, inputs) -> dict[str, torch.Tensor]:
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        inference_results = [
            encoder(input) for input, encoder in zip(inputs, self.encoders)
        ]
        # for input, encoder in zip(inputs, self.encoders):
        #     inference_results.append(encoder(input))


        means, vars, zs = zip(*inference_results)
        qzm = self._weighted_sum(means, self.weights)
        qzv = self._weighted_sum(vars, torch.square(self.weights))
        z = self._weighted_sum(zs, self.weights)

        return {"qzm": qzm, "qzv": qzv, "z": z, 'means_list': means, 'vars_list': vars, 'zs_list': zs,
                'inputs': inputs}


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
    def generative(self, z: torch.Tensor, batch_index: torch.Tensor, sample=False) -> dict[str, torch.Tensor]:
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

        decoder_results = []
        samples = []
        for decoder in self.decoders:
            results = (loc, scale) = decoder(z, batch_index)
            decoder_results.append(results)
            if sample:
                samples.append(Normal(loc=loc, scale=scale).sample())
        return {
            'px_params': decoder_results,
            'samples': samples,
        }


    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor],
        generative_outputs: dict[str, torch.Tensor],
        kl_weight: float=1.0
    ) -> LossOutput:
        
        # KL loss
        qzm = inference_outputs["qzm"]
        qzv = inference_outputs["qzv"]

        prior_dist = Normal(torch.zeros_like(qzm), torch.ones_like(qzv))
        var_post_dist = Normal(qzm, torch.sqrt(qzv))
        kl_divergence = kl(var_post_dist, prior_dist).sum(dim=-1)
    
        inputs = inference_outputs['inputs']
        log_lik = 0
        for input, (px_loc, px_scale) in zip(inputs, generative_outputs['px_params']):
            # x = tensors[REGISTRY_KEYS.X_KEY]
            mod_log_lik = Normal(loc=px_loc, scale=px_scale).log_prob(input).sum(dim=-1)
            log_lik += mod_log_lik

        elbo = log_lik - kl_weight*kl_divergence
        loss = torch.mean(-elbo)

        return LossOutput(
            loss=loss,
            reconstruction_loss=-log_lik,
            kl_local=kl_divergence,
            kl_global=0.0,
        )


class NormalSCVI(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """single-cell Variational Inference [Lopez18]_."""

    def __init__(
        self,
        adata: AnnData,
        n_latent: int = 10,
        n_hidden: int = 128,
        n_layers: int = 1,
        n_extra_modalities = 0,
        obsm_extra_modality_keys = [],
        weights = [1,],
        **model_kwargs,
    ):
        super().__init__(adata)

        assert n_extra_modalities + 1 == len(weights), "Number of extra modalities + 1 must match length of weights list"
        assert n_extra_modalities == len(obsm_extra_modality_keys), "Number of extra modalities must match length of obsm extra modality keys list"

        self.weights = torch.tensor(weights, dtype=float)
        self.weights /= self.weights.sum()

        self.n_extra_modalities = n_extra_modalities
        self.obsm_extra_modality_keys = obsm_extra_modality_keys

        if n_extra_modalities == 0:
            input_d = self.summary_stats['n_vars']
        else:
            input_d = [self.summary_stats['n_vars']] + [adata.obsm[k].shape[1] for k in obsm_extra_modality_keys]


        self.module = NormalVAE(
            input_d=input_d,
            latent_d=n_latent,
            hidden_d=n_hidden,
            n_hidden=n_layers,
            n_extra_modalities=self.n_extra_modalities,
            weights=self.weights,
            n_batch=self.summary_stats["n_batch"],
            **model_kwargs,
        )
        self._model_summary_string = (
            f"Normal SCVI Model with the following params: \nn_latent: {n_latent}\nn_extra_modalities: {self.n_extra_modalities}\nweights: {self.weights}\ninput_ds: {self.module.input_ds}"
        )
        self.init_params_ = self._get_init_params(locals())


    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        batch_key: str | None = None,
        layer: str | None = None,
        obsm_extra_modality_keys = [],
        is_count_data: bool = False,
        **kwargs,
    ) -> AnnData | None:
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=is_count_data),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            # Dummy fields required for VAE class.
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, None),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, None, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, None),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, None),
        ] + [
            ObsmField(REGISTRY_OBSM_MODALITY_KEYS[i], key) for i, key in enumerate(obsm_extra_modality_keys)
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
    
    def get_normalized_expression(self, adata=None, indices = None, batch_size = None, return_mean=True):
        raise NotImplementedError()
        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")
        
        if adata is None:
            adata = self.adata

        adata = self._validate_anndata(adata)
        dataloader = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        generated = []
        with torch.no_grad():
            for tensors in dataloader:
                inference_inputs = self.module._get_inference_input(tensors)
                inference_outputs = self.module.inference(**inference_inputs)
                if return_mean:
                    inference_outputs['z'] = inference_outputs['qzm']
                generative_inputs = self.module._get_generative_input(tensors=tensors, inference_outputs=inference_outputs)
                generative_outputs = self.module.generative(**generative_inputs, sample=not return_mean)
                if return_mean:
                    generated_tensor = generative_outputs['px_loc']
                else:
                    generated_tensor = generative_outputs['sample']

                generated += [generated_tensor.cpu()]
        return torch.cat(generated).numpy()



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


