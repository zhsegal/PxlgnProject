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

from anndata import AnnData

from scvi import REGISTRY_KEYS, settings
from scvi.module.base import (
    BaseModuleClass,
    LossOutput,
    PyroBaseModuleClass,
    auto_move_data,
)
from scvi.model._utils import _get_batch_code_from_category
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

from scvi.data import AnnDataManager

from torch.distributions import NegativeBinomial, Normal, Beta
from torch.distributions import kl_divergence as kl

import sys
sys.path.append('/home/projects/nyosef/zvise/PixelGen/PixelGen')

from multimodalvae import MultiModalVAE
from enums import AggMethod
from scvi_utils import DistributionConcatenator

import warnings


MAX_MODALITIES = 16
REGISTRY_MODALITY_KEYS = [f'modality_{i}' for i in range(MAX_MODALITIES)]


class MultiModalSCVI(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass):
    """Multi-modal version of SCVI."""

    def __init__(
        self,
        adata: AnnData,
        n_latent: int = 20,
        n_hidden: int = 128,
        n_layers: int = 1,
        **model_kwargs,
    ):
        super().__init__(adata)

        self.extra_modality_keys = self.get_setup_arg('extra_modality_keys')
        self.n_modalities = self.get_setup_arg('n_modalities')
        self.layer = self.get_setup_arg('layer')
        self.main_modality = self.layer or 'X'

        self.modality_key_to_index = {
            self.main_modality: 0,
            **{key: i+1 for i, key in enumerate(self.extra_modality_keys)}
        }

        self.modality_key_to_registry_key = {
            self.main_modality: REGISTRY_KEYS.X_KEY,
            **{key: REGISTRY_MODALITY_KEYS[i+1] for i, key in enumerate(self.extra_modality_keys)}
        }

        self.modality_key_to_df = {
            self.main_modality: adata.to_df(self.layer),
            **{key: adata.obsm[key] for key in self.extra_modality_keys}
        }

        self.modalities = self.modality_key_to_index.keys()



        if self.n_modalities == 1:
            self.input_ds = [adata.to_df(self.layer).shape[1], ]
        else:
            self.input_ds = [adata.to_df(self.layer).shape[1],] + [adata.obsm[k].shape[1] for k in self.extra_modality_keys]


        self.module = MultiModalVAE(
            input_ds=self.input_ds,
            n_latent=n_latent,
            n_hidden=n_hidden,
            n_layers=n_layers,
            n_batch=self.summary_stats["n_batch"],
            **model_kwargs,
        )
        self._model_summary_string = (
            f"Normal SCVI Model with the following params: \nn_latent: {n_latent}\nn_modalities: {self.n_modalities}\nmodalities: {self.modalities}\nagg_method: {self.module.agg_method}\ninput_ds: {self.input_ds}\nmodality indices: {self.modality_key_to_index}\ndistributions: {self.module.distrs}"
        )
        self.init_params_ = self._get_init_params(locals())


    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        batch_key: str | None = None,
        layer: str | None = None,
        n_modalities = 1,
        extra_modality_keys = [],
        is_count_data: bool = False,
        spatial_mask_key: str | None = None,
        **kwargs,
    ) -> AnnData | None:
        '''
        param extra_modality_keys: Sequence of keys in obsm for "extra" modalities (the main modality is passed through 'layer'). 
                                        The choice of a main modality is arbitrary and it has no special function.
        '''
        setup_method_args = cls._get_setup_method_args(**locals())

        assert n_modalities - 1 == len(extra_modality_keys), "Number of modalities - 1 must match length of obsm extra modality keys list"

        for key in extra_modality_keys:
            if key not in adata.obsm or not isinstance(adata.obsm[key], pd.DataFrame):
                raise RuntimeError('Each modality key must be a key to adata.obsm and the value must be a pandas Dataframe')

        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=is_count_data),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            # Dummy fields required for VAE class.
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, None),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, None, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, None),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, None),
        ] + [
            ObsmField(REGISTRY_MODALITY_KEYS[i+1], key) for i, key in enumerate(extra_modality_keys)
        ]
         # 🟩 NEW: register spatial mask if provided
        if spatial_mask_key is not None:
            if spatial_mask_key not in adata.obs:
                raise KeyError(f"spatial_mask_key '{spatial_mask_key}' not found in adata.obs.")
            anndata_fields.append(
                NumericalObsField("spatial_masked", spatial_mask_key, required=True)
            )
        
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
    
    def get_setup_arg(self, arg):
        manager = self.get_anndata_manager(self.adata)
        return manager._get_setup_method_args()['setup_args'][arg]
    
    def get_modality_df_dict(self):
        return self.modality_key_to_df

    def get_modality_df(self, modality):
        df = self.modality_key_to_df.get(modality)
        if not df:
            raise ValueError('Unregistered modality')
        return df
        
    
    @torch.inference_mode()
    def get_weights(        
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        batch_size: int | None = None,
        dataloader: Iterator[dict[str, Tensor | None]] = None,
    ):
        '''
        Returns numpy array of shape (n_modalities,) or (n_points, n_modalities) for global or local weights respectively
        '''
        self._check_if_trained(warn=False)

        global_weights = self.module.get_global_weights()
        if global_weights is not None:
            return global_weights

        if adata is not None and dataloader is not None:
            raise ValueError("Only one of `adata` or `dataloader` can be provided.")

        if dataloader is None:
            adata = self._validate_anndata(adata)
            dataloader = self._make_data_loader(
                adata=adata, indices=indices, batch_size=batch_size
            )

        weights = []
        for tensors in dataloader:
            weights.append(
                self.module.get_per_cell_weights(self.module._get_inference_input(tensors)['inputs'], grad=False, numpy=True)
            )
        return np.concatenate(weights, axis=0)


    @torch.inference_mode()
    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        give_mean: bool = True,
        modality: str = 'joint',
        mc_samples: int = 5_000,
        batch_size: int | None = None,
        return_dist: bool = False,
        dataloader: Iterator[dict[str, Tensor | None]] = None,
    ) -> npt.NDArray | tuple[npt.NDArray, npt.NDArray]:
        """Compute the latent representation of the data.

        This implementation adds the modality argument. Default ('joint') is the combined latent. For the main modality pass the layer kwarg used
        to initialize the model (or 'X' if layer=None was passed)

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

            if modality != 'joint':
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

    
    def get_normalized_expression(
            self, 
            adata=None, 
            indices = None, 
            batch_size = None, 
            use_mean_latent=True,
            return_mean_expression=False,
            return_px_distrs=True, 
            return_l2_error=True, 
            return_numpy=False,
            # n_samples=1,
            # return_mean=True, 

        ):
        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")
        
        if adata is None:
            adata = self.adata
        
        n_samples = 1
        # if n_samples > 1 and not return_mean and not return_numpy:
        #     raise RuntimeError('If generating with n_samples > 1 and return_mean=False, return type must be numpy')

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
                if use_mean_latent:
                    inference_outputs['z'] = inference_outputs['qzm']
                generative_inputs = self.module._get_generative_input(tensors=tensors, inference_outputs=inference_outputs)
                generative_outputs = self.module.generative(**generative_inputs)
                for mod, i in self.modality_key_to_index.items():
                    distr = generative_outputs['px_distrs'][i]

                    if return_mean_expression:
                        generated_tensor = distr.mean
                    else:
                        generated_tensor = distr.sample()

                    ret['exprs'][mod] += [generated_tensor.cpu()]

                    if return_l2_error:
                        input_tensor = inference_inputs['inputs'][self.modality_key_to_index[mod]]
                        # if n_samples > 1:
                        #     input_tensor = input_tensor.unsqueeze(0).repeat_interleave(n_samples, dim=0)
                        ret['errors'][mod] += [torch.square(generated_tensor.cpu() - input_tensor.cpu())]
                    if return_px_distrs:
                        ret['distrs'][mod].store_distribution(distr)

        mean_for = ['exprs', 'errors']
        pd_for = ['exprs', 'errors']
        for mod in self.modalities:
            for ret_key, ret_dict in ret.items():
                # cell dimension = 0 if n_samples == 1 , else 1
                if ret_key == 'distrs':
                    ret_dict[mod] = ret_dict[mod].get_concatenated_distributions(axis= int(n_samples > 1))
                else:
                    ret_dict[mod] = torch.cat(ret_dict[mod], dim= int(n_samples > 1)).numpy()
    
                # if return_mean and n_samples > 1 and ret_key in mean_for:
                #     ret_dict[mod] = ret_dict[mod].mean(axis=0)
                
                if not return_numpy and ret_key in pd_for:
                    df = self.modality_key_to_df[mod]
                    ret_dict[mod] = pd.DataFrame(ret_dict[mod], columns=df.columns, index=df.index)

        return ret


