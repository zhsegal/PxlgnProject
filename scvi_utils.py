import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import torch

def plot_losses(model):

    # n_epochs = len(model.history['validation_loss'])
    # validation_losses = ['reconstruction_loss_validation', 'kl_local_validation']
    # training_losses = ['reconstruction_loss_train', 'kl_local_train']

    # validation_df = pd.melt(pd.DataFrame({l: model.history[l][l] for l in validation_losses}).reset_index(), id_vars=['epoch'], value_name='Loss', var_name='Loss Type')
    # train_df = pd.melt(pd.DataFrame({l: model.history[l][l] for l in training_losses}).reset_index(), id_vars=['epoch'], value_name='Loss', var_name='Loss Type')
    # pd.set_option("future.no_silent_downcasting", True)
    # validation_df.replace({'reconstruction_loss_validation': 'Reconstruction', 'kl_local_validation': 'KL'}, inplace=True)
    # train_df.replace({'reconstruction_loss_train': 'Reconstruction', 'kl_local_train': 'KL'}, inplace=True)
    fig, ax = plt.subplots(2, 5, figsize=(18, 6), sharex=True, sharey='col')
    ax[0][0].set_title('Reconstruction')
    ax[0][1].set_title('KL*Weight')
    ax[0][2].set_title('KL')
    ax[0][3].set_title('Loss')
    ax[0][4].set_title('ELBO')

    n_epochs = len(model.history['validation_loss'])

    # if 'kl_weight' in model.history:
    #     kl_weight = model.history['kl_weight']
    # else:
    #     if kl_warmup == 0:
    #         kl_weight = torch.ones(shape=(n_epochs,))
    #     else:
    #         kl_weight = np.arange(start=0)

    for i, eval_set in enumerate(('train', 'validation')):
        full_loss_key = {'validation': 'validation_loss', 'train': 'train_loss_epoch'}[eval_set]
        for j, loss_type in enumerate(('reconstruction_loss', 'kl*weight', 'kl_local', 'full_loss', 'elbo')):
            if loss_type == 'kl*weight':
                if 'kl_weight' in model.history:
                    kl_weight = model.history['kl_weight']['kl_weight']
                    key = f'kl_local_{eval_set}'
                    loss = model.history[key][key]*kl_weight
                else:
                    recon_key = f'reconstruction_loss_{eval_set}'
                    loss = model.history[full_loss_key][full_loss_key] - model.history[recon_key][recon_key]
            elif loss_type == 'full_loss':
                key = full_loss_key
                loss = model.history[key][key]
            else:
                key = f'{loss_type}_{eval_set}'
                loss = model.history[key][key]
            sns.lineplot(loss, ax=ax[i][j], color=sns.color_palette()[j])
            ax[i][j].set_ylabel(None)
    
        ax[i][0].set_ylabel(eval_set)

    # sns.lineplot(validation_df, x='epoch', y='Loss', hue='Loss Type', ax=ax[0])
    # sns.lineplot(train_df, x='epoch', y='Loss', hue='Loss Type', ax=ax[1])
    # ax[0].set_title('Validation')
    # ax[1].set_title('Train')
    return ax