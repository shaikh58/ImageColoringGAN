import os
import numpy as np
from tqdm.notebook import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from utils import *
from models import *
from dataset import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
colored_imgs_folder = 'coloured'
original_imgs_folder = 'grayscale'


def pretrain_generator(net_G, train_dl, epochs):
    opt = optim.Adam(net_G.parameters(), lr=1e-4)
    criterion = nn.L1Loss()        
    for e in range(epochs):
        loss_meter = AverageMeter()
        for data in train_dl:
            L, ab = data['L'].to(device), data['ab'].to(device)
            preds = net_G(L)
            loss = criterion(preds, ab)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            loss_meter.update(loss.item(), L.size(0))
            
        print(f"Epoch {e + 1}/{epochs}")
        print(f"L1 Loss: {loss_meter.avg:.5f}")
        if e%5==0 or e==epochs-1:
            torch.save(net_G.state_dict(), "res18-unet.pt")


def train_model(model, train_dl, val_dl, lr_G, lr_D, epochs, beta1, beta2, lambda_L1, save_freq=10, display_every=200):
    GANcriterion = nn.BCEWithLogitsLoss()
    L1criterion = nn.L1Loss()
    opt_G = optim.Adam(model.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
    opt_D = optim.Adam(model.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))

    losses_iter_dict =  {'loss_D': [],
                         'loss_G': []}
    losses_epoch_dict =  {'loss_D': [],
                    'loss_G': []}

    os.makedirs('progress', exist_ok=True)
    val_data = next(iter(val_dl)) 
    for e in range(epochs):
        epoch_loss_D = 0
        epoch_loss_G = 0
        loss_meter_dict = create_loss_meters() 
        for batch_idx, data in enumerate(train_dl):
            L = data['L'].to(device)
            ab = data['ab'].to(device)

            fake_color = model.net_G(L)
            model.net_D.train()
            model.set_requires_grad(model.net_D, True)
            opt_D.zero_grad()
            fake_image = torch.cat([L, fake_color], dim=1)
            fake_preds = model.net_D(fake_image.detach())
            fake_labels = torch.zeros_like(fake_preds).to(device)
            loss_D_fake = GANcriterion(fake_preds, fake_labels)
            real_image = torch.cat([L, ab], dim=1)
            real_preds = model.net_D(real_image)
            real_labels = torch.ones_like(real_preds).to(device)
            loss_D_real = GANcriterion(real_preds, real_labels)
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward()
            opt_D.step()

            model.net_G.train()
            model.set_requires_grad(model.net_D, False)
            opt_G.zero_grad()
            fake_image = torch.cat([L, fake_color], dim=1)
            fake_preds = model.net_D(fake_image)
            loss_G_GAN = GANcriterion(fake_preds, real_labels)
            loss_G_L1 = L1criterion(fake_color, ab) * lambda_L1
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            opt_G.step()

            losses_dict =  {'loss_D_fake': loss_D_fake,
                            'loss_D_real': loss_D_real,
                            'loss_D': loss_D,
                            'loss_G_GAN': loss_G_GAN,
                            'loss_G_L1': loss_G_L1,
                            'loss_G': loss_G}
            losses_iter_dict['loss_D'].append(loss_D.item())
            losses_iter_dict['loss_G'].append(loss_G.item())
            epoch_loss_D += loss_D.item()
            epoch_loss_G += loss_G.item()
            update_losses(losses_dict, loss_meter_dict, count=data['L'].size(0)) 
            if batch_idx % 20 == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {batch_idx}/{len(train_dl)}")
            if batch_idx % display_every == 0:
                log_results(loss_meter_dict) 
                visualize(model, val_data, save=True) 
                model.net_G.train()

            if e % save_freq == 0 or e == len(range(epochs))-1:
                torch.save(model.state_dict(), "trained_model.pt")

        losses_epoch_dict['loss_D'].append(epoch_loss_D)
        losses_epoch_dict['loss_G'].append(epoch_loss_G)
    plot_losses(losses_epoch_dict, losses_iter_dict)


if __name__ == "__main__":
    args = {
        'use_pretrained_G': False,
        'pretrain' : False,
        'mode': 'lightness',
        'batch_size': 128,
        'lr_G': 2e-4,
        'lr_D': 2e-4,
        'epochs': 20,
        'beta1': 0.5,
        'beta2': 0.999,
        'lambda_L1': 100,
        'save_freq': 5,
        'display_every': 200,
    }
    paths = os.listdir(colored_imgs_folder)
    rand_idxs = np.random.permutation(len(paths))
    val_paths = paths[:int(len(paths)/5)] 
    train_paths = paths[int(len(paths)/5):] 
    train_dataset = ColorizationDataset(paths=train_paths, mode=args['mode'], split='train')
    val_dataset = ColorizationDataset(paths=val_paths, mode=args['mode'], split='test')
    train_dl = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=True)
    if args['use_pretrained_G']:
        net_G = build_res_unet(n_input=1, n_output=2, size=256)
        if args['pretrain']:
            print('Pretraining the generator...')
            pretrain_generator(net_G, train_dl, 20)
            print('Pretraining complete.')
        else:
            net_G.load_state_dict(torch.load("res18-unet.pt", map_location=device))
        model = MainModel(net_G=net_G)
        # 20 epochs is good
        train_model(
            model=model, 
            train_dl=train_dl, 
            val_dl=val_dl,
            lr_G=args['lr_G'],
            lr_D=args['lr_D'],
            epochs=args['epochs'],
            beta1=args['beta1'],
            beta2=args['beta2'],
            lambda_L1=args['lambda_L1'],
            save_freq=args['save_freq'],
            display_every=args['display_every']
            )
    else:   
        model = MainModel()
        # 100 epochs is good
        train_model(
            model=model, 
            train_dl=train_dl, 
            val_dl=val_dl,
            lr_G=args['lr_G'],
            lr_D=args['lr_D'],
            epochs=args['epochs'],
            beta1=args['beta1'],
            beta2=args['beta2'],
            lambda_L1=args['lambda_L1'],
            save_freq=args['save_freq'],
            display_every=args['display_every']
            )