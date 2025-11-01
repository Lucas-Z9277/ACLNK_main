# -*- coding: UTF-8 -*-
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import data_loader as Dataloader
import tools.metrics as metrics
import tools.plot as plot
from tools.utils import get_next_batch, loss_bpr_func, loss_ae

sys.path.append(os.getcwd())
matplotlib.use('Agg')
torch.manual_seed(1)
LAMBDA = .1  # Smaller lambda seems to help for toy tasks specifically
CRITIC_ITERS = 5  # How many critic iterations per generator iteration

date = "1209"
l2_loss = nn.MSELoss()

class l2_constraint(nn.Module):

    def __init__(self):
        super(l2_constraint, self).__init__()
        self.l2_loss = nn.MSELoss()

    def forward_2(self, a_embed, b_embed):
        return self.l2_loss(a_embed, b_embed)

def calc_gradient_penalty(netD, real_data, fake_data, BATCH_SIZE, device):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    # alpha = alpha.cuda() if use_cuda else alpha
    alpha = alpha.to(device)
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    # if use_cuda:
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def get_user_embed(model, seq, domain, param, max_len_all, device, pad_idx):
    mask = seq == pad_idx
    mask = (1 - mask.to(int)).view(-1).to(torch.float32).to(device)
    seq = seq.to(device)
    if torch.cuda.device_count() > 1:
        seq_embed = model.module.get_seq_embed(seq, domain=domain,
                                               mask=mask.view(-1, max_len_all))[:, -1, :]
    else:
        seq_embed = model.get_seq_embed(seq, domain=domain,
                                        mask=mask.view(-1, max_len_all))[:, -1, :]
    return seq_embed

def get_pad_mask(seq, pad_index, device):
    mask = seq == pad_index
    mask = (1 - mask.to(int)).view(-1).to(torch.float32)  # 0, 1
    return mask.to(device)

def train_gan_all(netG, netD, gan_loader, opt_d, opt_g, device, param, iterations, max_len_all, domain="a", overlap=False):
    if torch.cuda.device_count() > 1:
        l2_func = nn.DataParallel(l2_constraint())
    else:
        l2_func = l2_constraint()
    opt_final_rec = optim.Adam(netG.parameters(), lr=0.001, betas=(0.9, 0.98))

    k_val = [5, 10, 20]
    result = [{}, {}]
    for val in k_val:
        result[0][str(val)] = {"ht_eval": [], "ndcg_eval": [], "mrr_eval": [],
                               "ht_test": [], "ndcg_test": [], "mrr_test": []}
        result[1][str(val)] = {"ht_eval": [], "ndcg_eval": [], "mrr_eval": [],
                               "ht_test": [], "ndcg_test": [], "mrr_test": []}
    metrics_name = list(result[0]["5"].keys())
    FIXED_GENERATOR = False  # whether to hold the generator fixed at real data plus
    a_iterator = iter(gan_loader[0])
    b_iterator = iter(gan_loader[1])
    # if domain == "a":
    #     rec_iterator = iter(gan_loader[0])
    # else:
    #     rec_iterator = iter(gan_loader[1])
    ###iterations=300
    dis_loss_dict = {}
    real_loss_dict = {}
    fake_loss_dict = {}
    for iteration in tqdm(range(int(iterations))):
        # gan training of the NetD and GUR encoder *** phase 2 ***
        # if iteration < int(iterations * 0.6):  # or iteration % 2 == 0:
            ############################
            # (1) Update D network
            ###########################
            for p in netD.parameters():   # reset requires_grad
                p.requires_grad = True    # they are set to False below in netG update

            for iter_d in range(CRITIC_ITERS):
                # a domain data
                try:
                    in_seq_a, _, _ = get_next_batch(a_iterator, device)
                    # in_seq_a = next(a_iterator)
                except StopIteration:
                    if iteration > 1000:
                        a_iterator = iter(gan_loader[0])  # freq
                    else:
                        a_iterator = iter(gan_loader[0])  # random
                    in_seq_a, _, _ = get_next_batch(a_iterator, device)
                mask_a = in_seq_a == param.pad_index
                mask_a = (1 - mask_a.to(int)).view(-1).to(torch.float32).to(device)
                in_seq_a = in_seq_a.to(device)
                if torch.cuda.device_count() > 1:
                    in_seq_ae = netG.module.get_seq_embed(in_seq_a, domain="a",
                                                          mask=mask_a.view(-1, max_len_all))[:, -1, :]
                else:
                    in_seq_ae = netG.get_seq_embed(in_seq_a, domain="a",
                                                   mask=mask_a.view(-1, max_len_all))[:, -1, :]

                in_seq_ae = in_seq_ae.detach()  # detach when training,
                # b domain data
                try:
                    in_seq_b, _, _ = get_next_batch(b_iterator, device)
                except StopIteration:
                    if iteration > 1000:
                        b_iterator = iter(gan_loader[1])
                    else:
                        b_iterator = iter(gan_loader[1])
                    in_seq_b, _, _ = get_next_batch(b_iterator, device)
                mask_b = in_seq_b == param.pad_index
                mask_b = (1 - mask_b.to(int)).view(-1).to(torch.float32).to(device)
                in_seq_b = in_seq_b.to(device)
                if torch.cuda.device_count() > 1:
                    in_seq_be = netG.module.get_seq_embed(in_seq_b, domain="b",
                                                          mask=mask_b.view(-1, max_len_all))[:, -1, :]
                else:
                    in_seq_be = netG.get_seq_embed(in_seq_b, domain="b",
                                                   mask=mask_b.view(-1, max_len_all))[:, -1, :]
                in_seq_be = in_seq_be.detach()

                # go through discriminator.
                opt_d.zero_grad()
                D_real = netD(in_seq_ae)  # output of dis is a logits. (scalar)
                D_fake = netD(in_seq_be)
                # loss 1
                real_loss = D_real.mean()
                fake_loss = D_fake.mean()
                dis_loss = fake_loss - real_loss
                # loss 2
                # fake_loss, real_loss = loss_ce(D_fake, zero_label), loss_ce(D_real, one_label)
                # dis_loss = fake_loss + real_loss

                dis_loss.backward()
                # train with gradient penalty  in_seq_ae.data
                gradient_penalty = calc_gradient_penalty(netD, in_seq_ae, in_seq_be, max_len_all, device)
                gradient_penalty.backward()

                D_cost = fake_loss - real_loss + gradient_penalty  # loss_1
                # D_cost = fake_loss + real_loss + gradient_penalty
                Wasserstein_D = D_real.mean() - D_fake.mean()
                opt_d.step()

            if not FIXED_GENERATOR:
                ############################
                # (2) Update G network
                ###########################
                for p in netD.parameters():
                    p.requires_grad = False  # to avoid update discriminator
                opt_g.zero_grad()
                # ===================================================== discriminator loss
                # a-domain data
                try:
                    in_seq_a, _, _ = get_next_batch(a_iterator, device)
                except StopIteration:
                    if iteration > 1000:
                        a_iterator = iter(gan_loader[0])  # freq a-domain data
                    else:
                        a_iterator = iter(gan_loader[0])  # rand a-domain data
                    in_seq_a, _, _ = get_next_batch(a_iterator, device)
                    #  bs, sl are the same in different domain.
                in_seq_ae = get_user_embed(netG, in_seq_a, "a", param, max_len_all, device, 0)

                try:
                    in_seq_b, _, _ = get_next_batch(b_iterator, device)
                except StopIteration:
                    if iteration > 1000:
                        b_iterator = iter(gan_loader[1])  # freq b-domain
                    else:
                        b_iterator = iter(gan_loader[1])  # random b-domain
                    in_seq_b, _, _ = get_next_batch(b_iterator, device)
                in_seq_be = get_user_embed(netG, in_seq_b, "b", param, max_len_all, device, 0)

                D_real = netD(in_seq_ae)  # output of dis is a logits. (scalar)
                D_fake = netD(in_seq_be)
                # loss 1
                real_loss = D_real.mean()
                fake_loss = D_fake.mean()
                real_loss_dict[iteration] = real_loss.item()
                fake_loss_dict[iteration] = fake_loss.item()

                g_dis_loss = real_loss - fake_loss
                dis_loss_dict[iteration] = g_dis_loss.item()
                # with open(param.result_path + '/dis_loss_dict_3000.txt', 'w') as f2:
                #     f2.write(str(dis_loss_dict))

                # loss 2
                # fake_loss, real_loss = loss_ce(D_fake, one_label), loss_ce(D_real, zero_label)
                # g_dis_loss = fake_loss + real_loss
                g_dis_loss.backward()



                # mask_rec_a = get_pad_mask(dec_out_a, param.pad_index, device)
                # loss_recon_a = loss_ae(netG, in_seq_a, dec_in_a, dec_out_a, n_items_a, True, bs, sl,
                #                        param, mask_rec_a, device, domain="a")
                #
                # mask_rec_b = get_pad_mask(dec_out_b, param.pad_index, device)
                # loss_recon_b = loss_ae(netG, in_seq_b, dec_in_b, dec_out_b, n_items_b, True, bs, sl,
                #                        param, mask_rec_b, device, domain="b")
                #
                # loss_recon_a.backward()
                # loss_recon_b.backward()

                # ===================================================== Recommendation loss on target domain
                # None.
                # back-propagation
                # opt_g.step()

    min_key = min(dis_loss_dict, key=lambda k: abs(dis_loss_dict[k]))
    print(min_key, dis_loss_dict[min_key])
    return in_seq_ae


    # return in_seq_ae
