import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

from config import args
from gm_vsae import GM_VSAE


def make_mask(seqs, lengths):
    mask = torch.zeros_like(seqs)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    return mask


def auc_score(y_true, y_score):
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)

    if y_true.sum() == 0:
        return 0.0

    order = np.argsort(-y_score)
    y_true = y_true[order]

    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / tp[-1]

    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.trapz(precision, recall))


def collate_fn(batch):
    max_len = max(len(x) for x in batch)
    seq_lengths = list(map(len, batch))
    batch_trajs = [x + [0] * (max_len - len(x)) for x in batch]
    return torch.LongTensor(batch_trajs), np.array(seq_lengths)


class MyDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        data_seqs = self.seqs[index]
        return data_seqs


class train_gmvsae:
    def __init__(self, token_size, train_loader, outliers_loader, labels, args):
        self.model = GM_VSAE(token_size, args.embedding_size, args.hidden_size, args.n_cluster).to(args.device)

        self.crit = nn.CrossEntropyLoss()
        self.detect = nn.CrossEntropyLoss(reduction='none')

        self.gamma_params = [self.model.pi_prior, self.model.mu_prior, self.model.log_var_prior]
        gamma_ids = {id(param) for param in self.gamma_params}
        self.phi_theta_params = [param for param in self.model.parameters() if id(param) not in gamma_ids]

        self.pretrain_optimizer = optim.AdamW(self.phi_theta_params, lr=args.pretrain_lr)
        self.phi_theta_optimizer = optim.Adam(self.phi_theta_params, lr=args.lr)
        self.gamma_optimizer = optim.Adam(self.gamma_params, lr=args.lr)

        self.steplr_pretrain = StepLR(self.pretrain_optimizer, step_size=3, gamma=0.9)
        self.steplr_phi_theta = StepLR(self.phi_theta_optimizer, step_size=3, gamma=0.8)
        self.steplr_gamma = StepLR(self.gamma_optimizer, step_size=3, gamma=0.8)

        self.train_loader = train_loader
        self.outliers_loader = outliers_loader
        self.labels = labels

        os.makedirs('checkpoints', exist_ok=True)
        self.pretrain_path = f'checkpoints/pretrain_{args.n_cluster}_{args.dataset}.pth'
        self.model_path = f'checkpoints/model_{args.n_cluster}_{args.dataset}.pth'

        self.device = args.device
        self.n_cluster = args.n_cluster
        self.dataset = args.dataset
        self.hidden_size = args.hidden_size

    def set_parameter_state(self, phi_theta_grad, gamma_grad):
        for param in self.phi_theta_params:
            param.requires_grad_(phi_theta_grad)
        for param in self.gamma_params:
            param.requires_grad_(gamma_grad)

    def pretrain(self, epoch):
        self.model.train()
        self.set_parameter_state(True, False)

        epo_loss = 0
        for batch in self.train_loader:
            trajs, lengths = batch
            trajs = trajs.to(self.device)

            outputs, _, _, _ = self.model(trajs, lengths, 'pretrain', -1, sample_latent=False)
            mask = make_mask(trajs, lengths).bool()
            loss = self.crit(outputs[mask], trajs[mask])

            self.pretrain_optimizer.zero_grad()
            loss.backward()
            self.pretrain_optimizer.step()
            epo_loss += loss.item()

        self.set_parameter_state(True, True)
        self.steplr_pretrain.step()
        print(f'Warmup Epoch : {epoch + 1}, Recon Loss: {epo_loss / len(self.train_loader):.6f}')
        torch.save(self.model.state_dict(), self.pretrain_path)

    def pretrain_detection(self):
        self.model.eval()
        self.model.load_state_dict(torch.load(self.pretrain_path, weights_only=True))

        all_likelihood = []
        with torch.no_grad():
            for batch in self.outliers_loader:
                trajs, lengths = batch
                trajs = trajs.to(self.device)

                outputs, _, _, _ = self.model(trajs, lengths, 'pretrain', -1, sample_latent=False)
                mask = make_mask(trajs, lengths)
                log_likelihood = - self.detect(outputs.transpose(1, 2), trajs)
                likelihood = torch.exp((mask * log_likelihood).sum(dim=-1) / mask.sum(-1))
                all_likelihood.append(likelihood)
        all_likelihood = torch.cat(all_likelihood, dim=0)

        pr_auc = auc_score(self.labels, (1 - all_likelihood).cpu().detach().numpy())
        print(f'Warmup PR_AUC: {pr_auc:.6f}')

    def train(self, epoch):
        self.model.train()

        total_loss = 0.0
        total_recon = 0.0
        total_kl_c = 0.0
        total_kl_r = 0.0

        for batch in self.train_loader:
            trajs, lengths = batch
            trajs = trajs.to(self.device)

            self.set_parameter_state(True, False)
            self.phi_theta_optimizer.zero_grad()
            outputs, mu, log_var, z = self.model(trajs, lengths, 'train', -1, sample_latent=True)
            loss_phi_theta, stats_phi_theta = self.Loss(
                outputs, trajs, mu.squeeze(0), log_var.squeeze(0), z.squeeze(0), lengths
            )
            loss_phi_theta.backward()
            self.phi_theta_optimizer.step()

            self.set_parameter_state(False, True)
            self.gamma_optimizer.zero_grad()
            outputs, mu, log_var, z = self.model(trajs, lengths, 'train', -1, sample_latent=True)
            loss_gamma, stats_gamma = self.Loss(
                outputs, trajs, mu.squeeze(0), log_var.squeeze(0), z.squeeze(0), lengths
            )
            loss_gamma.backward()
            self.gamma_optimizer.step()

            batch_loss = 0.5 * (loss_phi_theta.item() + loss_gamma.item())
            total_loss += batch_loss
            total_recon += 0.5 * (stats_phi_theta['reconstruction'] + stats_gamma['reconstruction'])
            total_kl_c += 0.5 * (stats_phi_theta['category_kl'] + stats_gamma['category_kl'])
            total_kl_r += 0.5 * (stats_phi_theta['gaussian_kl'] + stats_gamma['gaussian_kl'])

        self.set_parameter_state(True, True)
        self.steplr_phi_theta.step()
        self.steplr_gamma.step()

        num_batches = len(self.train_loader)
        print(
            f'Epoch : {epoch + 1}, '
            f'Loss: {total_loss / num_batches:.6f}, '
            f'Recon: {total_recon / num_batches:.6f}, '
            f'KL(c): {total_kl_c / num_batches:.6f}, '
            f'KL(r): {total_kl_r / num_batches:.6f}'
        )
        torch.save(self.model.state_dict(), self.model_path)

    def detection(self):
        self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
        self.model.eval()

        all_likelihood = []
        with torch.no_grad():
            for c in range(self.n_cluster):
                print(c)
                c_likelihood = []
                for batch in self.outliers_loader:
                    trajs, seq_lengths = batch
                    trajs = trajs.to(self.device)
                    mask = make_mask(trajs, seq_lengths)
                    outputs, _, _, _ = self.model(trajs, seq_lengths, 'test', c)
                    log_likelihood = - self.detect(outputs.transpose(1, 2), trajs)
                    likelihood = torch.exp((mask * log_likelihood).sum(dim=-1) / mask.sum(-1))
                    c_likelihood.append(likelihood)
                all_likelihood.append(torch.cat(c_likelihood).unsqueeze(0))

        all_likelihood = torch.cat(all_likelihood, dim=0)
        likelihood, _ = torch.max(all_likelihood, dim=0)
        pr_auc = auc_score(self.labels, (1 - likelihood).cpu().detach().numpy())
        print(f'PR_AUC: {pr_auc:.6f}')

    def gaussian_pdf_log(self, x, mu, log_var):
        return -0.5 * torch.sum(
            math.log(2 * math.pi) + log_var + (x - mu).pow(2) / (torch.exp(log_var) + 1e-10),
            dim=1
        )

    def gaussian_pdfs_log(self, x, mus, log_vars):
        gaussian_logits = []
        for c in range(self.n_cluster):
            gaussian_logits.append(self.gaussian_pdf_log(x, mus[c: c + 1, :], log_vars[c: c + 1, :]).view(-1, 1))
        return torch.cat(gaussian_logits, dim=1)

    def route_type_posterior(self, z):
        log_pi = F.log_softmax(self.model.pi_prior, dim=-1)
        log_q_c = log_pi.unsqueeze(0) + self.gaussian_pdfs_log(z, self.model.mu_prior, self.model.log_var_prior)
        q_c = F.softmax(log_q_c, dim=-1)
        q_c = q_c.clamp_min(1e-10)
        q_c = q_c / q_c.sum(dim=-1, keepdim=True)
        return q_c, log_pi

    def Loss(self, x_hat, targets, z_mu, z_log_var, z, lengths):
        mask = make_mask(targets, lengths)
        reconstruction_loss = (self.crit(x_hat.transpose(1, 2), targets) * mask).sum() / mask.sum()

        q_c, log_pi = self.route_type_posterior(z)
        category_loss = torch.sum(q_c * (torch.log(q_c) - log_pi.unsqueeze(0)), dim=-1).mean()

        mu_c = self.model.mu_prior.unsqueeze(0)
        log_var_c = self.model.log_var_prior.unsqueeze(0)
        posterior_mu = z_mu.unsqueeze(1)
        posterior_log_var = z_log_var.unsqueeze(1)
        posterior_var = torch.exp(posterior_log_var)
        prior_var = torch.exp(log_var_c)

        kl_r_c = 0.5 * torch.sum(
            log_var_c - posterior_log_var
            + (posterior_var + (posterior_mu - mu_c).pow(2)) / (prior_var + 1e-10)
            - 1,
            dim=-1
        )
        gaussian_loss = torch.sum(q_c * kl_r_c, dim=-1).mean()

        loss = reconstruction_loss + category_loss + gaussian_loss
        stats = {
            'reconstruction': reconstruction_loss.item(),
            'category_kl': category_loss.item(),
            'gaussian_kl': gaussian_loss.item(),
        }
        return loss, stats


if __name__ == '__main__':
    if args.dataset == 'porto':
        token_size = 51 * 158

    train_trajs = np.load(f'./data/{args.dataset}/train_data.npy', allow_pickle=True)
    test_trajs = np.load(f'./data/{args.dataset}/outliers_data.npy', allow_pickle=True)
    outliers_idx = np.load(f'./data/{args.dataset}/outliers_idx.npy', allow_pickle=True)

    train_data = MyDataset(train_trajs)
    test_data = MyDataset(test_trajs)

    labels = np.zeros(len(test_trajs))
    for i in outliers_idx:
        labels[i] = 1

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    outliers_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    Train_gmvsae = train_gmvsae(token_size, train_loader, outliers_loader, labels, args)

    if args.pretrain_epochs > 0:
        print('---------------Warmup---------------')
        for epoch in range(args.pretrain_epochs):
            Train_gmvsae.pretrain(epoch)

        print('---------------Warmup Validation---------------')
        Train_gmvsae.model.load_state_dict(torch.load(Train_gmvsae.pretrain_path, weights_only=True))
        Train_gmvsae.pretrain_detection()

    print('---------------Joint Training---------------')
    for epoch in range(args.epochs):
        Train_gmvsae.train(epoch)
        Train_gmvsae.detection()

    print('---------------Testing---------------')
    Train_gmvsae.model.load_state_dict(torch.load(Train_gmvsae.model_path, weights_only=True))
    Train_gmvsae.detection()
