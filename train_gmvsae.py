import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_recall_curve, auc
from sklearn.mixture import GaussianMixture
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
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)


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

        self.crit = nn.CrossEntropyLoss(reduction='none')
        self.detect = nn.CrossEntropyLoss(reduction='none')

        self.pretrain_optimizer = optim.AdamW(self.model.parameters(), lr=args.pretrain_lr)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        self.steplr_pretrain = StepLR(self.pretrain_optimizer, step_size=3, gamma=0.9)
        self.steplr = StepLR(self.optimizer, step_size=3, gamma=0.8)

        self.train_loader = train_loader
        self.outliers_loader = outliers_loader
        self.labels = labels

        self.pretrain_path = f'checkpoints/pretrain_{args.n_cluster}_{args.dataset}.pth'
        self.model_path = f'checkpoints/model_{args.n_cluster}_{args.dataset}.pth'

        self.device = args.device
        self.n_cluster = args.n_cluster
        self.dataset = args.dataset
        self.hidden_size = args.hidden_size

    def pretrain(self, epoch):
        self.model.train()

        epo_loss = 0
        for batch in self.train_loader:
            trajs, lengths = batch
            trajs = trajs.to(self.device)

            outputs, _, _, _, = self.model(trajs, lengths, 'pretrain', -1)
            mask = make_mask(trajs, lengths)
            loss = (self.crit(outputs.transpose(1, 2), trajs) * mask).sum() / mask.sum()

            self.pretrain_optimizer.zero_grad()
            loss.backward()
            self.pretrain_optimizer.step()
            epo_loss += loss.item()

        self.steplr_pretrain.step()
        print(f'Epoch : {epoch + 1}, Loss: {epo_loss / len(self.train_loader)}')
        torch.save(self.model.state_dict(), self.pretrain_path)

    def pretrain_detection(self):
        self.model.eval()
        self.model.load_state_dict(torch.load(self.pretrain_path, weights_only=True))

        all_likelihood = []
        with torch.no_grad():
            for batch in self.outliers_loader:
                trajs, lengths = batch
                trajs = trajs.to(self.device)

                outputs, _, _, _ = self.model(trajs, lengths, 'pretrain', -1)
                mask = make_mask(trajs, lengths)
                log_likelihood = - self.detect(outputs.transpose(1, 2), trajs)
                likelihood = torch.exp((mask * log_likelihood).sum(dim=-1) / mask.sum(-1))
                all_likelihood.append(likelihood)
        all_likelihood = torch.cat(all_likelihood, dim=0)

        pr_auc = auc_score(self.labels, (1 - all_likelihood).cpu().detach().numpy())
        print(f'PR_AUC: {pr_auc}')

    def train_gmm(self):
        self.model.load_state_dict(torch.load(self.pretrain_path, weights_only=True))
        self.model.eval()

        z = []
        with torch.no_grad():
            for batch in self.train_loader:
                trajs, lengths = batch
                trajs = trajs.to(self.device)
                _, _, _, hidden = self.model(trajs, lengths, 'pretrain', -1)
                z.append(hidden.squeeze(0))
            z = torch.cat(z, dim=0)

        print('...Fiting Gaussian Mixture Model...')
        self.gmm = GaussianMixture(n_components=self.n_cluster, covariance_type='diag', n_init=3, reg_covar=1e-5)
        self.gmm.fit(z.cpu().numpy())

    def save_weights(self):
        print('...Saving Weights...')
        torch.save(torch.from_numpy(self.gmm.weights_).float(),
                   f'checkpoints/pi_prior_{self.n_cluster}_{self.dataset}.pth')
        torch.save(torch.from_numpy(self.gmm.means_).float(),
                   f'checkpoints/mu_prior_{self.n_cluster}_{self.dataset}.pth')
        torch.save(torch.log(torch.from_numpy(self.gmm.covariances_)).float(),
                   f'checkpoints/log_var_prior_{self.n_cluster}_{self.dataset}.pth')

    def train(self, epoch):
        self.model.train()

        total_loss = 0
        for batch in self.train_loader:
            trajs, lengths = batch
            trajs = trajs.to(self.device)

            output, mu, log_var, z = self.model(trajs, lengths, 'train', -1)
            loss = self.Loss(output, trajs, mu.squeeze(0), log_var.squeeze(0), z.squeeze(0), lengths)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        self.steplr.step()
        print(f'Epoch : {epoch + 1}, Loss: {total_loss / len(self.train_loader)}')
        torch.save(self.model.state_dict(), self.model_path)

    def detection(self):
        self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
        self.model.eval()

        all_likelihood = []
        with torch.no_grad():
            for c in range(self.n_cluster):
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
        print(f'PR_AUC: {pr_auc}')

    def gaussian_pdf_log(self, x, mu, log_var):
        return -0.5 * (torch.sum(np.log(np.pi * 2) + log_var + (x - mu).pow(2) / (torch.exp(log_var)), 1))

    def gaussian_pdfs_log(self, x, mus, log_vars):
        G = []
        for c in range(self.n_cluster):
            G.append(self.gaussian_pdf_log(x, mus[c: c + 1, :], log_vars[c: c + 1, :]).view(-1, 1))
        return torch.cat(G, 1)

    def Loss(self, x_hat, targets, z_mu, z_sigma2_log, z, lengths):
        pi = self.model.pi_prior
        log_sigma2_c = self.model.log_var_prior
        mu_c = self.model.mu_prior

        mask = make_mask(targets, lengths)
        reconstruction_loss = (self.crit(x_hat.transpose(1, 2), targets) * mask).sum() / mask.sum()

        logits = -(torch.square(z.unsqueeze(1) - mu_c.unsqueeze(0)) / (2 * torch.exp(log_sigma2_c.unsqueeze(0)))).sum(-1)
        logits = F.softmax(logits, dim=-1) + 1e-10
        category_loss = torch.mean(torch.sum(logits * (torch.log(logits) - torch.log(pi).unsqueeze(0)), dim=-1))

        gaussian_loss = (self.gaussian_pdf_log(z, z_mu, z_sigma2_log).unsqueeze(1)
                         - self.gaussian_pdfs_log(z, mu_c, log_sigma2_c)).mean()

        loss = reconstruction_loss + category_loss + gaussian_loss / self.hidden_size
        return loss


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

    print('---------------Pretrain---------------')
    for epoch in range(args.pretrain_epochs):
        Train_gmvsae.pretrain(epoch)

    print('---------------Validating---------------')
    Train_gmvsae.pretrain_detection()

    Train_gmvsae.train_gmm()
    Train_gmvsae.save_weights()

    print('---------------Training---------------')
    Train_gmvsae.model.load_state_dict(torch.load(Train_gmvsae.pretrain_path, weights_only=True))
    Train_gmvsae.model.pi_prior.data = torch.load(
        f'checkpoints/pi_prior_{args.n_cluster}_{args.dataset}.pth', weights_only=True).to(args.device)
    Train_gmvsae.model.mu_prior.data = torch.load(
        f'checkpoints/mu_prior_{args.n_cluster}_{args.dataset}.pth', weights_only=True).to(args.device)
    Train_gmvsae.model.log_var_prior.data = torch.load(
        f'checkpoints/log_var_prior_{args.n_cluster}_{args.dataset}.pth', weights_only=True).to(args.device)

    for epoch in range(args.epochs):
        Train_gmvsae.train(epoch)

    print('---------------Testing---------------')
    Train_gmvsae.detection()
