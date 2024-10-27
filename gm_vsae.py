import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GM_VSAE(nn.Module):
    def __init__(self, token_size, embedding_size, hidden_size, n_cluster):
        super(GM_VSAE, self).__init__()

        self.pi_prior = nn.Parameter(torch.ones(n_cluster) / n_cluster)
        self.mu_prior = nn.Parameter(torch.zeros(n_cluster, hidden_size))
        self.log_var_prior = nn.Parameter(torch.randn(n_cluster, hidden_size))

        self.embedding = nn.Embedding(token_size, embedding_size)
        self.encoder = nn.GRU(embedding_size, hidden_size, 1, batch_first=True)
        self.decoder = nn.GRU(embedding_size, hidden_size, 1, batch_first=True)

        self.fc_mu = nn.Linear(hidden_size, hidden_size)
        self.fc_logvar = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.fc_out = nn.Linear(hidden_size, token_size)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, trajs, lengths, mode, c):
        batch_size = len(trajs)
        e_input = self.embedding(trajs)
        d_input = torch.cat((torch.zeros(batch_size, 1, e_input.size(-1), dtype=torch.long).to(e_input.device),
                             e_input[:, :-1, :]), dim=1)
        decoder_inputs = pack_padded_sequence(d_input, lengths, batch_first=True, enforce_sorted=False)

        if mode == 'pretrain' or 'train':
            encoder_inputs = pack_padded_sequence(e_input, lengths, batch_first=True, enforce_sorted=False)
            _, encoder_final_state = self.encoder(encoder_inputs)

            mu = self.fc_mu(encoder_final_state)
            logvar = self.fc_logvar(encoder_final_state)
            z = self.reparameterize(mu, logvar)
            decoder_outputs, _ = self.decoder(decoder_inputs, z)
            decoder_outputs, _ = pad_packed_sequence(decoder_outputs, batch_first=True)

        elif mode == 'test':
            mu = torch.stack([self.mu_prior] * batch_size, dim=1)[c: c + 1]
            decoder_outputs, _ = self.decoder(decoder_inputs, mu)
            decoder_outputs, _ = pad_packed_sequence(decoder_outputs, batch_first=True)
            logvar, z = None, None

        output = self.fc_out(self.layer_norm(decoder_outputs))

        return output, mu, logvar, z
