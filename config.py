import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--embedding_size', type=int, default=128)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--n_cluster', type=int, default=10)

parser.add_argument('--pretrain_lr', type=float, default=1e-3)
parser.add_argument('--pretrain_epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--epochs', type=int, default=8)

parser.add_argument('--ratio', type=float, default=0.05, help='ratio of outliers')
parser.add_argument('--level', type=int, default=3)
parser.add_argument('--point_prob', type=float, default=0.1)
parser.add_argument('--obeserved_ratio', type=float, default=1.0)

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dataset', type=str, default='porto')
parser.add_argument('--task', type=str, default='train')
parser.add_argument('--seed', type=int, default=1234)

args = parser.parse_args()
