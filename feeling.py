import danling as dl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from chanfig import Config, ConfigParser
from danling.metrics import AverageMeter, accuracy
from danling.models.transformer import (TransformerEncoder,
                                        TransformerEncoderLayer,
                                        UnitedPositionEmbedding)
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset


class PollDataset(Dataset):
    def __init__(self, path: str, target: str = 'A3'):
        # self.indices = indices if indices is not None else list(range(len(self.data)))
        # self.data = self.data.iloc[indices]
        self.data = pd.read_csv(path)
        self.data = self.data.set_index('SRCID')
        self.target = target
        self.agg = torch.tensor(self.data['A4F2_agg'].values)
        self.excludes = ['undecided_voter', 'A4F2_agg']
        self.data = self.data.loc[:, ~self.data.columns.isin(self.excludes)].astype(np.int64)
        cols = [i for i in self.data.keys() if i not in (self.target)]
        self.inputs = self.data[cols]
        cols = [self.target]
        self.targets = torch.tensor(self.data[cols].values).squeeze()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        return torch.tensor(self.inputs.iloc[index].values), self.targets[index]


class Model(nn.Module):
    def __init__(self, embed_dim: int, dropout: float, transformer: dict):
        super().__init__()
        l = [2, 6, 26, 6, 6, 6, 6, 5, 5, 3, 3, 3, 3, 6, 7, 7, 4, 6, 6, 6, 5, 5, 7, 7, 5, 4, 4, 4, 4, 4, 4, 4, 4, 6, 6, 5, 5, 6, 6, 5, 5, 5, 5, 6, 7, 7, 3, 5, 4, 9, 9, 2]
        self.embeddings = nn. ModuleList([nn.Embedding(i, embed_dim) for i in l])
        self.dropout = nn.Dropout(dropout)
        num_layers = transformer.pop('num_layers')
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = UnitedPositionEmbedding(embed_dim, transformer['num_heads'], len(l))
        self.layer = TransformerEncoderLayer(embed_dim=embed_dim, **transformer)
        self.encoder = TransformerEncoder(self.layer, num_layers)
        self.classify = nn.Linear(embed_dim, 12)
        self.softmax = nn.Softmax()

    def forward(self, input):
        out = torch.stack([embed(i) for embed, i in zip(self.embeddings, input.T)], dim=1)
        out = self.dropout(out)
        cls_tokens = self.cls_token.expand(input.shape[0], -1, -1)
        out = torch.cat((cls_tokens, out), dim=1)
        pos_embed = self.pos_embed(out)
        out, attn = self.encoder(out, attn_bias=pos_embed)
        out = self.classify(out[:, 0, :])
        out = self.softmax(out)
        return out


class Runner(dl.BaseRunner):
    "Runner"
    def __init__(self, train_indices, val_indices, **kwargs):
        super().__init__(**kwargs)
        self.init_lr()
        self.model = Model(**self.net.dict())
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.optimize.weight_decay)
        self.dataset = PollDataset(self.data.path)
        self.criterion = nn.CrossEntropyLoss()
        self.model, self.optimizer = self.prepare(self.model, self.optimizer)
        self.dataloaders['train'] = self.get_dataloader(train_indices)
        self.dataloaders['val'] = self.get_dataloader(val_indices)
        self.iter_end = 3061*self.epoch_end//self.batch_size_actual
        self.scheduler = dl.Scheduler(self.optimizer, self.iter_end, policy='cosine')

    def run(self):
        for self.epochs in range(0, self.epoch_end):
            self.train()
            if self.is_best:
                self.save()
                self.result_best = self.result_last
            self.score_last = self.result_last = self.evaluate()
            self.results.append(self.result_last)

    def train(self):
        self.accuracies = AverageMeter(self.batch_size)
        self.losses = AverageMeter(self.batch_size)
        for iteration, (data, gt) in enumerate(self.dataloaders['train']):
            out = self.model(data)
            loss = self.criterion(out, gt)
            loss.backward()
            self.step()
            self.losses.update(loss.item())
            out, gt = self.gather((out, gt))
            acc1 = accuracy(out, gt)[0]
            self.accuracies.update(acc1.item())
        print(f"loss: {self.losses.avg}\tacc: {self.accuracies.avg}")
        return self.accuracies.avg

    @torch.no_grad()
    def evaluate(self):
        self.losses = AverageMeter(self.batch_size)
        outs, gts = [], []
        for iteration, (data, gt) in enumerate(self.dataloaders['train']):
            out = self.model(data)
            loss = self.criterion(out, gt)
            outs.extend(out)
            gts.extend(gt)
            self.losses.update(loss.item())
        outs, gts = torch.stack(outs), torch.stack(gts)
        outs, gts = self.gather((outs, gts))
        acc1 = accuracy(outs, gts)[0].item()
        print(f"evaluate: {acc1}")
        return self.accuracies.avg

    def get_dataloader(self, indices):
        subset = Subset(self.dataset, indices)
        dataloader = DataLoader(subset, batch_size=self.batch_size)
        return self.prepare(dataloader)
        #return {key: apriori(self.targets[[key]].join(self.inputs), min_support=min_support, use_colnames=True) for key in self.targets.keys()}


if __name__ == '__main__':
    defaults = {
        'seed': 1031,
        'name': 'feeling',
        'batch_size': 64,
        'batch_size_base': 64,
        'epoch_end': 100,
        'gradient_clip': 1.0,
        'net.dropout': 0.1,
        'net.embed_dim': 1024,
        'net.transformer.num_layers': 16,
        'net.transformer.num_heads': 16,
        'net.transformer.ffn_dim': 4096,
        'lr': 3e-4,
        'lr_final': 1e-8,
        'optimize.weight_decay': 0.01,
        'data.path': 'standard.csv',
    }
    kf = KFold(10, shuffle=True, random_state=1031)
    train_indices, val_indices = next(kf.split(range(3061)))
    runner = Runner(train_indices=train_indices, val_indices=val_indices, **defaults)
    runner = runner.parse()
    runner.run()
