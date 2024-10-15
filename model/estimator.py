import torch


class Estimator(torch.nn.Module):
    def __init__(self, xin=None, yin=None, y_cat_dim=10, num_layers=5, hidden_channels=100, norm=True, dropout=0.,
                 act=torch.nn.ReLU, cnn=None):

        '''
            xin: 输入数据的特征数目
            yin: 输入的标签 y 的维度
            y_cat_dim: y 的分类维度, 用于全连接层的维度设置
        '''


        super().__init__()

        self.cnn = cnn
        self.yin = yin

        if cnn is None:  # 如果没有传入 cnn, 则构建一个全连接网络 NN
            self.f1 = NN(in_channels=xin, out_channels=y_cat_dim, num_layers=int(num_layers - 2),
                         hidden_channels=hidden_channels, dropout=dropout, norm=norm, bias=False,
                         act=act, out_fn=act())

        seq = [torch.nn.Linear(y_cat_dim + yin, int(y_cat_dim), bias=False), act(),
               torch.nn.Linear(int(y_cat_dim), 1, bias=False)]

        self.f_out = torch.nn.Sequential(*seq)

    def forward(self, x, y, y_val=None):  # y_val: 额外的验证数据, 默认为 None

        if y_val is not None:
            y = torch.cat((y, y_val), dim=-1)

        if self.cnn is not None:
            z = self.cnn(x)
        else:
            x = x.view(x.size(0), -1)  # 将 x 重塑为二维 (即 batch_size * feature_size)
            z = self.f1(x)

        if self.yin > 0:  # 说明有额外的标签数据 y
            z = torch.cat((z, y), dim=1)

        out = self.f_out(z)
        return torch.sigmoid(out)


class NN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, hidden_channels, norm=True, dropout=0., bias=True,
                 act=torch.nn.ReLU, out_fn=None):
        ''''''

        super().__init__()

        seq = []
        norm_type = torch.nn.InstanceNorm1d

        # first layer
        if in_channels is not None:
            # seq.append(torch.nn.Linear(in_channels, hidden_channels, bias=bias))
            seq.append(torch.nn.Linear(3072, hidden_channels, bias=bias))
        else:
            seq.append(torch.nn.LazyLinear(hidden_channels, bias=bias))

        seq.append(torch.nn.Dropout(dropout))
        seq.append(act())

        for l in range(num_layers - 1):
            if norm: seq.append(norm_type(hidden_channels))
            seq.append(torch.nn.Linear(hidden_channels, hidden_channels, bias=bias))
            seq.append(torch.nn.Dropout(dropout))
            seq.append(act())

        if norm: seq.append(norm_type(hidden_channels))
        seq.append(torch.nn.Linear(hidden_channels, out_channels, bias=bias))

        if out_fn is not None:
            # softmax, sigmoid, etc
            seq.append(out_fn)

        self.f = torch.nn.Sequential(*seq)

    def forward(self, x, y=None, idx=None):
        return self.f(x)

    def reset_parameters(self):
        self.apply(weight_reset)


def weight_reset(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.ConvTranspose2d):
        m.reset_parameters()