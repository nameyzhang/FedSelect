import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from torch.optim.sgd import SGD


class MetaSGD(SGD):    # 继承自 SGD 优化器
    def __init__(self, net, *args, **kwargs):             # 类的初始化函数
        super(MetaSGD, self).__init__(*args, **kwargs)    # 调用基类 SGD 的初始化方法
        self.net = net                                    # 将传入的网络模型保存为类的属性

    def set_parameter(self, current_module, name, parameters):    # 设置模型中的参数
        if '.' in name:                                           # 检查参数名称中是否包含 ., 以判断是否为嵌套的模块路径;
            name_split = name.split('.')                          # 如果为嵌套路径, 将 name 按 . 分割, 生成列表 name_split;
            module_name = name_split[0]                           # 获取路径中第一个模块名称
            rest_name = '.'.join(name_split[1:])                  # 获取路径中的其余部分, 以 . 连接;
            for children_name, children in current_module.named_children():    # 遍历 current_module 的所有子模块; 使用 named_children 方法获取子模块名称和子模块对象;
                if module_name == children_name:                  # 如果找到匹配的子模块名称, 递归调用 set_parameter 方法, 将子模块 children 作为新的 current_module, 并传递剩余的名称 rest_name 和参数值 parameters
                    self.set_parameter(children, rest_name, parameters)
                    break
        else:
            current_module._parameters[name] = parameters         # 如果参数名称 name 不包含., 表示这是当前模块的直接参数

    def meta_step(self, grads):                 # 用于在元训练步骤中根据计算的梯度更新模型参数; grads 是梯度列表, 包含每个参数对应的梯度
        group = self.param_groups[0]            # 获取优化器的第一个参数组
        weight_decay = group['weight_decay']    # 权重衰减
        momentum = group['momentum']            # 动量
        dampening = group['dampening']          # 阻尼
        nesterov = group['nesterov']            # Nesterov 动量
        lr = group['lr']                        # 学习率

        for (name, parameter), grad in zip(self.net.named_parameters(), grads):   # 遍历参数和对应的梯度; self.net.named_parameters() 返回模型参数的名称和参数值; zip(self.net.named_parameters(), grads) 将参数名称、参数值与对应的梯度打包在一起进行迭代
            parameter.detach_()                                                   # 将参数从计算图中分离出来, 以便对其进行原地操作; 这一步可以防止梯度传播到更新步骤之前的计算中;
            if weight_decay != 0:                                                 # 如果 weight_decay 不为零, 将参数值乘以权重衰减系数并加到梯度上; 为零就直接使用原始梯度;
                grad_wd = grad.add(parameter, alpha=weight_decay)
            else:
                grad_wd = grad

            if momentum != 0 and 'momentum_buffer' in self.state[parameter]:      # 如果 momentum 不为零, 并且参数存在动量缓存区momentum_buffer, 则计算动量梯度 grad_b
                buffer = self.state[parameter]['momentum_buffer']
                grad_b = buffer.mul(momentum).add(grad_wd, alpha=1-dampening)     # 计算带动量的梯度
            else:
                grad_b = grad_wd                                                  # 如果 momentum 为零或参数没有动量缓存区, 直接使用权重衰减后的梯度 grad_wd

            if nesterov:                                                          # 如果使用 Nesterov 动量, 则将动量梯度乘以动量系数后加到衰减后的梯度上
                grad_n = grad_wd.add(grad_b, alpha=momentum)
            else:
                grad_n = grad_b                                                   # 如果不使用 Nesterov 动量, 直接使用动量梯度 grad_b

            self.set_parameter(self.net, name, parameter.add(grad_n, alpha=-lr))

# Build you torch or tf model class here
def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MLP(MetaModule):
    def __init__(self, input, hidden1, output):
        super(MLP, self).__init__()
        self.linear1 = MetaLinear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden1, 2*hidden1)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = MetaLinear(2*hidden1, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        out = self.linear3(x)
        return F.sigmoid(out)




class VNet(nn.Module):
    def __init__(self, input, hidden1, output):         # 接收三个参数
        super(VNet, self).__init__()                    # 调用了父类 nn.Module 的构造函数, 确保父类的初始化过程被执行, 这是继承 nn.Module 类的自定义网络模块所必需的
        self.linear1 = nn.Linear(input, hidden1)        # 线性变换模块 (即全连接层)
        self.relu1 = nn.ReLU(inplace=True)              # 第一个全连接层之后应用, inplace=True 表示直接对输入进行修改, 而不是创建新的输出
        self.linear2 = nn.Linear(hidden1, output)       # 定义第二个全连接层

    def forward(self, x):             # 定义了前向传播网络, 接收输入张量 x 并返回张量; 前向传播函数定义了数据在网络中的流动方式
        x = self.linear1(x)
        x = self.relu1(x)
        out = self.linear2(x)
        return F.sigmoid(out)         # 输出值通过 Sigmoid 激活函数以限制在 [0, 1] 之间


# meta model
def load_VNet():
    model = VNet(1, 100, 1)
    return model