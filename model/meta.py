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
