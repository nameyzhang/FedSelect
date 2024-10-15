import torch
import torch.nn as nn
import torchvision.models as models

from model.lenet import LeNet5
from model.model_resnet import ResNet18, ResNet34, ResNet32
from model.model_resnet_official import ResNet50

from utils.MyDataloader import build_dataset
from utils.utils import get_device






def build_model(args):

    data_train, data_val, data_test, num_classes = build_dataset(args.dataset_name)

    device = get_device(args)


    # choose different Neural network model for different args or input
    if args.model == 'lenet':
        netglob = LeNet5(num_classes)
        netglob = netglob.to(device)

        return netglob


    elif args.model == 'resnet18':
        netglob = ResNet18(num_classes)
        netglob = netglob.to(device)

        return netglob

    elif args.model == 'resnet32':
        netglob = ResNet32(args, num_classes)
        netglob = netglob.to(device)

        return netglob

    elif args.model == 'resnet34':
        netglob = ResNet34(num_classes)
        netglob = netglob.to(device)

        return netglob

    elif args.model == 'resnet50':
        netglob = ResNet50(pretrained=False)
        if args.pretrained:
            model = models.resnet50(pretrained=True)
            # Rename the 'fc' layer to 'fc1'
            model.fc1 = model.fc
            del model.fc
            netglob.load_state_dict(model.state_dict())

        netglob.fc1 = nn.Linear(2048, num_classes)
        netglob = netglob.to(device)

        return netglob

    elif args.model == 'vgg11':
        netglob = models.vgg11()
        netglob.fc = nn.Linear(4096, num_classes)
        netglob = netglob.to(device)

        return netglob

    else:
        exit('Error: unrecognized model')

    # return netglob, shared_model    # netglob = mian_model



