import argparse
import torch
from train import trainer
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.0, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--load_weights', action='store_true', default=False,
                    help='load pretrained weights')
parser.add_argument("--pretrained_weights", type=str,
                    default="./checkpoints/expbs1024_epoch_2_model.pth")
parser.add_argument("--model_name", type=str,
                    default="bs64start1024e30lr10x")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

_trainer = trainer(args)


def train_model():
    for epoch in range(1, args.epochs + 1):
        _trainer.train(epoch)


def experiment1(repeats, weight_file):
    maxbs = 1024
    values = [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 512, 1024]
    gradients = {}

    for trial in tqdm(range(repeats)):
        idxs = np.random.randint(0, len(_trainer.train_loader.dataset), maxbs).tolist()

        data = []
        target = []
        for i in idxs:
            item = _trainer.train_loader.dataset[i]
            data.append(item[0])
            target.append(item[1])

        data = torch.stack(data)
        target = torch.Tensor(target).long()
        data, target = Variable(data, requires_grad=False).cuda(), Variable(target, requires_grad=False).cuda()
        for bs in values:
            gradient_sum = inner(bs, maxbs, data, target, weight_file)

            if bs in gradients:
                gradients[bs].append(gradient_sum)
            else:
                gradients[bs] = [gradient_sum]

    torch.save(gradients, './results/{}.result'.format(weight_file))


def inner(bs, max_bs, data, target, weight_file):
    weight_path = "./checkpoints/" + weight_file
    _trainer.load_weights(weight_path)
    _trainer.model.train()

    grad_sum = torch.zeros(1024, 10)

    for i in range(max_bs/bs):
        _trainer.optimizer.zero_grad()
        minibatch_data = data[i*bs: i*bs+bs]
        minibatch_target = target[i*bs: i*bs+bs]
        output = _trainer.model(minibatch_data)
        loss = F.nll_loss(output, minibatch_target)
        loss.backward()

        for key, value in _trainer.model.named_parameters():
            if key == "fc3.weight":
                grad_sum = grad_sum + value.grad.data.cpu()

        _trainer.optimizer.step()

    # each minibatch contributes magnitude X so maxbs/bs minibatches gives maxbs/bs times the magnitude
    grad_sum = grad_sum*(float(bs)/float(max_bs))
    return grad_sum


if __name__ == "__main__":
    # train_model()
    experiment1(1000, 'expbs1024_epoch_2_model.pth')
    experiment1(1000, 'expbs1024_epoch_30_model.pth')
    print("Done")
