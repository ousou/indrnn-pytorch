from indrnn import SingleLayerIndRNN, TwoLayerIndRNN
import torch
import torch.nn as nn
import argparse


class AddingProblemIndRNN(nn.Module):

    def __init__(self, hidden_size):
        super(AddingProblemIndRNN, self).__init__()
        self.indrnn = TwoLayerIndRNN(2, hidden_size)
        self.final_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        hidden_state = self.indrnn(x)
        return self.final_layer(hidden_state).squeeze()

def train(model, data, target, loss_fn, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()

def test(model, data, target, loss_fn):
    model.eval()
    output = model(data)
    loss = loss_fn(output, target)
    return loss.item()

def generate_data(time_steps, data_points, device):
    x = torch.zeros(data_points, time_steps, 2)
    target = torch.zeros(data_points)
    x[:,:,0] = torch.rand(data_points, time_steps)
    for i in range(data_points):
        one_indices = torch.randperm(time_steps)[:2]
        x[i,:,1][one_indices] = 1
        target[i] = torch.sum(x[i,:,0][one_indices])
    x.to(device)
    target.to(device)
    return x, target


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch IndRNN adding problem Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--steps', type=int, default=50000, metavar='N',
                        help='number of training steps to take (default: 50000)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--sequence-length', type=int, default=10, metavar='N',
                        help='Length of training sequence')
    parser.add_argument('--hidden-size', type=int, default=5, metavar='N',
                        help='Size of hidden layer in IndRNN')
    args = parser.parse_args()
    print('Training with arguments', args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    model = AddingProblemIndRNN(args.hidden_size)
    optimizer = torch.optim.Adam(lr=args.lr, params=model.parameters())
    loss = nn.MSELoss()
    lr = args.lr
    for steps in range(1, args.steps + 1):
        x, y = generate_data(args.sequence_length, args.batch_size, device)
        train(model, x, y, loss, optimizer)
        if steps % 1000 == 0:
            x_test, y_test = generate_data(args.sequence_length, args.test_batch_size, device)
            test_loss = test(model, x_test, y_test, loss)
            print('Test Step: {} \tLoss: {:.6f}'.format(
                steps, test_loss))
        if steps % 10000 == 0:
            lr = lr / 10
            print('Decreasing learning rate to', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr



if __name__ == '__main__':
    main()
