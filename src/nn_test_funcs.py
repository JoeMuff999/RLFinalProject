from nn import PolicyNN
import torch
import random

def test1():
    pi = PolicyNN(2, 1, 32, 1)
    optim = torch.optim.SGD(pi.parameters(), lr=1e-3, momentum=0.9)

    pi.train()

    for i in range(10000):
        in_x = random.random() * 5
        in_y = random.random() * 5
        answer = torch.tensor([in_x + in_y])
        prediction = pi(torch.tensor([in_x, in_y]))
        lossFN = torch.nn.MSELoss()
        # print(prediction)
        loss = abs(prediction - answer)
        # print(loss)
        optim.zero_grad()
        loss.backward()
        optim.step()

    print("test 1")
    pi.eval()
    print(pi(torch.tensor([2.0, 2.0])))
    for param in pi.parameters():
        print(param)

def test3():
    pi = PolicyNN(2, 1, 4, 2)
    optim = torch.optim.SGD(pi.parameters(), lr=1e-3, momentum=0.9)

    pi.train()

    for i in range(10000):
        in_x = random.random() * 5
        in_y = random.random() * 5
        answer = torch.tensor([in_x, in_y])
        prediction = pi(torch.tensor([in_x, in_y]))
        lossFN = torch.nn.MSELoss()
        # print(prediction)
        loss = lossFN(prediction, answer)
        # print(loss)
        optim.zero_grad()
        loss.backward()
        optim.step()

    print("test 3")
    pi.eval()
    # grads = torch.autograd.grad(loss, pi.parameters())
    # print(pi(torch.tensor([2.0, 2.0])).data[0])
    # print(grads)
        
    # print(pi(torch.tensor([2.0, 2.0])))
    

def test2():
    pi = PolicyNN()
    def get_kl():
        mean1, log_std1, std1 = PolicyNN([])

        mean0 = mean1.data
        # log_std0 = log_std1.data
        # std0 = std1.data
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

test3()