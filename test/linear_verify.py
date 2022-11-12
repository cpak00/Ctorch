import torch

if __name__ == '__main__':
    x = torch.tensor([[0., 1., 2.], [3., 4., 5.]], requires_grad=True)
    # y = torch.tensor([[0., 1., 2.], [3., 4., 5.]], requires_grad=True)
    w = torch.tensor([[0.], [1.]], requires_grad=True)

    l = torch.nn.Linear(3, 2, True)
    l.weight.data = torch.tensor([float(i) for i in range(3*2)]).reshape(2, 3)
    l.bias.data = torch.tensor([1., 2.])

    l2 = torch.nn.Linear(2, 1, True)
    l2.weight.data = w.T
    l2.bias.data = torch.tensor([1.])

    out = l(x)
    out2 = l2(out)
    print(out)
    out2.backward(torch.ones(2, 1))
    print("output data: ", out2)
    print("x grad: ", x.grad)
    print("y grad: ", l.weight.grad)
    print("w grad: ", l2.weight.grad)

    print(l.bias.grad)
    print(l2.bias.grad)
