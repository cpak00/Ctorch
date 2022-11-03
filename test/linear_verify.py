import torch

if __name__ == '__main__':
    x = torch.tensor([[0., 1., 2.], [3., 4., 5.]], requires_grad=True)
    y = torch.tensor([[0., 1.], [2., 3.], [4., 5.]], requires_grad=True)
    w = torch.tensor([[0.], [1.]], requires_grad=True)

    l = torch.nn.Linear(3, 2, False)
    l.weight.data = y.T

    l2 = torch.nn.Linear(2, 1, False)
    l2.weight.data = w.T

    out = l(x)
    out2 = l2(out)
    print(out2)
    out2.backward(torch.ones(2, 1))
    print("x grad: ", x.grad)
    print("y grad: ", l.weight.grad)
    print("w grad: ", l2.weight.grad)
