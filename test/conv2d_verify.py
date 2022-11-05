import torch

if __name__ == '__main__':
    x = torch.tensor([float(i) for i in range(2*2*2)], requires_grad=True).reshape(2, 1, 2, 2)
    y = torch.tensor([float(i) for i in range(2*3*3)]).reshape(2, 1, 3, 3)

    x.retain_grad()

    m = torch.nn.Conv2d(1, 2, 3, 1, 1, bias=False)
    m.weight.data = y.clone()

    print(x.data)
    print(m.weight.data)

    output = m(x)
    print(output)

    output.backward(torch.ones_like(output))

    print(x.grad)

    print(m.weight.grad)
