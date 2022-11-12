import torch

if __name__ == '__main__':
    x = torch.tensor([float(i) for i in range(2*3*4*4)], requires_grad=True).reshape(2, 3, 4, 4)
    y = torch.tensor([float(i) for i in range(5*3*3*3)]).reshape(5, 3, 3, 3)

    x.retain_grad()

    m = torch.nn.Conv2d(3, 5, 3, 1, 1, bias=True)
    m.weight.data = y.clone()
    for i in range(m.bias.size()[0]):
        m.bias.data[i] = i
    

    print(x.data)
    print(m.weight.data)

    output = m(x)
    print(output)

    grad = torch.ones_like(output)

    grad[0][0][1][0] = 1.

    output.backward(grad)

    print(m.weight.grad)

    print(x.grad)

    print(m.bias.grad)
