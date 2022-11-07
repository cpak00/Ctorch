import torch

if __name__ == '__main__':
    cri = torch.nn.CrossEntropyLoss()
    output = torch.tensor([[-0.1, 0.2, 0.3, 0.4], [0.2, 0.1, -0.3, 0.1]], requires_grad = True)
    label = torch.tensor([2, 1])

    l = cri(output, label)
    print(l)
    l.backward()
    print(output.grad)

