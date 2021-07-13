import torch


def binary_cross_entropy(r, x, eps):
    return -torch.mean(x*torch.log(torch.sigmoid(r) + eps) + (1 - x)*torch.log(1 - torch.sigmoid(r) + eps))


def f1_accuracy_metrics_oncuda(y_hat_hard: torch.Tensor, y: torch.Tensor, epsilon=1e-8) -> (torch.Tensor, torch.Tensor):
    y_pred = y_hat_hard.cuda()
    y_true = y.cuda()

    assert y_true.dim() == 1
    assert y_pred.dim() == 1 or y_pred.dim() == 2

    if y_pred.dim() == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    tnr = tn / (tn + fp + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return f1, accuracy, precision, recall, tnr
