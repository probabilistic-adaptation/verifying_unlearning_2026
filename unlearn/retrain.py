from trainer import train

from .impl import iterative_unlearn


@iterative_unlearn
def retrain(retain_loader, model, criterion, optimizer, epoch, print_freq):
    return train(retain_loader, model, criterion, optimizer, epoch, print_freq)
