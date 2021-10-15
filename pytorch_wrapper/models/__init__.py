from torch import nn

class nnModule(nn.Module):

    def __init__(self):
        super(nnModule, self).__init__()

    def on_fit_start(self, trainer):
        pass

    def on_fit_end(self, trainer):
        pass

    def on_epoch_start(self, trainer):
        pass

    def on_epoch_end(self, trainer):
        pass

    def on_train_start(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_validation_start(self, trainer):
        pass

    def on_validation_end(self, trainer):
        pass

    def on_batch_start(self, trainer):
        pass

    def on_validation_batch_start(self, trainer, batch, batch_idx):
        pass

    def on_validation_batch_end(self, trainer, outputs, batch, batch_idx):
        pass

    def on_batch_end(self, trainer):
        pass

    def on_before_backward(self, trainer, loss):
        pass

    def on_after_backward(self, trainer):
        pass

    def on_before_optimizer_step(self, trainer, optimizer):
        pass

    def on_before_zero_grad(self, trainer, optimizer):
        pass
        
    def on_batch_end(self, trainer):
        pass

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("training_step has not been implmented")

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError("validation_step has not been implmented")
