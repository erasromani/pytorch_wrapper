class Callback:

    def __init__(self):
        pass

    def on_fit_start(self, trainer, model):
        pass

    def on_fit_end(self, trainer, model):
        pass

    def on_epoch_start(self, trainer, model):
        pass

    def on_epoch_end(self, trainer, model):
        pass

    def on_train_start(self, trainer, model):
        pass

    def on_train_end(self, trainer, model):
        pass

    def on_validation_start(self, trainer, model):
        pass

    def on_validation_end(self, trainer, model):
        pass

    def on_batch_start(self, trainer, model):
        pass

    def on_validation_batch_start(self, trainer, model, batch, batch_idx):
        pass

    def on_validation_batch_end(self, trainer, model, outputs, batch, batch_idx):
        pass

    def on_batch_end(self, trainer, model):
        pass

    def on_before_backward(self, trainer, model, loss):
        pass

    def on_after_backward(self, trainer, model):
        pass

    def on_before_optimizer_step(self, trainer, model, optimizer):
        pass

    def on_before_zero_grad(self, trainer, model, optimizer):
        pass
        
    def on_batch_end(self, trainer, model):
        pass
