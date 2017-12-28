#pylint: disable=C,R,E1101
import torch
import random

class Model:
    def initialize(self, **kargs):
        pass

    def get_cnn(self):
        """
        Returns a torch.nn.Module
        """
        raise NotImplementedError

    def get_batch_size(self, epoch=None):
        raise NotImplementedError

    def create_train_batches(self, epoch, files, labels): #pylint: disable=W0613
        bs = self.get_batch_size(epoch)

        indices = list(range(len(files)))
        random.shuffle(indices)

        return [[(files[j], labels[j]) for j in indices[i: i + bs]] for i in range(0, len(files), bs)]

    def get_learning_rate(self, epoch):
        raise NotImplementedError

    def number_of_epochs(self):
        raise NotImplementedError

    def get_optimizer(self):
        return torch.optim.Adam(self.get_cnn().parameters())

    def get_criterion(self):
        return torch.nn.CrossEntropyLoss()

    def get_train_criterion(self):
        return self.get_criterion()

    def load_eval_files(self, files):
        """
        Returns a torch.FloatTensor
        """
        raise NotImplementedError

    def load_train_batch(self, batch):
        """
        :param: batch : a list of things that ceate a batch
        :return: (images, labels)
        """
        x = self.load_eval_files([f for f, l in batch])
        y = torch.LongTensor([l for f, l in batch])
        return x, y

    def evaluate(self, x):
        x = torch.autograd.Variable(x, volatile=True)
        y = self.get_cnn()(x)
        return y.data.cpu().numpy()

    def training_done(self, avg_loss):
        pass
