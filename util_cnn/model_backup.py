#pylint: disable=C,R,E1101
from util_cnn.model import Model
import logging
import copy

class ModelBackup(Model):
    def __init__(self,
        success_factor=2 ** (1/4),
        decay_factor=2 ** (-1/8),
        reject_factor=2 ** (-1/2),
        reject_ratio=1.5,
        min_learning_rate=1e-5,
        max_learning_rate=0.2,
        initial_learning_rate=1e-2):

        self.cnn = None
        self.optimizer = None

        self.best = None

        self.learning_rate = initial_learning_rate
        self.true_epoch = 0

        self.SUCCESS_FACTOR = success_factor
        self.DECAY_FACTOR = decay_factor
        self.REJECT_FACTOR = reject_factor
        self.REJECT_RATIO = reject_ratio
        self.MIN_LEARNING_RATE = min_learning_rate
        self.MAX_LEARNING_RATE = max_learning_rate


    def initialize(self, **kargs):
        """
        self.cnn and self.optimizer must be created in this function
        """
        raise NotImplementedError

    def get_cnn(self):
        return self.cnn

    def get_learning_rate(self, epoch):
        logger = logging.getLogger("trainer")
        logger.info("[%d] Learning rate set to %.1e", epoch, self.learning_rate)
        return self.learning_rate

    def get_optimizer(self):
        return self.optimizer

    def training_done(self, avg_loss):
        '''
        3 possibilities

        1. Improve best loss
        2. Does not improve but not rejected
        3. Incrase too much the loss
        '''
        logger = logging.getLogger("trainer")

        # only after first epoch
        if self.best is None:
            self.best = {
                'loss': avg_loss,
                'cnn_state': copy.deepcopy(self.cnn.state_dict()),
                'optimizer_state': copy.deepcopy(self.optimizer.state_dict()),
                'learning_rate': self.learning_rate,
                'epoch': self.true_epoch
            }
            self.true_epoch += 1
            return

        if avg_loss < self.best['loss']:
            # Case 1 : Improve best loss
            logger.info("Loss %.1e -> %.1e (improve)", self.best['loss'], avg_loss)
            self.best = {
                'loss': avg_loss,
                'cnn_state': copy.deepcopy(self.cnn.state_dict()),
                'optimizer_state': copy.deepcopy(self.optimizer.state_dict()),
                'learning_rate': self.learning_rate,
                'epoch': self.true_epoch
            }
            self.learning_rate = min(self.SUCCESS_FACTOR * self.learning_rate, self.MAX_LEARNING_RATE)
            self.true_epoch += 1
            return
        elif avg_loss < self.REJECT_RATIO * self.best['loss']:
            # Case 2 : Does not improve but not rejected
            self.true_epoch += 1
            logger.info("Loss %.1e -> %.1e (worst)", self.best['loss'], avg_loss)
            self.learning_rate = max(
                min(
                    self.DECAY_FACTOR * self.learning_rate,
                    self.best['learning_rate']),
                self.MIN_LEARNING_RATE)
            return
        else:
            # Case 3 : Incrase too much the loss
            logger.info("Loss %.1e -> %.1e (rejected)", self.best['loss'], avg_loss)
            self.learning_rate = max(
                self.REJECT_FACTOR * min(
                    self.learning_rate,
                    self.best['learning_rate']),
                self.MIN_LEARNING_RATE)
            self.cnn.load_state_dict(self.best['cnn_state'])
            self.optimizer.load_state_dict(self.best['optimizer_state'])
            self.true_epoch = self.best['epoch']
            return
