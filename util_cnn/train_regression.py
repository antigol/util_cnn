#pylint: disable=C,R,E1101
import os
import imp
import argparse
import logging
import csv
import glob
import random
import numpy as np
import torch
import gc
import shutil
from time import perf_counter
from util_cnn import gpu_memory
from util_cnn import time_logging
import IPython

QUEUE_SIZE = 4

def import_module(path):
    file_name = os.path.basename(path)
    model_name = file_name.split('.')[0]
    module = imp.load_source(model_name, path)
    return module

class Dataset:
    def __init__(self, files, ids):
        self.files = files
        self.ids = ids


def load_data(files_pattern):
    files = glob.glob(files_pattern)
    random.shuffle(files)

    ids = [file.split("/")[-1].split(".")[0] for file in files]

    return Dataset(files, ids)


def train_one_epoch(epoch, model, train_files, optimizer, criterion, number_of_process):
    cnn = model.get_cnn()
    bs = model.get_batch_size(epoch)
    logger = logging.getLogger("trainer")

    indicies = list(range(len(train_files)))
    random.shuffle(indicies)

    queue = torch.multiprocessing.Queue(maxsize=QUEUE_SIZE)
    event_done = torch.multiprocessing.Event()

    class Batcher(torch.multiprocessing.Process):
        def __init__(self, n=1, i=0):
            super().__init__(daemon=True)
            self.n = n
            self.i = i

        def run(self):
            s = 0

            for i in range(0, len(train_files), bs):
                if s % self.n == self.i:
                    j = min(i + bs, len(train_files))
                    gc.collect()
                    x, y = model.load_train_files([train_files[g] for g in indicies[i:j]])

                    queue.put((x, y))
                s += 1
            event_done.wait()

    for i in range(number_of_process):
        batcher = Batcher(number_of_process, i)
        batcher.start()

    losses = []

    cnn.train()
    if torch.cuda.is_available():
        cnn.cuda()

    for i in range(0, len(train_files), bs):
        t0 = perf_counter()
        gc.collect()

        t = time_logging.start()

        x, y = queue.get()

        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)

        x = torch.autograd.Variable(x)
        y = torch.autograd.Variable(y)

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        t = time_logging.end("batch", t)

        optimizer.zero_grad()
        outputs = cnn(x)
        loss = criterion(outputs, y)
        t = time_logging.end("forward", t)
        loss.backward()
        optimizer.step()

        t = time_logging.end("backward", t)

        loss_ = float(loss.data.cpu().numpy())
        losses.append(loss_)

        logger.info("[%d.%.2d|%d/%d] RMSE=%.1e <RMSE>=%.1e Queue=%d Memory=%s Time=%.2fs",
            epoch, 100 * i // len(train_files), i, len(train_files),
            loss_ ** 0.5, np.mean(losses) ** 0.5,
            queue.qsize(),
            gpu_memory.format_memory(gpu_memory.used_memory()),
            perf_counter() - t0)

        del x
        del y
        del outputs
        del loss

    event_done.set()
    return np.mean(losses)


def evaluate(model, files, epoch=0, number_of_process=1):
    cnn = model.get_cnn()
    bs = model.get_batch_size()
    logger = logging.getLogger("trainer")

    queue = torch.multiprocessing.Queue(maxsize=QUEUE_SIZE)
    event_done = torch.multiprocessing.Event()

    class Batcher(torch.multiprocessing.Process):
        def __init__(self, n=1, i=0):
            super().__init__(daemon=True)
            self.n = n
            self.i = i

        def run(self):
            s = 0

            for i in range(0, len(files), bs):
                if s % self.n == self.i:
                    j = min(i + bs, len(files))
                    gc.collect()
                    x, y = model.load_eval_files(files[i:j])

                    queue.put((s, x, y))
                s += 1
            event_done.wait()

    for i in range(number_of_process):
        batcher = Batcher(number_of_process, i)
        batcher.start()

    cnn.eval()
    if torch.cuda.is_available():
        cnn.cuda()

    all_outputs = [None] * len(range(0, len(files), bs))
    all_targets = [None] * len(range(0, len(files), bs))

    for i in range(0, len(files), bs):
        gc.collect()
        s, x, y = queue.get()

        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y)

        if torch.cuda.is_available():
            x = x.cuda()

        outputs = model.evaluate(x)

        all_outputs[s] = outputs
        all_targets[s] = y.numpy()

        logger.info("Evaluation [%d.%.2d|%d/%d] Memory=%s Queue=%d",
            epoch, 100 * i // len(files), i, len(files),
            gpu_memory.format_memory(gpu_memory.used_memory()),
            queue.qsize())

        del s
        del x
        del y
        del outputs
    event_done.set()
    return np.concatenate(all_outputs, axis=0), np.concatenate(all_targets, axis=0)


def save_evaluation(eval_ids, logits, log_dir, number):
    logits = np.array(logits)
    filename = os.path.join(log_dir, "eval{}.csv".format(number))

    with open(filename, "wt") as file:
        writer = csv.writer(file)

        for i, ilogits in zip(eval_ids, logits):
            writer.writerow([i] + list(ilogits))

    logging.getLogger("trainer").info("Evaluation saved into %s", filename)


def train(args):

    if os.path.isdir(args.log_dir):
        print("{} exists already".format(args.log_dir))
        return

    os.mkdir(args.log_dir)

    logger = logging.getLogger("trainer")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    fh = logging.FileHandler(os.path.join(args.log_dir, "log.txt"))
    logger.addHandler(fh)

    logger.info("Arguments = %s", repr(args))

    ############################################################################
    # Files and labels
    train_data = None
    eval_datas = []

    if args.train_data_path is not None:
        train_data = load_data(args.train_data_path)
        logger.info("%d training files", len(train_data.files))

    if args.eval_data_path is not None:
        for pattern in args.eval_data_path:
            eval_data = load_data(pattern)
            eval_datas.append(eval_data)
            logger.info("%d evaluation files", len(eval_data.files))

    ############################################################################
    # Import model
    model_path = shutil.copy2(args.model_path, os.path.join(args.log_dir, "model.py"))
    module = import_module(model_path)
    model = module.MyModel()
    model.initialize()
    cnn = model.get_cnn()

    logger.info("There is %d parameters to optimize", sum([x.numel() for x in cnn.parameters()]))

    if args.restore_path is not None:
        restore_path = shutil.copy2(
            os.path.join(args.restore_path, "model.pkl"),
            os.path.join(args.log_dir, "model.pkl"))
        checkpoint = torch.load(restore_path)
        args.start_epoch = checkpoint['epoch']
        cnn.load_state_dict(checkpoint['state_dict'])
        logger.info("Restoration from file %s", os.path.join(args.restore_path, "model.pkl"))

    ############################################################################
    # Only evaluation
    if train_data is None:
        if args.restore_path is None:
            logger.info("Evalutation with randomly initialized parameters")
        for i, data in enumerate(eval_datas):
            outputs, targets = evaluate(model, data.files, number_of_process=args.number_of_process)
            save_evaluation(data.ids, outputs, args.log_dir, i)
            rmse = np.mean((outputs - targets) ** 2) ** 0.5
            logger.info("Evaluation RMSE = %f", rmse)
        return

    ############################################################################
    # Optimizer
    optimizer = model.get_optimizer()
    criterion = model.get_criterion()
    if torch.cuda.is_available():
        criterion.cuda()

    if args.restore_path is not None:
        checkpoint = torch.load(os.path.join(args.restore_path, "model.pkl"))
        optimizer.load_state_dict(checkpoint['optimizer'])

    ############################################################################
    # Training
    statistics_train = []
    statistics_eval = [[] for _ in eval_datas]

    IPython.embed()

    for epoch in range(args.start_epoch, args.number_of_epochs):
        time_logging.clear()
        t = time_logging.start()

        lr = model.get_learning_rate(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        avg_loss = train_one_epoch(epoch, model, train_data.files, optimizer, criterion, args.number_of_process)
        statistics_train.append([epoch, avg_loss])

        model.training_done(avg_loss, None)

        time_logging.end("training epoch", t)
        logger.info("%s", time_logging.text_statistics())

        cnn.cpu()
        path = os.path.join(args.log_dir, 'model.pkl')
        torch.save({
            'epoch': epoch + 1,
            'state_dict': cnn.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, path)
        logger.info("Saved in %s", path)

        if epoch % args.eval_each == args.eval_each - 1:
            for i, (data, stat) in enumerate(zip(eval_datas, statistics_eval)):
                outputs, targets = evaluate(model, data.files, epoch, number_of_process=args.number_of_process)
                save_evaluation(data.ids, outputs, args.log_dir, i)
                rmse = np.mean((outputs - targets) ** 2) ** 0.5
                logger.info("Evaluation RMSE = %f", rmse)
                stat.append([epoch, rmse])

    statistics_train = np.array(statistics_train)
    np.save(os.path.join(args.log_dir, "statistics_train.npy"), statistics_train)
    statistics_eval = np.array(statistics_eval)
    np.save(os.path.join(args.log_dir, "statistics_eval.npy"), statistics_eval)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--number_of_epochs", type=int)
    parser.add_argument("--start_epoch", type=int, default=0)

    parser.add_argument("--train_data_path", type=str)

    parser.add_argument("--eval_data_path", type=str, nargs="+")
    parser.add_argument("--eval_each", type=int, default=1)

    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--restore_path", type=str)

    parser.add_argument("--number_of_process", type=int, default=4)

    args = parser.parse_args()

    train(args)

if __name__ == '__main__':
    main()
