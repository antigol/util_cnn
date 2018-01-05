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

def import_module(path):
    file_name = os.path.basename(path)
    model_name = file_name.split('.')[0]
    module = imp.load_source(model_name, path)
    return module

class Dataset:
    def __init__(self, files, ids, labels):
        self.files = files
        self.ids = ids
        self.labels = labels

def load_data_with_csv(csv_file, files_pattern, classes=None):
    labels = {}

    with open(csv_file, 'rt') as file:
        reader = csv.reader(file)
        for row in reader:
            labels[row[0]] = row[1]

    files = glob.glob(files_pattern)
    random.shuffle(files)

    ids = [file.split("/")[-1].split(".")[0] for file in files]

    # keep only files that appears in the csv
    files, ids = zip(*[(f, i) for f, i in zip(files, ids) if i in labels])

    labels = [labels[i] for i in ids]
    if classes is None:
        classes = sorted(set(labels))
    labels = [classes.index(x) for x in labels]

    return Dataset(files, ids, labels), classes


def load_data(files_pattern):
    files = glob.glob(files_pattern)
    random.shuffle(files)

    ids = [file.split("/")[-1].split(".")[0] for file in files]

    return Dataset(files, ids, None)


def train_one_epoch(epoch, model, train_files, train_labels, optimizer, criterion, number_of_process, queue_size):
    cnn = model.get_cnn()
    logger = logging.getLogger("trainer")

    batches = model.create_train_batches(epoch, train_files, train_labels) # list of lists [first batch, second batch, ...]

    queue = torch.multiprocessing.Queue(maxsize=queue_size)
    event_done = torch.multiprocessing.Event()

    class Batcher(torch.multiprocessing.Process):
        def __init__(self, n=1, i=0):
            super().__init__(daemon=True)
            self.n = n
            self.i = i

        def run(self):
            for s, batch in enumerate(batches):
                if s % self.n == self.i:
                    gc.collect()
                    x, y = model.load_train_batch(batch)

                    queue.put((x, y))

            event_done.wait()

    for i in range(number_of_process):
        batcher = Batcher(number_of_process, i)
        batcher.start()

    losses = []
    total_correct = 0
    total_trained = 0

    cnn.train()
    if torch.cuda.is_available():
        cnn.cuda()

    for s, batch in enumerate(batches):
        t0 = perf_counter()
        gc.collect()

        t = time_logging.start()

        x, y = queue.get()

        x = torch.autograd.Variable(x)
        y = torch.autograd.Variable(y)

        t = time_logging.end("load batch", t)

        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        t = time_logging.end("upload batch", t)

        optimizer.zero_grad()
        outputs = cnn(x)
        loss = criterion(outputs, y)
        t = time_logging.end("forward", t)
        loss.backward()
        optimizer.step()

        t = time_logging.end("backward", t)

        loss_ = float(loss.data.cpu().numpy())
        losses.append(loss_)
        if outputs.size(-1) > 1:
            if y.dim() == 1:
                correct = sum(outputs.data.cpu().numpy().argmax(-1) == y.data.cpu().numpy())
            else:
                correct = sum(outputs.data.cpu().numpy().argmax(-1) == y.data.cpu().numpy().argmax(-1))
        else:
            correct = np.sum(np.sign(outputs.data.cpu().numpy().reshape((-1,))) == 2 * y.data.cpu().numpy() - 1)
        total_correct += correct
        total_trained += len(batch)

        logger.info("[%d.%.2d|%d/%d] Loss=%.1e <Loss>=%.1e Accuracy=%d/%d <Accuracy>=%.2f%% Queue=%d Memory=%s Time=%.2fs",
            epoch, 100 * s // len(batches), s, len(batches),
            loss_, np.mean(losses),
            correct, len(batch), 100 * total_correct / total_trained,
            queue.qsize(),
            gpu_memory.format_memory(gpu_memory.used_memory()),
            perf_counter() - t0)

        del x
        del y
        del outputs
        del loss

    event_done.set()
    return (np.mean(losses), total_correct / total_trained)


def evaluate(model, files, epoch, number_of_process, queue_size):
    cnn = model.get_cnn()
    bs = model.get_batch_size(epoch)
    logger = logging.getLogger("trainer")

    queue = torch.multiprocessing.Queue(maxsize=queue_size)
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
                    x = model.load_eval_files(files[i:j])

                    queue.put((s, x))
                s += 1
            event_done.wait()

    for i in range(number_of_process):
        batcher = Batcher(number_of_process, i)
        batcher.start()

    cnn.eval()
    if torch.cuda.is_available():
        cnn.cuda()

    all_outputs = [None] * len(range(0, len(files), bs))

    for i in range(0, len(files), bs):
        t = time_logging.start()

        gc.collect()
        s, x = queue.get()

        t = time_logging.end("load batch", t)

        if torch.cuda.is_available():
            x = x.cuda()

        t = time_logging.end("upload batch", t)

        outputs = model.evaluate(x)

        t = time_logging.end("forward", t)

        all_outputs[s] = outputs

        logger.info("Evaluation [%d.%.2d|%d/%d] Memory=%s Queue=%d",
            epoch, 100 * i // len(files), i, len(files),
            gpu_memory.format_memory(gpu_memory.used_memory()),
            queue.qsize())

        del s
        del x
        del outputs
    event_done.set()
    return np.concatenate(all_outputs, axis=0)


def save_evaluation(eval_ids, logits, labels, log_dir, number):
    if labels is None:
        labels = [-1] * len(logits)

    logits = np.array(logits)
    labels = np.array(labels)
    filename = os.path.join(log_dir, "eval{}.csv".format(number))

    with open(filename, "wt") as file:
        writer = csv.writer(file)

        for i, label, ilogits in zip(eval_ids, labels, logits):
            writer.writerow([i, label] + list(ilogits))

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
    classes = None

    train_data = None
    eval_datas = []

    if args.train_csv_path is not None or args.train_data_path is not None:
        train_data, classes = load_data_with_csv(args.train_csv_path, args.train_data_path, classes)
        logger.info("%s=%d training files", "+".join([str(train_data.labels.count(x)) for x in set(train_data.labels)]), len(train_data.files))

    if args.eval_data_path is not None and args.eval_csv_path is not None:
        assert len(args.eval_data_path) == len(args.eval_csv_path)

        for csv_file, pattern in zip(args.eval_csv_path, args.eval_data_path):
            eval_data, classes = load_data_with_csv(csv_file, pattern, classes)
            eval_datas.append(eval_data)
            logger.info("%s=%d evaluation files", "+".join([str(eval_data.labels.count(x)) for x in set(eval_data.labels)]), len(eval_data.files))
    elif args.eval_data_path is not None and args.eval_csv_path is None:
        for pattern in args.eval_data_path:
            eval_data = load_data(pattern)
            eval_datas.append(eval_data)
            logger.info("%d evaluation files", len(eval_data.files))
    elif args.eval_data_path is None and args.eval_csv_path is None:
        pass
    else:
        raise AssertionError("eval_data_path or eval_csv_path missing ?")

    if args.number_of_classes is not None and classes is None:
        classes = list(range(args.number_of_classes))

    ############################################################################
    # Import model
    model_path = shutil.copy2(args.model_path, os.path.join(args.log_dir, "model.py"))
    module = import_module(model_path)
    model = module.MyModel()
    model.initialize(number_of_classes=len(classes))
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

            time_logging.clear()
            t = time_logging.start()

            outputs = evaluate(model, data.files, -1, args.number_of_process, args.queue_size)

            time_logging.end("evaluation", t)
            logger.info("%s", time_logging.text_statistics())

            save_evaluation(data.ids, outputs, data.labels, args.log_dir, i)
            if data.labels is not None:
                if outputs.shape[-1] > 1:
                    correct = np.sum(np.argmax(outputs, axis=1) == np.array(data.labels, np.int64))
                else:
                    correct = np.sum(np.sign(outputs).reshape((-1,)) == 2 * np.array(data.labels, np.int64) - 1)

                logger.info("%d / %d = %.2f%%", correct, len(data.labels), 100 * correct / len(data.labels))
        return

    ############################################################################
    # Optimizer
    optimizer = model.get_optimizer()
    criterion = model.get_criterion()
    train_criterion = model.get_train_criterion()
    if torch.cuda.is_available():
        criterion.cuda()

    if args.restore_path is not None:
        checkpoint = torch.load(os.path.join(args.restore_path, "model.pkl"))
        optimizer.load_state_dict(checkpoint['optimizer'])

    ############################################################################
    # Training
    statistics_train = []
    statistics_eval = [[] for _ in eval_datas]

    if args.restore_path is not None:
        statistics_train = list(np.load(os.path.join(args.restore_path, "statistics_train.npy")))
        statistics_eval = [list(x) for x in np.load(os.path.join(args.restore_path, "statistics_eval.npy"))]
        logger.info("Restoration from numpy statistics files")

    if args.number_of_epochs is not None:
        number_of_epochs = args.number_of_epochs
    else:
        number_of_epochs = model.number_of_epochs()

    IPython.embed()

    for epoch in range(args.start_epoch, number_of_epochs):
        time_logging.clear()
        t = time_logging.start()

        lr = model.get_learning_rate(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        avg_loss, accuracy = train_one_epoch(epoch, model, train_data.files, train_data.labels, optimizer, train_criterion, args.number_of_process, args.queue_size)
        statistics_train.append([epoch, avg_loss, accuracy])

        model.training_done(avg_loss)

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
                time_logging.clear()
                t = time_logging.start()

                outputs = evaluate(model, data.files, epoch, args.number_of_process, args.queue_size)

                time_logging.end("evaluation", t)
                logger.info("%s", time_logging.text_statistics())

                save_evaluation(data.ids, outputs, data.labels, args.log_dir, i)

                if data.labels is not None:
                    if outputs.shape[-1] > 1:
                        correct = np.sum(np.argmax(outputs, axis=1) == np.array(data.labels, np.int64))
                    else:
                        correct = np.sum(np.sign(outputs).reshape((-1,)) == 2 * np.array(data.labels, np.int64) - 1)

                    criterion.cpu()
                    loss = criterion(
                        torch.autograd.Variable(torch.FloatTensor(outputs)),
                        torch.autograd.Variable(torch.LongTensor(data.labels))
                        ).data[0]
                    if torch.cuda.is_available():
                        criterion.cuda()
                    logger.info("Evaluation accuracy %d / %d = %.2f%%, Loss = %1e",
                        correct,
                        len(data.labels), 100 * correct / len(data.labels),
                        loss
                        )
                    stat.append([epoch, loss, correct / len(data.labels)])

        np.save(os.path.join(args.log_dir, "statistics_train.npy"), np.array(statistics_train))
        np.save(os.path.join(args.log_dir, "statistics_eval.npy"), np.array(statistics_eval))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--number_of_epochs", type=int)
    parser.add_argument("--start_epoch", type=int, default=0)

    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--train_csv_path", type=str)

    parser.add_argument("--eval_data_path", type=str, nargs="+")
    parser.add_argument("--eval_csv_path", type=str, nargs="+")
    parser.add_argument("--eval_each", type=int, default=1)

    parser.add_argument("--number_of_classes", type=int)

    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--restore_path", type=str)

    parser.add_argument("--number_of_process", type=int, default=1)
    parser.add_argument("--queue_size", type=int, default=4)

    args = parser.parse_args()

    train(args)

if __name__ == '__main__':
    main()
