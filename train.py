""" General structure of train.py borrowed from https://github.com/JiahuiYu/slimmable_networks """

import importlib
import os
import time
import random
import sys

import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.utils as vutils
import numpy as np

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

from genutil.config import FLAGS
from genutil.model_profiling import model_profiling

from tensorboardX import SummaryWriter

best_acc1 = 0
writer = None


def getter(name):
    name = getattr(FLAGS, name)
    if ":" in name:
        name = name.split(":")
        return getattr(importlib.import_module(name[0]), name[1])
    return importlib.import_module(name)


def get_lr_scheduler(optimizer):
    """get learning rate"""
    if FLAGS.lr_scheduler == "multistep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=FLAGS.multistep_lr_milestones,
            gamma=FLAGS.multistep_lr_gamma,
        )
    elif FLAGS.lr_scheduler == "exp_decaying":
        lr_dict = {}
        for i in range(FLAGS.num_epochs):
            if i == 0:
                lr_dict[i] = 1
            else:
                lr_dict[i] = lr_dict[i - 1] * FLAGS.exp_decaying_lr_gamma
        lr_lambda = lambda epoch: lr_dict[epoch]
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif FLAGS.lr_scheduler == "linear_decaying":
        lr_dict = {}
        for i in range(FLAGS.num_epochs):
            lr_dict[i] = 1.0 - i / FLAGS.num_epochs
        lr_lambda = lambda epoch: lr_dict[epoch]
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif FLAGS.lr_scheduler == "cosine":
        if hasattr(FLAGS, "epoch_len"):
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, FLAGS.epoch_len * FLAGS.num_epochs
            )
        else:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, FLAGS.len_loader * FLAGS.num_epochs
            )
    else:
        try:
            lr_scheduler_lib = importlib.import_module(FLAGS.lr_scheduler)
            return lr_scheduler_lib.get_lr_scheduler(optimizer)
        except ImportError:
            raise NotImplementedError(
                "Learning rate scheduler {} is not yet implemented.".format(
                    FLAGS.lr_scheduler
                )
            )
    return lr_scheduler


def get_optimizer(model):
    """get optimizer"""
    if FLAGS.optimizer == "sgd":

        optimizer = torch.optim.SGD(
            model.parameters(),
            FLAGS.lr,
            momentum=FLAGS.momentum,
            weight_decay=FLAGS.weight_decay,
            nesterov=FLAGS.nestorov,
        )
    else:
        try:
            optimizer_lib = importlib.import_module(FLAGS.optimizer)
            return optimizer_lib.get_optimizer(model)
        except ImportError:
            raise NotImplementedError(
                "Optimizer {} is not yet implemented.".format(FLAGS.optimizer)
            )
    return optimizer


class Meter(object):
    """Meter is to keep track of statistics along steps.
    Meters cache values for purpose like printing average values.
    Meters can be flushed to log files (i.e. TensorBoard) regularly.
    Args:
        name (str): the name of meter
    """

    def __init__(self, name):
        self.name = name
        self.steps = 0
        self.reset()

    def reset(self):
        self.values = []

    def cache(self, value, pstep=1):
        self.steps += pstep
        self.values.append(value)

    def cache_list(self, value_list, pstep=1):
        self.steps += pstep
        self.values += value_list

    def flush(self, value, reset=True):
        pass


class ScalarMeter(Meter):
    """ScalarMeter records scalar over steps.
    """

    def __init__(self, name):
        super(ScalarMeter, self).__init__(name)

    def flush(self, value, step=-1, reset=True):
        if reset:
            self.reset()


def flush_scalar_meters(meters, method="avg"):
    """Docstring for flush_scalar_meters"""
    results = {}
    assert isinstance(meters, dict), "meters should be a dict."
    for name, meter in meters.items():
        if not isinstance(meter, ScalarMeter):
            continue
        if method == "avg":
            if len(meter.values) == 0:
                value = 0
            else:
                value = sum(meter.values) / len(meter.values)
        elif method == "sum":
            value = sum(meter.values)
        elif method == "max":
            value = max(meter.values)
        elif method == "min":
            value = min(meter.values)
        else:
            raise NotImplementedError(
                "flush method: {} is not yet implemented.".format(method)
            )
        results[name] = value
        meter.flush(value)
    return results


def set_random_seed():
    """set random seed"""
    if hasattr(FLAGS, "random_seed"):
        seed = FLAGS.random_seed
    else:
        seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_meters(phase, model):
    """util function for meters"""
    meters = {}
    meters["CELoss"] = ScalarMeter("{}_CELoss".format(phase))
    for k in FLAGS.topk:
        meters["top{}_accuracy".format(k)] = ScalarMeter(
            "{}_top{}_accuracy".format(phase, k)
        )

    if hasattr(model, 'module') and hasattr(model.module, "__losses__"):
        loss_info = model.module.__losses__
        for i in range(1, len(loss_info)):
            meters[loss_info[i][0]] = ScalarMeter(
                "{}_{}".format(loss_info[i][0], phase)
            )
        meters["total_loss"] = ScalarMeter("{}_total_loss".format(phase))

    return meters


def forward_loss(model, criterion, input, target, meter):
    """forward model """

    output = model(input)
    if type(output) is tuple:
        assert hasattr(model.module, "__losses__")
        losses_info = model.module.__losses__

        loss = torch.mean(criterion(output[0], target))
        meter["CELoss"].cache(loss.cpu().detach().numpy())
        loss = loss * losses_info[0][1]

        for i in range(1, len(output)):
            ext_loss = torch.mean(output[i])
            meter[losses_info[i][0]].cache(ext_loss.cpu().detach().numpy())
            loss = loss + ext_loss * losses_info[i][1]

        meter["total_loss"].cache(loss.cpu().detach().numpy())

        output = output[0]
    else:
        loss = torch.mean(criterion(output, target))
        meter["CELoss"].cache(loss.cpu().detach().numpy())

    # topk
    _, pred = output.topk(max(FLAGS.topk))
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    for k in FLAGS.topk:
        correct_k = correct[:k].float().sum(0)
        accuracy_list = list(correct_k.cpu().detach().numpy())
        meter["top{}_accuracy".format(k)].cache_list(accuracy_list)

    return loss


def run_one_epoch(
    epoch,
    loader,
    model,
    criterion,
    optimizer,
    meters,
    phase="train",
    iter=0.0,
    scheduler=None,
):
    """run one epoch for train/val/test"""
    print("epoch:", epoch, "phase:", phase)

    model.apply(lambda m: setattr(m, "epoch", epoch))

    t_start = time.time()
    assert phase in ["train", "val", "test"], "phase not be in train/val/test."
    train = phase == "train"
    if train:
        model.train()
    else:
        model.eval()

    if train and FLAGS.lr_scheduler == "linear_decaying":
        if hasattr(FLAGS, "epoch_len"):
            linear_decaying_per_step = (
                FLAGS.lr / FLAGS.num_epochs / FLAGS.epoch_len * FLAGS.batch_size
            )
        else:
            linear_decaying_per_step = (
                FLAGS.lr / FLAGS.num_epochs / len(loader.dataset) * FLAGS.batch_size
            )

    end = time.time()
    for batch_idx, (input, target) in enumerate(loader):

        data_time = time.time() - end

        input, target = (
            input.to(FLAGS.device, non_blocking=True),
            target.to(FLAGS.device, non_blocking=True),
        )

        if train:

            ############################## Train ################################
            if FLAGS.lr_scheduler == "linear_decaying":
                for param_group in optimizer.param_groups:
                    param_group["lr"] -= linear_decaying_per_step
            elif FLAGS.lr_scheduler == "cosine":
                scheduler.step()

            iter += 1

            optimizer.zero_grad()

            loss = forward_loss(model, criterion, input, target, meters)
            loss.backward()
            optimizer.step()

        else:
            ############################### VAL #################################

            loss = forward_loss(model, criterion, input, target, meters)

        batch_time = time.time() - end
        end = time.time()

        if (batch_idx % 10) == 0:
            print(
                "Epoch: [{}][{}/{}]\tTime {:.3f}\tData {:.3f}\tLoss {:.3f}\t".format(
                    epoch, batch_idx, len(loader), batch_time, data_time, loss.item()
                )
            )

    # Log.
    writer.add_scalar(phase + "/epoch_time", time.time() - t_start, epoch)
    results = flush_scalar_meters(meters)
    print(
        "{:.1f}s\t{}\t{}/{}: ".format(
            time.time() - t_start, phase, epoch, FLAGS.num_epochs
        )
        + ", ".join("{}: {:.3f}".format(k, v) for k, v in results.items())
    )

    for k, v in results.items():
        if k != "best_val":
            writer.add_scalar(phase + "/" + k, v, epoch)

    # Visualize the adjacency matrix.
    if hasattr(model.module, "get_weight"):
        weights = model.module.get_weight()
        if type(weights) is list:
            for i, w in enumerate(weights):
                w = w.squeeze().t()
                nz = (w != 0).float()
                nz_grid = vutils.make_grid(nz)
                writer.add_image(phase + "/non_zero_{}".format(i), nz_grid, epoch)
        else:
            w = weights.squeeze().t()
            nz = (w != 0).float()
            nz_grid = vutils.make_grid(nz)
            writer.add_image(phase + "/non_zero", nz_grid, epoch)

    if train:
        return results, iter
    return results


def train_val_test():
    global writer

    if not os.path.exists(FLAGS.save_dir):
        os.mkdir(FLAGS.save_dir)

    # Set data_dir.
    FLAGS.data_dir = os.environ["DATA_DIR"]

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    FLAGS.device = device

    if hasattr(FLAGS, "random_seed"):
        seed = FLAGS.random_seed
    else:
        seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print("=> loading dataset '{}'".format(FLAGS.data))

    data = getter("data")()
    train_loader = data.train_loader
    FLAGS.len_loader = len(train_loader)
    val_loader = data.val_loader

    criterion = torch.nn.CrossEntropyLoss(reduction="none").to(device)

    print("=> creating model '{}'".format(FLAGS.model))
    model = getter("model")()

    optimizer = get_optimizer(model)

    if not FLAGS.evaluate:
        model = nn.DataParallel(model)
        model = model.to(device)

    start_epoch = 0
    lr_scheduler = get_lr_scheduler(optimizer)
    best_val = 0.0
    iter = 0.0

    # optionally use the graph of another network
    if getattr(FLAGS, "use_graph", False):
        assert FLAGS.graph == "fine_tune"
        print("=> loading '{}'".format(FLAGS.use_graph))
        checkpoint = torch.load(
            FLAGS.use_graph, map_location=lambda storage, loc: storage
        )
        state_dict = checkpoint["model"]
        model.load_state_dict(state_dict)
        # make a call to get_weight -- this will initialize the masks.
        model.module.get_weight()

    # optionally use the initialization of another network
    if getattr(FLAGS, "use_init", False):
        assert hasattr(FLAGS, "use_graph")
        assert FLAGS.graph == "fine_tune"
        print("=> loading '{}'".format(FLAGS.use_init))
        checkpoint = torch.load(
            FLAGS.use_init, map_location=lambda storage, loc: storage
        )
        state_dict = checkpoint["model"]
        for k, v in model.state_dict().items():
            if k not in state_dict:
                state_dict[k] = v
                print("inserting {}".format(k))
        model.load_state_dict(state_dict)

    # optionally resume from a checkpoint
    if FLAGS.resume:
        if os.path.isfile(FLAGS.resume):
            print("=> loading checkpoint '{}'".format(FLAGS.resume))
            checkpoint = torch.load(
                FLAGS.resume, map_location=lambda storage, loc: storage
            )
            start_epoch = checkpoint["last_epoch"] + 1
            best_val = checkpoint["best_val"]
            iter = checkpoint["iter"]
            state_dict = checkpoint["model"]
            if FLAGS.evaluate:
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    FLAGS.resume, checkpoint["last_epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(FLAGS.resume))

    torch.backends.cudnn.benchmark = True

    # Logging.
    start_time = time.time()
    local_start_time_str = time.strftime(
        "%Y-%m-%d_%H:%M:%S", time.localtime(start_time)
    )
    if hasattr(FLAGS, "title"):
        title = FLAGS.title
    else:
        title = "-".join(sys.argv[-1].split(":")[-1].split("/"))

    if getattr(FLAGS, "log_dir", False):
        log_prefix = FLAGS.log_dir
    else:
        log_prefix = "./runs/"

    if getattr(FLAGS, "save_dir", False):
        checkpoint_prefix = FLAGS.save_dir
    else:
        checkpoint_prefix = "./checkpoints"

    log_dir = os.path.join(log_prefix, title + "-" + local_start_time_str)
    checkpoint_dir = os.path.join(checkpoint_prefix, title.replace("/", "_"))
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    writer = SummaryWriter(log_dir=log_dir)

    train_meters = get_meters("train", model)
    val_meters = get_meters("val", model)
    val_meters["best_val"] = ScalarMeter("best_val")

    if FLAGS.evaluate:
        model_profiling(model)
        print("Start evaluation.")
        if getattr(FLAGS, "fast_eval", False):
            model.prepare_for_fast_eval()
        model = nn.DataParallel(model)
        model = model.to(device)
        with torch.no_grad():
            results = run_one_epoch(
                0,
                val_loader,
                model,
                criterion,
                optimizer,
                val_meters,
                phase="val",
                iter=iter,
                scheduler=lr_scheduler,
            )
        return

    # save init.
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "last_epoch": 0,
            "best_val": best_val,
            "meters": (train_meters, val_meters),
            "iter": iter,
        },
        os.path.join(checkpoint_dir, "init.pt"),
    )

    print("Start training.")
    for epoch in range(start_epoch, FLAGS.num_epochs):
        if FLAGS.lr_scheduler != "cosine":
            lr_scheduler.step(epoch)
        # train
        results, iter = run_one_epoch(
            epoch,
            train_loader,
            model,
            criterion,
            optimizer,
            train_meters,
            phase="train",
            iter=iter,
            scheduler=lr_scheduler,
        )

        # val
        val_meters["best_val"].cache(best_val)
        with torch.no_grad():
            results = run_one_epoch(
                epoch,
                val_loader,
                model,
                criterion,
                optimizer,
                val_meters,
                phase="val",
                iter=iter,
                scheduler=lr_scheduler,
            )

        if results["top1_accuracy"] > best_val:
            best_val = results["top1_accuracy"]
            torch.save({"model": model.state_dict()}, os.path.join(log_dir, "best.pt"))
            print("New best validation top1 accuracy: {:.3f}".format(best_val))

        writer.add_scalar("val/best_val", best_val, epoch)

        # save latest checkpoint.
        if epoch == 0 or (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "last_epoch": epoch,
                    "best_val": best_val,
                    "meters": (train_meters, val_meters),
                    "iter": iter,
                },
                os.path.join(checkpoint_dir, "epoch_{}.pt".format(epoch)),
            )

            flops, _ = model_profiling(model.module)
            writer.add_scalar("flops/flops", flops, epoch)

    return


def main():
    """train and eval model"""
    train_val_test()


if __name__ == "__main__":
    main()
