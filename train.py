import torch
import torch.nn as nn
import random
import os
import numpy as np

from collections import Counter
from loguru import logger
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import ex
from data import get_dataset
from model import MAC, TFBCELoss


@ex.capture
def train(
    dataset,
    net,
    net_running,
    criterion,
    optimizer,
    epoch,
    writer,
    batch_size,
    workers,
    grad_clip,
    print_freq,
):
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=dataset.collate_data,
        pin_memory=True,
        shuffle=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    moving_acc, moving_loss = 0, 0
    net.nfe = 0
    net.train()
    net = net.to(device)
    net_running = net_running.to(device)

    for i, (image, question, q_len, answer, _) in enumerate(train_loader):
        image, question, q_len, answer = (
            image.to(device),
            question.to(device),
            q_len.to(device),
            answer.to(device),
        )

        output = net(image, question, q_len)
        if dataset.name == "gqa":
            onehot_answer = (
                torch.eye(len(dataset.adic["w2i"])).to(answer)[answer].float()
            )
            loss = criterion(output, onehot_answer)
        elif dataset.name == "clevr":
            loss = criterion(output, answer)
        else:
            raise KeyError(f"Dataset {dataset_name} does not exist")

        nfe_forward = net.nfe
        net.nfe = 0

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 8)
        optimizer.step()

        nfe_backward = net.nfe
        net.nfe = 0

        correct = output.detach().argmax(1) == answer
        correct = correct.float().sum() / answer.shape[0]

        if moving_acc == 0:
            moving_acc = correct
            moving_loss = loss.item()
        else:
            moving_acc = moving_acc * 0.99 + correct * 0.01
            moving_loss = moving_loss * 0.99 + loss.item() * 0.01

        logger.info(
            f"Epoch: {epoch}; Train [{i}/{len(train_loader)}] Loss: {moving_loss:.5f}; Acc: {moving_acc:.5f}; nfe_forward: {nfe_forward}; nfe_backward: {nfe_backward}"
        )
        accumulate(net_running, net)

        if i % print_freq == 0:
            step = epoch * len(train_loader) + i
            for j, param_group in enumerate(optimizer.param_groups):
                writer.add_scalar(
                    f"train/optim/learning_rate_group{j}", param_group["lr"], step
                )
            writer.add_scalar("train/loss", moving_loss, step)
            writer.add_scalar("train/acc", moving_acc, step)
            writer.add_scalar("train/nfe_forward", nfe_forward, step)
            writer.add_scalar("train/nfe_backward", nfe_backward, step)
            writer.add_histogram("train/mem_start", net.start_mem, step)
            writer.add_histogram("train/control_start", net.start_cont, step)
            writer.add_histogram("train/mem_end", net.end_mem, step)
            writer.add_histogram("train/control_end", net.end_cont, step)

    return moving_acc, moving_loss


@ex.capture
def valid(dataset, net_running, epoch, writer, val_batch_size, workers, print_freq):
    val_loader = DataLoader(
        dataset,
        batch_size=val_batch_size,
        num_workers=workers,
        collate_fn=dataset.collate_data,
        pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net_running.eval()
    net_running = net_running.to(device)
    net_running.computation_direction = "forward"
    family_correct = Counter()
    family_total = Counter()

    with torch.no_grad():
        for i, (image, question, q_len, answer, family) in enumerate(val_loader):
            image, question, q_len = (
                image.to(device),
                question.to(device),
                q_len.to(device),
            )
            output = net_running(image, question, q_len)
            correct = output.detach().argmax(1) == answer.to(device)
            for c, fam in zip(correct, family):
                if c:
                    family_correct[fam] += 1
                family_total[fam] += 1

            logger.info(f"Epoch: {epoch}; Validate [{i}/{len(val_loader)}]")

    for k, v in family_total.items():
        logger.info(f"{k}: {family_correct[k] / v:.5f}")

    writer.add_scalar(
        "valid/acc", sum(family_correct.values()) / sum(family_total.values()), epoch
    )

    logger.info(
        f"Avg Acc: {sum(family_correct.values()) / sum(family_total.values()):.5f}"
    )

    return sum(family_correct.values()) / sum(family_total.values())


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


@ex.automain
def main(learning_rate, use_daft, dataset_name, epochs, _seed, _config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_name = f"{'daftmac' if use_daft else 'mac'}_{_config['dataset_name']}_step{_config['max_step']}_{_seed}"
    os.makedirs("result/log", exist_ok=True)
    logger.add(f"result/log/{task_name}.txt")

    logger.info(f"Making Code Deterministic with seed {_seed}")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(_seed)
    np.random.seed(_seed)  # numpy cpu
    random.seed(_seed)  # python cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed(_seed)
        torch.cuda.manual_seed_all(_seed)

    train_dataset = get_dataset("train")
    val_dataset = get_dataset("val")

    n_words = len(train_dataset.qdic["w2i"])
    n_answers = len(train_dataset.adic["w2i"])

    net = MAC(n_words, classes=n_answers, use_daft=use_daft, qdic=train_dataset.qdic)
    net_running = MAC(
        n_words, classes=n_answers, use_daft=use_daft, qdic=train_dataset.qdic
    )
    accumulate(net_running, net, 0)

    if dataset_name == "clevr":
        criterion = nn.CrossEntropyLoss()
    elif dataset_name == "gqa":
        criterion = TFBCELoss(train_dataset.pos_weight.to(device))
    else:
        raise KeyError(f"Dataset {dataset_name} does not exist")
    writer = SummaryWriter(f"result/summary/{task_name}")
    optimizer = torch.optim.Adam(net.parameters(), learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, mode="max")

    for epoch in range(epochs):
        train_acc, train_loss = train(
            train_dataset, net, net_running, criterion, optimizer, epoch, writer
        )

        val_acc = valid(val_dataset, net_running, epoch, writer)
        scheduler.step(val_acc)
        os.makedirs(f"result/model/{task_name}", exist_ok=True)
        torch.save(
            net_running.state_dict(),
            f"result/model/{task_name}/checkpoint_{epoch:02}.model",
        )

        if optimizer.param_groups[0]["lr"] < 1e-7:
            break
