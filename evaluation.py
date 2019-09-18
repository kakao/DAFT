import torch
import os
import glob
import time
import ipdb

from loguru import logger
from model import MAC, length_to_mask
from config import ex
from data import get_dataset
from torch.utils.data import DataLoader
from collections import Counter


def get_tlts(attentions, modal, dataset_name, q_len, image):
    start_att = torch.zeros(1, attentions.shape[1], attentions.shape[2])

    if modal == "text":
        mask = length_to_mask(q_len).to(start_att)
    elif modal == "image":
        if dataset_name == "clevr":
            img_len, img_max = (torch.tensor([196 for _ in range(image.shape[0])]), 196)
        elif dataset_name == "gqa":
            img_len, img_max = ((image.sum(dim=-1) != 0).long().sum(dim=-1), 100)
        else:
            raise KeyError(f"Dataset {dataset_name} does not exist")
        mask = length_to_mask(img_len, max_len=img_max).to(start_att)

    mask = (1.0 - mask).float().unsqueeze(0)
    start_att = start_att + (mask * -1e30)

    attentions = torch.nn.Softmax(dim=2)(attentions)
    attentions = attentions.clamp(min=1e-10, max=1 - 1e-10)

    former, latter = attentions[1:, :, :], attentions[:-1, :, :]
    mid = (former + latter) / 2
    kl_former_mid = (former * torch.log(former / mid)).sum(dim=2)
    kl_latter_mid = (latter * torch.log(latter / mid)).sum(dim=2)
    js = (kl_former_mid + kl_latter_mid) / 2
    normalized_js = js / torch.log(torch.tensor(2.0))
    tlts = normalized_js.sum(dim=0)

    return tlts, normalized_js


@ex.capture
def evaluate(dataset, net, val_batch_size, workers, print_freq, dataset_name):
    val_loader = DataLoader(
        dataset,
        batch_size=val_batch_size,
        num_workers=workers,
        collate_fn=dataset.collate_data,
        pin_memory=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.eval()
    net = net.to(device)

    family_correct, family_total = Counter(), Counter()
    family_text_tlt, family_image_tlt = Counter(), Counter()
    times = list()

    for i, (image, question, q_len, answer, family) in enumerate(val_loader):
        image, question, q_len = (
            image.to(device),
            question.to(device),
            q_len.to(device),
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            output = net(image, question, q_len)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
        times.append(t1 - t0)

        correct = output.detach().argmax(1) == answer.to(device)

        for modal in ["text", "image"]:
            attentions = net.mac.attentions[modal].cpu().squeeze(3)
            tlts, lts = get_tlts(attentions, modal, dataset_name, q_len, image)

            for tlt, fam in zip(tlts, family):
                if modal == "text":
                    family_text_tlt[fam] += tlt.item()
                elif modal == "image":
                    family_image_tlt[fam] += tlt.item()

        for c, fam in zip(correct, family):
            if c:
                family_correct[fam] += 1
            family_total[fam] += 1

        if i % print_freq == 0:
            for k, v in family_total.items():
                logger.info(
                    f"[{i} / {len(val_loader)}] Family {k} | Accuracy: {family_correct[k] / v:.5f} | Text TLT {family_text_tlt[k] / v:.5f} | Image TLT {family_image_tlt[k] / v:.5f}"
                )

            logger.info(
                f"[{i} / {len(val_loader)}] Avg Acc: {sum(family_correct.values()) / sum(family_total.values()):.5f} | Avg Text TLT: {sum(family_text_tlt.values()) / sum(family_total.values()):.5f} | Avg Image TLT: {sum(family_image_tlt.values()) / sum(family_total.values()):.5f} | Avg Time: {torch.tensor(times[-1000:]).mean().item():.5f}"
            )

    logger.info("DONE!")

    for k, v in family_total.items():
        logger.info(
            f"[{i} / {len(val_loader)}] Family {k} | Accuracy: {family_correct[k] / v:.5f} | Text TLT {family_text_tlt[k] / v:.5f} | Image TLT {family_image_tlt[k] / v:.5f}"
        )

    logger.info(
        f"Avg Acc: {sum(family_correct.values()) / sum(family_total.values()):.5f} | Avg Text TLT: {sum(family_text_tlt.values()) / sum(family_total.values()):.5f} | Avg Image TLT: {sum(family_image_tlt.values()) / sum(family_total.values()):.5f} | Avg Batch {val_batch_size} Time: {torch.tensor(times[-1000:]).mean().item():.5f}"
    )

    return sum(family_correct.values()) / sum(family_total.values())


@ex.automain
def main(use_daft, load_seed, _config):
    assert load_seed is not None
    task_name = f"{'daftmac' if use_daft else 'mac'}_{_config['dataset_name']}_step{_config['max_step']}_{load_seed}"

    os.makedirs("result/log", exist_ok=True)
    logger.add(f"result/log/{task_name}_eval.txt")

    val_dataset = get_dataset("val")
    n_words = len(val_dataset.qdic["w2i"])
    n_answers = len(val_dataset.adic["w2i"])

    net = MAC(n_words, classes=n_answers, use_daft=use_daft, qdic=val_dataset.qdic)

    list_of_weights = glob.glob(f"result/model/{task_name}/*.model")
    weight_path = max(list_of_weights, key=os.path.getctime)
    weight = torch.load(weight_path)
    net.load_state_dict(weight)

    eval_result = evaluate(val_dataset, net)
