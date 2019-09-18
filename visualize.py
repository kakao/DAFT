import torch
import glob
import ipdb
import imageio
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import json
import h5py
import torch.nn.functional as F
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from model import MAC
from data import get_dataset
from evaluation import get_tlts
from config import ex
from loguru import logger
from einops import rearrange
from tqdm import tqdm

mpl.use("agg")
sns.set()

nice_fonts = {
    # Use LaTeX for writing all texts
    "text.usetex": True,
    "font.family": "serif",
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
}

mpl.rcParams.update(nice_fonts)

image_dims = (14, 14)
cdict = dict(
    {
        "red": ((0.0, 0.0, 0.0), (0.6, 0.8, 0.8), (1.0, 1, 1)),
        "green": ((0.0, 0.0, 0.0), (0.6, 0.8, 0.8), (1.0, 1, 1)),
        "blue": ((0.0, 0.0, 0.0), (0.6, 0.8, 0.8), (1.0, 1, 1)),
        "alpha": ((0.0, 0.35, 0.35), (1.0, 0.65, 0.65)),
    }
)
plt.register_cmap(name="custom", data=cdict)


@ex.capture
def image_path(idx, root, task_root, dataset_name, visualization_split):
    if dataset_name == "clevr":
        image_dir = f"{root}/{task_root}/images/{visualization_split}"
        path = f"{image_dir}/CLEVR_{visualization_split}_{idx:06d}.png"
    elif dataset_name == "gqa":
        image_dir = f"{root}/{task_root}/images"
        path = f"{image_dir}/{idx}.jpg"
    return path


@ex.capture
def draw_attentions(
    attentions, question, answer, image_id, vis_dir, dataset, dataset_name
):
    text = F.softmax(attentions["text"].cpu().squeeze(3), dim=2)
    # text = attentions["text"].cpu().squeeze(3)  # for logit visualization
    text = rearrange(text, "steps b length -> steps (b length)", b=1)
    image = F.softmax(attentions["image"].cpu().squeeze(3), dim=2)
    if dataset_name == "clevr":
        image = rearrange(image, "steps b (h w) -> steps (b h) w", b=1, h=14, w=14)
    elif dataset_name == "gqa":
        info = dataset.object_info[image_id[0]]
        image = rearrange(image, "steps b length -> steps (b length)", b=1, length=100)
    path = image_path(image_id[0])

    # draw question and its attention
    steps = range(1, len(text) + 1)
    tokens = question.split()
    table = text[:, : (len(tokens) + 1)].permute(1, 0).numpy()

    df = pd.DataFrame(data=table, index=tokens, columns=steps)

    heatmap = sns.heatmap(
        df,
        annot=True,
        annot_kws={"size": 4},
        fmt=".2f",
        cmap="Purples",
        cbar=False,
        linewidths=0.5,
        linecolor="gray",
        square=True,
        robust=False,
    )

    heatmap.xaxis.tick_top()
    heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=0)
    fig = heatmap.get_figure()
    fig.savefig(
        f"{vis_dir}/text.pdf",
        dpi=480,
        bbox_inches="tight",
        format="pdf",
        pad_inches=0.0,
    )
    plt.close(fig)

    # draw image and its attention
    img = imageio.imread(path)

    if dataset_name == "gqa":
        bbox = torch.load(
            f"{'/'.join(path.split('/')[:-2])}/features/{image_id[0]}_bbox.pth"
        )

    for i, ia in enumerate(image):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        x, y = torch.arange(-1.5, 1.5, 0.05), torch.arange(-1.0, 1.0, 0.05)
        extent = x.min().item(), x.max().item(), y.min().item(), y.max().item()
        ax.imshow(img, interpolation="nearest", extent=extent)

        if dataset_name == "clevr":
            ax.imshow(
                ia.reshape(image_dims).numpy(),
                cmap=plt.get_cmap("custom"),
                interpolation="bicubic",
                extent=extent,
            )
        elif dataset_name == "gqa":
            canvas = torch.zeros(img.shape[0], img.shape[1])  # H x W
            for att, box in zip(ia, bbox):
                x0, y0, x1, y1 = box
                canvas[int(y0) : int(y1), int(x0) : int(x1)] += att
            ax.imshow(
                canvas.numpy(),
                cmap=plt.get_cmap("custom"),
                interpolation="nearest",
                extent=extent,
            )

        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)

        file_name = f"{vis_dir}/image_step{i}.png"
        fig.savefig(file_name, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)


@ex.automain
def main(
    dataset_name, root, task_root, visualization_split, use_daft, load_seed, _config
):
    assert dataset_name in ["clevr", "gqa"]
    assert visualization_split in ["train", "val"]

    task_name = f"{'daftmac' if use_daft else 'mac'}_{_config['dataset_name']}_step{_config['max_step']}_{load_seed}"
    logger.add(f"result/log/{task_name}_vis.txt")
    image_dir = f"{root}/{task_root}/images"
    os.makedirs(f"result/vis/{task_name}", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = get_dataset(visualization_split)
    n_words = len(dataset.qdic["w2i"])
    n_answers = len(dataset.adic["w2i"])

    net = MAC(n_words, classes=n_answers, use_daft=use_daft, qdic=dataset.qdic)

    list_of_weights = glob.glob(f"result/model/{task_name}/*.model")
    weight_path = max(list_of_weights, key=os.path.getctime)
    weight = torch.load(weight_path)
    net.load_state_dict(weight)
    net.eval()
    net = net.to(device)

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=dataset.collate_data_vis,
        pin_memory=True,
    )

    for i, (image, question, q_len, answer, dataset_id, image_id) in enumerate(
        tqdm(loader)
    ):
        original_question = " ".join(
            [dataset.qdic["i2w"][e.item()] for e in question[0]]
        )
        original_answer = dataset.adic["i2w"][answer[0].item()]

        image, question, q_len = image.to(device), question.to(device), q_len.to(device)
        with torch.no_grad():
            output = net(image, question, q_len)
        pred = output.detach().argmax(dim=1)
        is_right = (pred.cpu() == answer).item() == 1

        logger.info(
            f'Visualizing Question "{original_question}" -> "{original_answer}", [{i}/{len(loader)}] image {image_id}'
        )
        tlts, lts = dict(), dict()

        for modal in ["text", "image"]:
            attentions = net.mac.attentions[modal].cpu().squeeze(3)
            tlts[modal], lts[modal] = get_tlts(
                attentions, modal, dataset_name, q_len, image
            )

        vis_dir = f"result/vis/{task_name}/{original_question}->{original_answer}"
        os.makedirs(f"{vis_dir}", exist_ok=True)
        with open(f"{vis_dir}/tlt.txt", "w") as fp:
            json.dump(
                {
                    "text_lts": lts["text"].tolist(),
                    "text_tlt": tlts["text"].tolist(),
                    "image_lts": lts["image"].tolist(),
                    "image_tlt": tlts["image"].tolist(),
                },
                fp,
                indent=2,
            )

        attentions = net.mac.attentions
        draw_attentions(
            attentions, original_question, original_answer, image_id, vis_dir, dataset
        )
