import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

from torch.nn.init import xavier_uniform_
from einops import rearrange
from torchdiffeq import odeint  # adjoint took more nfe_backward and nfe_forward
from torchnlp.word_to_vector import GloVe
from config import ex


def linear(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    if bias:
        lin.bias.data.zero_()

    return lin


def linear2(in_dim, out_dim, bias=True):
    lin = nn.Linear(in_dim, out_dim, bias=bias)
    lin2 = nn.Linear(out_dim, out_dim, bias=bias)
    xavier_uniform_(lin.weight)
    xavier_uniform_(lin2.weight)
    if bias:
        lin.bias.data.zero_()
        lin2.bias.data.zero_()

    return nn.Sequential(lin, nn.ELU(), lin2)


def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, "Length shape should be 1 dimensional."
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask


class DAFT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = linear(dim + 1, 1)
        self.different_question_projs = linear(dim + 1, dim)
        self.nfe = 0

    def set_context(self, context, question):
        self.context = context
        self.question = question

    def forward(self, t, attn_weight):
        # DOSE not use prev attn_weight
        self.nfe += 1

        tt = torch.ones_like(self.question[:, :1]) * t
        ttx = torch.cat([tt, self.question], dim=1)
        question = self.different_question_projs(ttx)

        context_prod = question.unsqueeze(1) * self.context
        context_prod = torch.cat([context_prod, attn_weight], dim=2)
        attn_weight_diff = self.attn(context_prod)

        return attn_weight_diff


class ControlUnit(nn.Module):
    def __init__(self, dim, max_step, use_daft=False):
        super().__init__()
        self.use_daft = use_daft

        if use_daft:
            self.dafts = nn.ModuleList([DAFT(dim) for _ in range(max_step)])
            self.attn_zero = nn.Parameter(torch.tensor(0.0))
        else:
            self.attn = linear(dim, 1)
            self.shared_question_proj = linear(dim, dim)
            self.different_question_projs = nn.ModuleList(
                [linear(dim, dim) for _ in range(max_step)]
            )

    @ex.capture
    def forward(
        self,
        step,
        context,
        question,
        context_lens,
        prev_attention=None,
        interpolate_num=0,
        interpolate=False,
        use_daft=False,
        solver_tol=1e-3,
    ):
        b, l, c = context.shape

        if self.use_daft:
            self.dafts[step].set_context(context, question)
            time_step = (
                torch.linspace(0, 1, 1 + interpolate_num)
                if interpolate
                else torch.linspace(0, 1, 2)
            )

            attn_zero = (
                self.attn_zero.reshape(1, 1, 1).expand(b, l, 1)
                if prev_attention is None
                else prev_attention
            )
            attn_weights = odeint(
                self.dafts[step],
                attn_zero,
                time_step.to(attn_zero),
                rtol=solver_tol,
                atol=solver_tol,
            )
            attn_weight = attn_weights[-1]
        else:
            question = self.shared_question_proj(question)
            question = torch.tanh(question)
            question = self.different_question_projs[step](question)
            context_prod = question.unsqueeze(1) * context
            attn_weight = self.attn(context_prod)
            attn_weights = [attn_weight]

        control_mask = length_to_mask(context_lens).to(attn_weight)
        control_mask = 1.0 - control_mask
        control_mask = control_mask.float().unsqueeze(2)
        attn_weight = attn_weight + (control_mask * -1e30)

        attn = F.softmax(attn_weight, dim=1)
        control = (attn * context).sum(dim=1)

        return (control, attn_weight, attn_weights)

    @property
    def nfe(self):
        nfe = 0
        if self.use_daft:
            for attn in self.dafts:
                nfe += attn.nfe
        return nfe

    @nfe.setter
    def nfe(self, value):
        if self.use_daft:
            for attn in self.dafts:
                attn.nfe = value


class ReadUnit(nn.Module):
    @ex.capture
    def __init__(self, dim, read_dropout):
        super().__init__()

        self.dropout_know = nn.Dropout(read_dropout)
        self.dropout_mem = nn.Dropout(read_dropout)
        self.proj_know = linear(dim, dim)
        self.proj_mem = linear(dim, dim)

        self.readMemProj = linear2(dim * 2, dim)
        self.inter2att = linear(dim, 1)
        self.inter_dropout = nn.Dropout(read_dropout)

    @ex.capture
    def gen_dropout_mask(self, memory, read_dropout):
        mask = torch.empty_like(memory).bernoulli_(1 - read_dropout)
        mask = mask / (1 - read_dropout)
        self.memory_dropout_mask = mask

    def forward(self, memory, _know, control, know_lens, know_max):
        if self.training:
            memory = self.memory_dropout_mask * memory

        know = self.dropout_know(_know)
        mem = self.dropout_mem(memory)

        know = rearrange(know, "b c hw -> b hw c")
        backup_know = know.detach()
        know = self.proj_know(know)
        mem = self.proj_mem(mem)

        mem = mem.unsqueeze(1)
        out = mem * know
        inter = torch.cat([out, know], dim=2)
        inter = self.readMemProj(inter)

        inter = inter * control.unsqueeze(1)
        inter = F.elu(inter)

        inter = self.inter_dropout(inter)
        attention_weight = self.inter2att(inter)

        read_mask = length_to_mask(know_lens, max_len=know_max).to(attention_weight)
        read_mask = 1.0 - read_mask
        read_mask = read_mask.float().unsqueeze(2)

        attention_weight = attention_weight + (read_mask * -1e30)

        attention = F.softmax(attention_weight, dim=1)
        info = (attention * know).sum(dim=1)

        return info, attention_weight


class WriteUnit(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.newMemory = linear(dim * 2, dim)

    def forward(self, memories, info, controls):
        next_mem = self.newMemory(torch.cat([memories[-1], info], 1))
        return next_mem


class MACUnit(nn.Module):
    def __init__(self, dim, max_step=12, interpolate_at=list(), use_daft=False):
        super().__init__()

        self.control = ControlUnit(dim, max_step, use_daft=use_daft)
        self.interpolate_at = interpolate_at
        self.read = ReadUnit(dim)
        self.write = WriteUnit(dim)

        self.mem_0 = nn.Parameter(torch.randn(1, dim))

        self.dim = dim
        self.max_step = max_step
        self.attentions = {"text": None, "image": None, "inter": None, "start": None}

    def forward(
        self,
        context,
        question,
        knowledge,
        context_lens,
        know_lens,
        know_max,
        interpolate_num=None,
    ):
        self.attentions = {"text": None, "image": None, "inter": None, "start": None}
        b_size = question.size(0)

        control = question
        memory = self.mem_0.expand(b_size, self.dim)
        self.read.gen_dropout_mask(memory)

        controls = [control]
        memories = [memory]
        spatial_attentions = list()
        text_attentions = list()
        interpolations = list()

        for i in range(self.max_step):
            control, text_attention, _inter = self.control(
                i,
                context,
                question,
                context_lens,
                prev_attention=text_attentions[-1]
                if len(text_attentions) != 0
                else None,
                interpolate_num=interpolate_num,
                interpolate=(i in self.interpolate_at),
            )
            if i in self.interpolate_at:
                interpolations.append(_inter)

            controls.append(control)
            text_attentions.append(text_attention)
            info, attention = self.read(memory, knowledge, control, know_lens, know_max)
            spatial_attentions.append(attention)
            memory = self.write(memories, info, controls)
            memories.append(memory)

        text_attentions = torch.stack(text_attentions)
        self.attentions["text"] = text_attentions
        spatial_attentions = torch.stack(spatial_attentions)
        self.attentions["image"] = spatial_attentions
        self.attentions["inter"] = interpolations

        return memory, (controls, memories)

    @property
    def nfe(self):
        return self.control.nfe

    @nfe.setter
    def nfe(self, value):
        self.control.nfe = value


class MAC(nn.Module):
    @ex.capture
    def __init__(
        self,
        n_vocab,
        dim,
        dataset_name,
        root,
        task_root,
        embed_hidden=300,
        max_step=100,
        interpolate_at=list(),
        classes=28,
        use_daft=False,
        img_enc_dropout=0.18,
        emb_dropout=0.15,
        question_dropout=0.08,
        classifier_dropout=0.15,
        qdic=None,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        if dataset_name == "clevr":
            self.conv = nn.Sequential(
                nn.Dropout(img_enc_dropout),
                nn.Conv2d(dim * 2, dim, 3, padding=1),
                nn.ELU(),
                nn.Dropout(img_enc_dropout),
                nn.Conv2d(dim, dim, 3, padding=1),
                nn.ELU(),
            )
        elif dataset_name == "gqa":
            self.conv = nn.Sequential(
                nn.Dropout(img_enc_dropout),
                nn.Conv1d(2048, dim, 1, padding=0),
                nn.ELU(),
            )
        else:
            raise KeyError(f"Dataset {dataset_name} does not exist")

        self.embed = nn.Embedding(n_vocab, embed_hidden)
        self.embed_hidden = embed_hidden
        self.embed_drop = nn.Dropout(emb_dropout)  # no var drop embed

        self.lstm = nn.LSTM(
            embed_hidden, dim // 2, batch_first=True, bidirectional=True
        )
        self.question_drop = nn.Dropout(question_dropout)

        self.mac = MACUnit(
            dim, max_step, interpolate_at=interpolate_at, use_daft=use_daft
        )

        self.question_prep = linear(dim, dim)

        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),
            linear(dim * 2, dim),
            nn.ELU(),
            nn.Dropout(classifier_dropout),
            linear(dim, classes),
        )

        self.max_step = max_step
        self.dim = dim

        nn.init.xavier_uniform_(self.conv[1].weight)
        nn.init.zeros_(self.conv[1].bias)

        if dataset_name == "clevr":
            nn.init.uniform_(self.embed.weight, -1.0, 1.0)
            nn.init.xavier_uniform_(self.conv[4].weight)
            nn.init.zeros_(self.conv[4].bias)
        elif dataset_name == "gqa":
            glove = GloVe(name="6B", cache=f"{root}/{task_root}/word_vectors_cache")
            for v, i in qdic["w2i"].items():
                self.embed.weight.data[i, :] = glove[v]
        else:
            raise KeyError(f"Dataset {dataset_name} does not exist")

    def forward(self, image, question, question_len, interpolate_num=None):
        if self.dataset_name == "clevr":
            img = self.conv(image)
            img = rearrange(img, "b c h w -> b c (h w)")
            img_len, img_max = (
                torch.tensor([img.shape[2] for _ in range(img.shape[0])]),
                img.shape[2],
            )
        elif self.dataset_name == "gqa":
            img = rearrange(image, "b d c -> b c d")
            img = F.normalize(img, dim=-1)
            img = self.conv(img)
            img_len, img_max = (image.sum(dim=-1) != 0).long().sum(dim=-1), 100

        embed = self.embed(question)
        embed = self.embed_drop(embed)

        packed_embed = nn.utils.rnn.pack_padded_sequence(
            embed, question_len, batch_first=True
        )
        lstm_out_pack, (qvec_pack, _) = self.lstm(packed_embed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out_pack, batch_first=True, total_length=question_len.max()
        )
        qvec = rearrange(qvec_pack, "dir b dim -> b (dir dim)")
        qvec = self.question_drop(qvec)

        cvec, qvec, know = lstm_out, qvec, img
        memory, (controls, memories) = self.mac(
            cvec,
            qvec,
            know,
            question_len,
            img_len,
            img_max,
            interpolate_num=interpolate_num,
        )

        self.start_cont, self.start_mem = controls[0], memories[0]
        self.end_cont, self.end_mem = controls[-1], memories[-1]

        qvec = self.question_prep(qvec)
        out = torch.cat([memory, qvec], dim=1)
        out = self.classifier(out)

        return out

    @property
    def nfe(self):
        return self.mac.nfe

    @nfe.setter
    def nfe(self, value):
        self.mac.nfe = value


class TFBCELoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        relu_logits = F.relu(logits)
        neg_abs_logits = -torch.abs(logits)

        term1 = relu_logits - logits * targets
        term2 = torch.log1p(torch.exp(neg_abs_logits))
        loss = term1 + term2
        loss = loss.sum(dim=-1).mean(dim=-1)
        return loss
