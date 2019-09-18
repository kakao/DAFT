import torch
import msgpack
import json
import numpy as np
import re

from collections import Counter
from torch.utils.data import Dataset
from config import ex


@ex.capture
def get_dataset(split, dataset_name):
    if dataset_name == "clevr":
        dataset = CLEVR(split=split)
    elif dataset_name == "gqa":
        dataset = GQA(split=split)
    else:
        raise KeyError(f"Dataset {dataset_name} does not exist")
    return dataset


class CLEVR(Dataset):
    @ex.capture
    def __init__(self, root, task_root, debug, split="train", data_fraction=1.0):
        self.root = f"{root}/{task_root}"
        self.split = split
        self.name = "clevr"

        with open(f"{self.root}/{split}.msgpack", "rb") as fp:
            self.data = msgpack.load(fp, encoding="utf8")
        if split == "val":
            self.process_family()

        if split == "train":
            partial = int(len(self.data) * data_fraction)
            self.data = self.data[:partial]

        with open(f"{self.root}/dictionary.json", "r") as fp:
            dic = json.load(fp)
            self.qdic = dic["question"]
            self.adic = dic["answer"]

        if debug:
            self.data = self.data[:1000]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        imgid, question, answer, family = (
            data["imageId"],
            data["questionSeq"],
            data["answer"],
            data["program"][-1]["function"],
        )
        question = [self.qdic["w2i"][t] for t in question]
        answer = self.adic["w2i"][answer]
        family = 0 if self.split != "val" else self.family_dict[family]

        img_path = f"{self.root}/features/{self.split}/{imgid}.pth"
        img = torch.load(img_path)
        return img, question, len(question), answer, family, index, imgid

    def process_family(self):
        self.family_dict = {
            "count": 0,
            "exist": 1,
            "query_color": 3,
            "query_size": 3,
            "query_material": 3,
            "query_shape": 3,
            "equal_material": 4,
            "equal_shape": 4,
            "equal_size": 4,
            "equal_color": 4,
            "less_than": 2,
            "greater_than": 2,
            "equal_integer": 2,
        }

    def collate_data(self, batch):
        images, lengths, answers, families = list(), list(), list(), list()
        batch_size = len(batch)

        max_len = max(map(lambda x: len(x[1]), batch))
        questions = np.zeros((batch_size, max_len), dtype=np.int64)
        sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

        for i, b in enumerate(sort_by_len):
            image, question, length, answer, family, index, imgid = b
            images.append(image)
            length = len(question)
            questions[i, :length] = question
            lengths.append(length)
            answers.append(answer)
            families.append(family)

        return (
            torch.stack(images),
            torch.from_numpy(questions),
            torch.LongTensor(lengths),
            torch.LongTensor(answers),
            families,
        )

    def collate_data_vis(self, batch):
        images, lengths, answers, dataset_ids, image_ids = (
            list(),
            list(),
            list(),
            list(),
            list(),
        )
        batch_size = len(batch)

        max_len = max(map(lambda x: len(x[1]), batch))
        questions = np.zeros((batch_size, max_len), dtype=np.int64)
        sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

        for i, b in enumerate(sort_by_len):
            image, question, length, answer, family, index, imgid = b
            images.append(image)
            length = len(question)
            questions[i, :length] = question
            lengths.append(length)
            answers.append(answer)
            dataset_ids.append(index)
            image_ids.append(imgid)

        return (
            torch.stack(images),
            torch.from_numpy(questions),
            torch.LongTensor(lengths),
            torch.LongTensor(answers),
            dataset_ids,
            image_ids,
        )


class GQA(Dataset):
    @ex.capture
    def __init__(self, root, task_root, debug, split="train", data_fraction=1.0):
        self.processor = Preprocesser()
        self.root = f"{root}/{task_root}"
        self.split = split
        self.name = "gqa"

        with open(f"{self.root}/gqa_objects_info.json", "r") as f:
            self.object_info = json.load(f)

        if split == "train":
            data = list()
            with open(f"{self.root}/balanced_train_data.json", "r") as f:
                data += json.load(f, encoding="utf8")["questions"]
            with open(f"{self.root}/balanced_val_data.json", "r") as f:
                data += json.load(f, encoding="utf8")["questions"]
            self.data = data
            partial = int(len(self.data) * data_fraction)
            self.data = self.data[:partial]
        else:
            with open(f"{self.root}/balanced_testdev_data.json", "r") as f:
                self.data = json.load(f, encoding="utf8")["questions"]

        with open(f"{self.root}/newqVocabFile.json", "r") as f:
            qdic = json.load(f)
            self.qdic = dict(
                {
                    "w2i": {v: i for i, v in enumerate(qdic)},
                    "i2w": {i: v for i, v in enumerate(qdic)},
                }
            )

        with open(f"{self.root}/newaVocabFile.json", "r") as f:
            adic = json.load(f)
            self.adic = dict(
                {
                    "w2i": {v: i for i, v in enumerate(adic)},
                    "i2w": {i: v for i, v in enumerate(adic)},
                }
            )

        if debug:
            self.data = self.data[:1000]

        self.pos_weight = self.get_weight()

    def __len__(self):
        return len(self.data)

    def get_weight(self):
        answers = [
            self.adic["w2i"][data["answer"]]
            if data["answer"] in self.adic["w2i"]
            else 1
            for data in self.data
        ]
        answers_cnt = Counter(answers)
        pos_weights = list()
        for i in range(len(self.adic["w2i"])):
            pos = answers_cnt[i]
            neg = len(answers) - pos
            if pos != 0:
                pos_weights.append(neg / pos)
            else:
                pos_weights.append(0)
        return torch.tensor(pos_weights)

    def __getitem__(self, index):
        data = self.data[index]
        imgid, question, answer = (data["imageId"], data["question"], data["answer"])
        question = self.processor.tokenize(question)

        question = [
            self.qdic["w2i"][t] if t in self.qdic["w2i"] else 1 for t in question
        ]
        answer = self.adic["w2i"][answer] if answer in self.adic["w2i"] else 1

        info = self.object_info[imgid]

        img_path = f"{self.root}/features/{imgid}_feature.pth"
        img = torch.load(img_path)

        return img, question, len(question), answer, 0, index, imgid

    def collate_data(self, batch):
        images, lengths, answers, families = list(), list(), list(), list()
        batch_size = len(batch)

        max_len = max(map(lambda x: len(x[1]), batch))
        questions = np.zeros((batch_size, max_len), dtype=np.int64)
        sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

        for i, b in enumerate(sort_by_len):
            image, question, length, answer, family, index, imgid = b
            images.append(image)
            length = len(question)
            questions[i, :length] = question
            lengths.append(length)
            answers.append(answer)
            families.append(family)

        return (
            torch.stack(images),
            torch.from_numpy(questions),
            torch.LongTensor(lengths),
            torch.LongTensor(answers),
            families,
        )

    def collate_data_vis(self, batch):
        images, lengths, answers, dataset_ids, image_ids = (
            list(),
            list(),
            list(),
            list(),
            list(),
        )
        batch_size = len(batch)

        max_len = max(map(lambda x: len(x[1]), batch))
        questions = np.zeros((batch_size, max_len), dtype=np.int64)
        sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

        for i, b in enumerate(sort_by_len):
            image, question, length, answer, family, index, imgid = b
            images.append(image)
            length = len(question)
            questions[i, :length] = question
            lengths.append(length)
            answers.append(answer)
            dataset_ids.append(index)
            image_ids.append(imgid)

        return (
            torch.stack(images),
            torch.from_numpy(questions),
            torch.LongTensor(lengths),
            torch.LongTensor(answers),
            dataset_ids,
            image_ids,
        )


class Preprocesser(object):
    # sentence tokenizer
    allPunct = ["?", "!", "\\", "/", ")", "(", ".", ",", ";", ":"]
    fullPunct = [
        ";",
        r"/",
        "[",
        "]",
        '"',
        "{",
        "}",
        "(",
        ")",
        "=",
        "+",
        "\\",
        "_",
        "-",
        ">",
        "<",
        "@",
        "`",
        ",",
        "?",
        "!",
        "%",
        "^",
        "&",
        "*",
        "~",
        "#",
        "$",
    ]
    contractions = {
        "aint": "ain't",
        "arent": "aren't",
        "cant": "can't",
        "couldve": "could've",
        "couldnt": "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        "didnt": "didn't",
        "doesnt": "doesn't",
        "dont": "don't",
        "hadnt": "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        "hasnt": "hasn't",
        "havent": "haven't",
        "hed": "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        "hes": "he's",
        "howd": "how'd",
        "howll": "how'll",
        "hows": "how's",
        "Id've": "I'd've",
        "I'dve": "I'd've",
        "Im": "I'm",
        "Ive": "I've",
        "isnt": "isn't",
        "itd": "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        "itll": "it'll",
        "let's": "let's",
        "maam": "ma'am",
        "mightnt": "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        "mightve": "might've",
        "mustnt": "mustn't",
        "mustve": "must've",
        "neednt": "needn't",
        "notve": "not've",
        "oclock": "o'clock",
        "oughtnt": "oughtn't",
        "ow's'at": "'ow's'at",
        "'ows'at": "'ow's'at",
        "'ow'sat": "'ow's'at",
        "shant": "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "she's": "she's",
        "shouldve": "should've",
        "shouldnt": "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": "somebodyd",
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        "somebodyll": "somebody'll",
        "somebodys": "somebody's",
        "someoned": "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        "someonell": "someone'll",
        "someones": "someone's",
        "somethingd": "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        "somethingll": "something'll",
        "thats": "that's",
        "thered": "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        "therere": "there're",
        "theres": "there's",
        "theyd": "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        "theyll": "they'll",
        "theyre": "they're",
        "theyve": "they've",
        "twas": "'twas",
        "wasnt": "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        "weve": "we've",
        "werent": "weren't",
        "whatll": "what'll",
        "whatre": "what're",
        "whats": "what's",
        "whatve": "what've",
        "whens": "when's",
        "whered": "where'd",
        "wheres": "where's",
        "whereve": "where've",
        "whod": "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        "wholl": "who'll",
        "whos": "who's",
        "whove": "who've",
        "whyll": "why'll",
        "whyre": "why're",
        "whys": "why's",
        "wont": "won't",
        "wouldve": "would've",
        "wouldnt": "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        "yall": "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        "youd": "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        "youll": "you'll",
        "youre": "you're",
        "youve": "you've",
    }
    nums = {
        "none": "0",
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    articles = {"a": "", "an": "", "the": ""}
    allReplaceQ = {}
    for replace in [contractions, nums, articles]:  # ,
        allReplaceQ.update(replace)

    allReplaceA = {}
    for replace in [contractions, nums]:  # ,
        allReplaceA.update(replace)

    periodStrip = lambda self, s: re.sub(r"(?!<=\d)(\.)(?!\d)", " ", s)  # :,' ?
    collonStrip = lambda self, s: re.sub(
        r"(?!<=\d)(:)(?!\d)", " ", s
    )  # replace with " " or ""?
    commaNumStrip = lambda self, s: re.sub(r"(\d)(\,)(\d)", r"\1\3", s)

    vqaProcessText = lambda self, text, tokenize, question: self.processText(
        text,
        ignoredPunct=list(),
        keptPunct=list(),
        endPunct=list(),
        delimPunct=self.fullPunct,
        replacelistPost=self.allReplaceQ if question else self.allReplaceA,
        reClean=True,
        tokenize=tokenize,
    )

    def processText(
        self,
        text,
        ignoredPunct=["?", "!", "\\", "/", ")", "("],
        keptPunct=[".", ",", ";", ":"],
        endPunct=[">", "<", ":"],
        delimPunct=list(),
        delim=" ",
        clean=False,
        replacelistPre=dict(),
        replacelistPost=dict(),
        reClean=False,
        tokenize=True,
    ):

        if reClean:
            text = self.periodStrip(text)
            text = self.collonStrip(text)
            text = self.commaNumStrip(text)

        if clean:
            for word in replacelistPre:
                origText = text
                text = text.replace(word, replacelistPre[word])
                if origText != text:
                    print(origText)
                    print(text)
                    print("")

            for punct in endPunct:
                if text[-1] == punct:
                    print(text)
                    text = text[:-1]
                    print(text)
                    print("")

        for punct in keptPunct:
            text = text.replace(punct, delim + punct + delim)

        for punct in ignoredPunct:
            text = text.replace(punct, "")

        for punct in delimPunct:
            text = text.replace(punct, delim)

        text = text.lower()

        ret = text.split()  # delim

        ret = [replacelistPost.get(word, word) for word in ret]

        ret = [t for t in ret if t != ""]
        if not tokenize:
            ret = delim.join(ret)

        return ret

    def tokenize(self, questionStr):
        question = self.vqaProcessText(questionStr, True, True)
        return question
