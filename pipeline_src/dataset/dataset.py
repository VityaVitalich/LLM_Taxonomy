import pandas as pd

import torch
import torch.nn as nn

from tqdm import tqdm_notebook as tqdm

from torch.utils.data import Dataset

from dataset.prompt_schemas import hypo_term_hyper
import pandas as pd
from multiprocessing import cpu_count
from torch.utils.data import DataLoader


class HypernymDataset(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        tokenizer_encode_args={"return_tensors": "pt"},
        semeval_format=False,
        gold_path=None,
        transforms=[hypo_term_hyper],
    ):
        self.tokenizer = tokenizer
        self.transforms = transforms
        # сюда могут идти немного другие аргументы если допустим я использую Dolly а не T5
        self.tokenizer_encode_args = tokenizer_encode_args
        # в формате SemEval дебильные датасеты, мы их тут соединим
        if semeval_format:
            assert gold_path is not None
            train_data_en_data = pd.read_csv(
                data_path, header=None, sep="\t", names=["term", "relation"]
            )
            train_gold_en_data = pd.read_csv(gold_path, header=None, names=["hypernym"])

            self.df = pd.concat([train_data_en_data, train_gold_en_data], axis=1)[
                ["term", "hypernym"]
            ]
        # предположительно в нашем датасете уже все ок, но это опицональная часть
        else:
            self.df = pd.read_csv(
                data_path, header=None, sep="\t", names=["term", "hypernym"]
            )

        self.df.index = list(range(len(self.df)))

    # в данном случае выход под LM модельку с маск токеном -100
    def __getitem__(self, index):
        row = self.df.loc[index]
        term = row["term"]
        target = ", ".join(row["hypernym"].split("\t"))

        # заранее пишу более общо, чтобы мы могли разне процессинги пробовать, а в будущем рандомно выбирать и тд
        # это типа мы подаем список трансформаций затравок
        processed_term = self.transforms[0](term)

        # токенизируем
        encoded_term = self.tokenizer.encode(
            processed_term, **self.tokenizer_encode_args
        )
        encoded_target = self.tokenizer.encode(target, **self.tokenizer_encode_args)

        input_seq = torch.concat([encoded_term, encoded_target], dim=1)
        labels = input_seq.clone()
        labels[0, : encoded_term.size()[1]] = -100

        return {
            "encoded_term": encoded_term.squeeze(),  # думаю потребуется при генерации, или для сек 2 сек
            "encoded_target": encoded_target.squeeze(0),  # отдельно токены для таргета
            "input_seq": input_seq.squeeze(),  # полное предложение без масок
            "labels": labels.squeeze(),  # маскированный контекст
        }

    def __len__(self):
        return len(self.df)

    # ничего необычного, складываем, паддим


class Collator:
    def __init__(self, pad_token_id, eos_token_id, mask_token_id):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.mask_token_id = mask_token_id

    def __call__(self, batch):
        terms = []
        targets = []
        inputs = []
        labels = []

        for elem in batch:
            terms.append(elem["encoded_term"].flip(dims=[0]))
            targets.append(elem["encoded_target"])
            inputs.append(elem["input_seq"])
            labels.append(elem["labels"])

        terms = torch.nn.utils.rnn.pad_sequence(
            terms, batch_first=True, padding_value=self.pad_token_id
        ).flip(dims=[1])
        targets = torch.nn.utils.rnn.pad_sequence(
            targets, batch_first=True, padding_value=self.eos_token_id
        )
        inputs = torch.nn.utils.rnn.pad_sequence(
            inputs, batch_first=True, padding_value=self.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=self.mask_token_id
        )

        att_mask_inputs = torch.zeros_like(inputs)
        att_mask_inputs[inputs != self.pad_token_id] = 1

        att_mask_terms = torch.zeros_like(terms)
        att_mask_terms[terms != self.pad_token_id] = 1

        return (terms, att_mask_terms, targets, inputs, att_mask_inputs, labels)


def init_data(tokenizer, config, mask_label_token=-100, semeval_format=True):
    # data
    train_dataset = HypernymDataset(
        data_path=config.data_path,
        tokenizer=tokenizer,
        gold_path=config.gold_path,
        semeval_format=semeval_format,
    )
    test_dataset = HypernymDataset(
        data_path=config.test_data_path,
        tokenizer=tokenizer,
        gold_path=config.test_gold_path,
        semeval_format=semeval_format,
    )

    num_workers = cpu_count()

    collator = Collator(
        tokenizer.eos_token_id, tokenizer.eos_token_id, mask_label_token
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=collator,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )

    return train_dataset, test_dataset, train_loader, val_loader