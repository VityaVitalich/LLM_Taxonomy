from torch.utils.data import Dataset
from .prompt_schemas import hypo_term_hyper
import torch
import pandas as pd


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
            "encoded_target": encoded_target.squeeze(),  # отдельно токены для таргета
            "input_seq": input_seq.squeeze(),  # полное предложение без масок
            "labels": labels.squeeze(),  # маскированный контекст
        }

    def __len__(self):
        return len(self.df)

    # ничего необычного, складываем, паддим
    @staticmethod
    def collate_fn(batch):
        terms = []
        targets = []
        inputs = []
        labels = []

        for elem in batch:
            terms.append(elem["encoded_term"])
            targets.append(elem["encoded_target"])
            inputs.append(elem["input_seq"])
            labels.append(elem["labels"])

        terms = torch.nn.utils.rnn.pad_sequence(terms, batch_first=True)
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)
        inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

        return (terms, targets, inputs, labels)
