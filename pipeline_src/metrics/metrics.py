import numpy as np
from .calculator import MeanReciprocalRank, PrecisionAtK, MeanAveragePrecision
from typing import List


class Metric:
    def __init__(self, golds: List[str], preds: List[List[str]]):
        self.golds = golds
        self.preds = preds

    @staticmethod
    def get_hypernyms(line):
        clean_line = line.strip().replace("\n", ",").replace("-", " ").split(",")

        res = []
        for hyp in clean_line:
            if not hyp in ("", " ", ", ", ","):
                res.append(hyp.lower().strip())

        return res

    def get_metrics(self, scores=None, limit=15, return_raw=False):
        if not scores:
            scores = self.default_metrics()

        all_scores = {str(score): [] for score in scores}

        for goldline, pred_options in zip(self.golds, self.preds):
            one_line_metrics = {str(score): [] for score in scores}

            for predline in pred_options:
                one_option_metrics = self.get_one_prediction(
                    goldline, predline, scores, limit
                )

                for score in scores:
                    one_line_metrics[str(score)].append(one_option_metrics[str(score)])

            for key in all_scores:
                max_line_value = np.mean(one_line_metrics[key])  # mean to max
                all_scores[key].append(max_line_value)

        res = {}
        for key in all_scores:
            mean_value = np.mean(all_scores[key])
            res[key] = mean_value

        return all_scores if return_raw else res

    def default_metrics(self):
        scores = [
            MeanReciprocalRank(),
            MeanAveragePrecision(),
            PrecisionAtK(1),
            PrecisionAtK(3),
            PrecisionAtK(5),
            PrecisionAtK(15),
        ]
        return scores

    def get_one_prediction(self, goldline, predline, scores, limit):
        gold_hyps = self.get_hypernyms(goldline)
        pred_hyps = self.get_hypernyms(predline)
        gold_hyps_n = len(gold_hyps)
        r = [0 for i in range(limit)]

        for j in range(min(len(pred_hyps), limit)):
            pred_hyp = pred_hyps[j]
            if pred_hyp in gold_hyps:
                r[j] = 1

        res = {}
        for score in scores:
            res[str(score)] = score(r, gold_hyps_n)

        return res
