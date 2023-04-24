import torch
from typing import List, Dict, Union, NoReturn


class DataCollatorWithPadding:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            features,
            padding=True,
        )
        ybatch = self.tokenizer.pad(
            {'input_ids': batch['labels'], 'attention_mask': batch['decoder_attention_mask']},
            padding=True,
        ) 
        batch['labels'] = ybatch['input_ids']
        batch['decoder_attention_mask'] = ybatch['attention_mask']
        
        return {k: torch.tensor(v) for k, v in batch.items()}