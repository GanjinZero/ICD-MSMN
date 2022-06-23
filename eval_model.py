from main import eval_func
import sys
import os
import torch
from torch.utils.data import DataLoader
from data_util import MimicFullDataset, my_collate_fn
from evaluation import all_metrics, print_metrics
from pandas import DataFrame
import numpy as np
import copy
from find_threshold import find_threshold_micro


if __name__ == "__main__":
    device = "cuda:0"
    model_path = sys.argv[1]

    if model_path.find('mimic3-50') >= 0:
        version = 'mimic3-50'
        batch_size = 8
    else:
        version = 'mimic3'
        batch_size = 4
    print(f"Version: {version}")

    model = torch.load(model_path).to(device)

    word_embedding_path = '' # please add it by yourself

    dev_dataset = MimicFullDataset(version, "dev", word_embedding_path, 4000)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=my_collate_fn, shuffle=False, num_workers=1)
    test_dataset = MimicFullDataset(version, "test", word_embedding_path, 4000)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=my_collate_fn, shuffle=False, num_workers=1)

    dev_metric, (dev_yhat, dev_y, dev_yhat_raw), threshold = eval_func(model, dev_dataloader, device, tqdm_bar=True)
    print('Default Threshold on Dev')
    print_metrics(dev_metric, suffix="Dev")

    if isinstance(threshold, float) or (version == 'mimic3-50'):
        print('Threshold:', threshold)

    print(f'Adjust Threshold on Test')
    test_metric, _, _ = eval_func(model, test_dataloader, device, tqdm_bar=True, threshold=threshold)
    print_metrics(test_metric, suffix='Test')
