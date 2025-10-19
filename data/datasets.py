# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

# data/datasets.py
from pathlib import Path
from collections import defaultdict
import numpy as np
from asteroid.data import LibriMix
import random
import torch
import soundfile as sf

import os
import csv
import pandas as pd
from collections import defaultdict
import random

from torch.utils.data import Dataset, DataLoader, ConcatDataset

from utils.transforms import stft_torch, istft_torch
from utils.helper import sample_mixing_ratio_by_snr_range, calc_mixing_ratio_by_signal


def get_dataloaders(config, is_ddp=False, world_size=1, rank=0):

    train_360_dataset = LibriMixInformed(
        csv_dir=config['dataset']['train_360_dir'],
        librimix_meta_dir=config['dataset']['librimix_meta_dir'],
        task=config['dataset']['task'],
        sample_rate=config['dataset']['sample_rate'],
        n_src=config['dataset']['n_src'],
        segment=config['dataset']['segment'],
        segment_aux=config['dataset']['segment_aux'],
        n_fft=config['dataset']['n_fft'],
        win_length=config['dataset']['win_length'],
        hop_length=config['dataset']['hop_length'],
        snr_range=config['dataset']['snr_range'],
    )
    train_100_dataset = LibriMixInformed(
        csv_dir=config['dataset']['train_100_dir'],
        librimix_meta_dir=config['dataset']['librimix_meta_dir'],
        task=config['dataset']['task'],
        sample_rate=config['dataset']['sample_rate'],
        n_src=config['dataset']['n_src'],
        segment=config['dataset']['segment'],
        segment_aux=config['dataset']['segment_aux'],
        n_fft=config['dataset']['n_fft'],
        win_length=config['dataset']['win_length'],
        hop_length=config['dataset']['hop_length'],
        snr_range=config['dataset']['snr_range'],
    )
    train_dataset = ConcatDataset([train_360_dataset, train_100_dataset])

    val_dataset = LibriMixInformed(
        csv_dir=config['dataset']['val_dir'],
        librimix_meta_dir=config['dataset']['librimix_meta_dir'],
        task=config['dataset']['task'],
        sample_rate=config['dataset']['sample_rate'],
        n_src=config['dataset']['n_src'],
        segment=config['dataset']['segment'],
        segment_aux=config['dataset']['segment_aux'],
        n_fft=config['dataset']['n_fft'],
        win_length=config['dataset']['win_length'],
        hop_length=config['dataset']['hop_length'],
        snr_range=config['dataset']['snr_range'],
    )

    if is_ddp:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        sampler=train_sampler,
        shuffle=True,
        num_workers=config['train']['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        sampler=val_sampler,
        shuffle=False,
        num_workers=config['train']['num_workers'],
        pin_memory=True
    )

    return train_loader, val_loader

def read_enrollment_csv(csv_path):
    data = defaultdict(dict)
    with open(csv_path, 'r') as f:
        f.readline() # csv header

        for line in f:
            mix_id, utt_id, *aux = line.strip().split(',')
            aux_it = iter(aux)
            aux = [(auxpath,int(float(length))) for auxpath, length in zip(aux_it, aux_it)]
            data[mix_id][utt_id] = aux
    return data

class LibriMixInformed(Dataset):
    def __init__(
        self, csv_dir, librimix_meta_dir, task="sep_clean", sample_rate=16000, n_src=2, 
        segment=3, segment_aux=3, 
        n_fft=512, hop_length=128, win_length=512,
        snr_range=(-15.0, 25.0),
        ):
        self.base_dataset = LibriMix(csv_dir, task, sample_rate, n_src, segment)
        self.data_aux = read_enrollment_csv(Path(csv_dir) / 'mixture2enrollment.csv')
        task_name = os.path.basename(csv_dir)
        if task_name == 'train-360':
            task = 'train-clean-360'
        elif task_name == 'train-100':
            task = 'train-clean-100'
        elif task_name == 'dev':
            task = 'dev-clean'
        elif task_name == 'test':
            task = 'test-clean'
        else:
            raise ValueError(f"Unknown task name: {task_name}")
        librimix_meta_path = os.path.join(
            librimix_meta_dir, 
            f'Libri{n_src}Mix', 
            f'libri{n_src}mix_{task}.csv',
        )
        self.librimix_meta = pd.read_csv(librimix_meta_path)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_src = n_src
        self.snr_range = snr_range

        if segment_aux is not None:
            max_len = np.sum([len(self.data_aux[m][u]) for m in self.data_aux 
                                                     for u in self.data_aux[m]])
            self.seg_len_aux = int(segment_aux * sample_rate)
            self.data_aux = {m: {u:  
                [(path,length) for path, length in self.data_aux[m][u]
                    if length >= self.seg_len_aux
                    ]
                for u in self.data_aux[m]} for m in self.data_aux}
            new_len = np.sum([len(self.data_aux[m][u]) for m in self.data_aux 
                                                     for u in self.data_aux[m]])
            print(
                f"Drop {max_len - new_len} utterances from {max_len} "
                f"(shorter than {segment_aux} seconds)"
            )
        else:
            self.seg_len_aux = None

        self.seg_len = self.base_dataset.seg_len

        # to choose pair of mixture and target speaker by index
        self.data_aux_list = [(m,u) for m in self.data_aux 
                                    for u in self.data_aux[m]]

    def __len__(self):
        return len(self.data_aux_list)

    def _get_segment_start_stop(self, seg_len, length):
        if seg_len is not None:
            start = random.randint(0, length - seg_len)
            stop = start + seg_len
        else:
            start = 0
            stop = None
        return start, stop

    def __getitem__(self, idx):
        mix_id, utt_id = self.data_aux_list[idx]
        row = self.base_dataset.df[self.base_dataset.df['mixture_ID'] == mix_id].squeeze()
        meta_row = self.librimix_meta[self.librimix_meta['mixture_ID'] == mix_id].squeeze()

        mixture_path = row['mixture_path']
        self.mixture_path = mixture_path
        tgt_spk_idx = mix_id.split('_').index(utt_id)
        self.target_speaker_idx = tgt_spk_idx

        # read mixture
        start, stop = self._get_segment_start_stop(self.seg_len, row['length'])
        mixture,_ = sf.read(mixture_path, dtype="float32", start=start, stop=stop)
        mixture = torch.from_numpy(mixture)

        # read source
        source_path = row[f'source_{tgt_spk_idx+1}_path']
        source,_ = sf.read(source_path, dtype="float32", start=start, stop=stop)
        source = torch.from_numpy(source)

        # read enrollment
        enroll_path, enroll_length = random.choice(self.data_aux[mix_id][utt_id])
        start_e, stop_e = self._get_segment_start_stop(self.seg_len_aux, enroll_length)
        enroll,_ = sf.read(enroll_path, dtype="float32", start=start_e, stop=stop_e)
        enroll = torch.from_numpy(enroll)

        # calculate background
        background = mixture - source

        # calculate mixing ratio
        all_spk_ids = [1, 2] if self.n_src == 2 else [1, 2, 3]
        all_spk_ids.remove(tgt_spk_idx + 1)
        tgt_spk_gain = meta_row[f'source_{tgt_spk_idx + 1}_gain']
        bak_spk_gain = 0
        for spk_id in all_spk_ids:
            bak_spk_gain += meta_row[f'source_{spk_id}_gain']
        # real mixing ratio that LibriMix used to generate mixture
        mixing_ratio = tgt_spk_gain / (tgt_spk_gain + bak_spk_gain)

        # rescale source and background
        source_rescaled = source / tgt_spk_gain
        background_rescaled = background / bak_spk_gain

        # mixing ratio within certain SNR range that we used to train FM model
        alpha = sample_mixing_ratio_by_snr_range(
            source=source_rescaled.unsqueeze(0), 
            background=background_rescaled.unsqueeze(0),
            snr_range=self.snr_range
        )

        return {
            'mixture_filename': os.path.basename(mixture_path),
            'utt_id': utt_id,
            'mixture': mixture, 
            'source': source, 
            'source_rescaled': source_rescaled,
            'background': background,
            'background_rescaled': background_rescaled,
            'enroll': enroll,
            'enroll_path': enroll_path,
            'mixture_spec': stft_torch(mixture, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length),
            'source_spec': stft_torch(source, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length),
            'source_rescaled_spec': stft_torch(source_rescaled, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length),
            'background_spec': stft_torch(background, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length),
            'background_rescaled_spec': stft_torch(background_rescaled, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length),
            'enroll_spec': stft_torch(enroll, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length),
            'mixing_ratio': mixing_ratio,
            'alpha': alpha,
            # 'mixture_rescaled': mixture / mixture.abs().max(),
            # 'mixture_spec_rescaled': stft_torch(mixture / mixture.abs().max(), n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length),
            'mixture_rescaled': mixture  / (tgt_spk_gain + bak_spk_gain),
            'mixture_spec_rescaled': stft_torch(mixture / (tgt_spk_gain + bak_spk_gain), n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length),
        }

    def get_infos(self):
        return self.base_dataset.get_infos()

