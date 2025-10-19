import argparse
import os
import random
import torch
import torchaudio
import pytorch_lightning as pl
import pandas as pd
from train import LightningModule as FlowTSELightningModule, parse_config as parse_flowtse_config
from models.t_predicter import TPredicter
from data.datasets import LibriMixInformed
from torch.utils.data import Dataset, DataLoader
from asteroid.metrics import get_metrics
from tqdm import tqdm
from flow_matching.solver.ode_solver import ODESolver
from flow_matching.path import CondOTProbPath
from utils.transforms import istft_torch


def parse_args():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--config', default=None, help='Path to the config file.')
    parser.add_argument('--t_predicter', type=str, choices=['ECAPAMLP', 'GT', 'RAND', 'ONE', 'ZERO'], default='GT', help='Type of t_predicter to use.')
    args = parser.parse_args()
    return args

def scale_audio(audio):
    max_val = torch.max(torch.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val
    return audio

def calculate_metrics(mixture, reference, estimation):
    metrics = get_metrics(mixture, reference, estimation, sample_rate=16000, metrics_list=["si_sdr", "pesq", "stoi"], ignore_metrics_errors=True)
    return metrics

def pad_and_reshape(tensor, multiple):
    """
    Pads the tensor along the last dimension to make its length a multiple of `multiple`
    and reshapes it into (n*k, d, multiple) using torch.chunk.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (n, d, l).
        multiple (int): The multiple to pad the length to.
    
    Returns:
        reshaped_tensor (torch.Tensor): Reshaped tensor of shape (n*k, d, multiple).
        original_length (int): Original length of the last dimension before padding.
    """
    n, d, l = tensor.shape
    padding_length = (multiple - (l % multiple)) % multiple
    padded_tensor = torch.nn.functional.pad(tensor, (0, padding_length))
    # Split the last dimension into chunks of size `multiple`
    reshaped_tensor = torch.cat(torch.chunk(padded_tensor, padded_tensor.shape[-1] // multiple, dim=-1), dim=0)
    return reshaped_tensor, l

def reshape_and_remove_padding(tensor, original_length):
    """
    Reshapes the tensor back to its original shape and removes the padding.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (n*k, d, multiple).
        original_length (int): Original length of the last dimension before padding.
    
    Returns:
        original_tensor (torch.Tensor): Tensor reshaped back to (n, d, original_length).
    """
    n_k, d, multiple = tensor.shape
    n = original_length // multiple + (1 if original_length % multiple != 0 else 0)
    # Combine chunks back into the original shape
    reshaped_tensor = torch.cat(torch.chunk(tensor, n, dim=0), dim=-1)
    original_tensor = reshaped_tensor[:, :, :original_length]
    return original_tensor

def generate_samples_from_testset(config_path, lightning_module_class, output_dir, predicter_type, error_range=[-0.0, 0.0], save_audio=False):
    config = parse_flowtse_config(config_path)
    model_name = 'FlowTSE'
    pl.seed_everything(config['seed'])
    checkpoint_path = config['eval']['checkpoint']
    print(f'Loading model from {checkpoint_path}')
    model = lightning_module_class.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()
    model = model.cuda() if torch.cuda.is_available() else model

    if predicter_type == "ECAPAMLP":
        t_predicter_checkpoint = config['eval']['t_predicter']
        print(f'Loading t_predicter model from {t_predicter_checkpoint}')
        t_predicter = TPredicter(**config['t_predicter'])
        t_predicter.eval()
        t_predicter = t_predicter.cuda() if torch.cuda.is_available() else t_predicter
        config['eval']['t_predicter'] = t_predicter

    import pdb; pdb.set_trace()
    test_dataset = LibriMixInformed(
        csv_dir=config['dataset']['test_dir'],
        librimix_meta_dir=config['dataset']['librimix_meta_dir'],
        task=config['dataset']['task'],
        sample_rate=config['dataset']['sample_rate'],
        n_src=config['dataset']['n_src'],
        n_fft=config['dataset']['n_fft'],
        hop_length=config['dataset']['hop_length'],
        win_length=config['dataset']['win_length'],
        segment=None,
        segment_aux=3,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['train']['num_workers'],
        pin_memory=True
    )
    os.makedirs(output_dir, exist_ok=True)
    results = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc=f"Generating {model_name} samples")):
            mixture = batch['mixture_spec'].cuda() if torch.cuda.is_available() else batch['mixture_spec']
            enrollment = batch['enroll_spec'].cuda() if torch.cuda.is_available() else batch['enroll_spec']

            if predicter_type == "GT":
                alpha = batch['mixing_ratio'].cuda() if torch.cuda.is_available() else batch['mixing_ratio']
            elif predicter_type == "RAND":
                alpha = torch.rand(1).cuda() if torch.cuda.is_available() else torch.rand(1)
            elif predicter_type == "ONE":
                alpha = torch.tensor([1.0]).cuda() if torch.cuda.is_available() else torch.tensor([1.0])
            elif predicter_type == "ZERO":
                alpha = torch.tensor([0.0]).cuda() if torch.cuda.is_available() else torch.tensor([0.0])
            else:
                mixture_wav = batch['mixture'].cuda() if torch.cuda.is_available() else batch['mixture']
                enrollment_wav = batch['enroll'].cuda() if torch.cuda.is_available() else batch['enroll']
                alpha_true = batch['mixing_ratio'].cuda() if torch.cuda.is_available() else batch['mixing_ratio']
                alpha = t_predicter(mixture_wav, enrollment_wav, aug=False)

            multiple = config['dataset']['sample_rate'] * 3 // config['dataset']['hop_length'] + 1 # 3 seconds
            mixture, original_length = pad_and_reshape(mixture, multiple)
            solver = ODESolver(velocity_model=model)
            alpha_grid = torch.tensor([alpha.item(), 1.0], device=mixture.device)  # Solve from t=0 to t=1
            source_hat_spec = solver.sample(
                time_grid=alpha_grid,
                x_init=mixture.float(),
                method=config['solver']['method'],
                step_size=config['solver']['test_step_size'],
                enrollment=enrollment.repeat(mixture.shape[0], 1, 1),
            )
            source_hat_spec = reshape_and_remove_padding(source_hat_spec, original_length)
            source_hat = istft_torch(
                source_hat_spec, 
                n_fft=config['dataset']['n_fft'], 
                hop_length=config['dataset']['hop_length'], 
                win_length=config['dataset']['win_length'],
                length=batch['source'].shape[-1]
            )
            source_hat = scale_audio(source_hat.cpu())
            if save_audio:
                output_path = os.path.join(
                    output_dir,
                    batch['utt_id'][0],
                    batch['mixture_filename'][0].replace('.wav', ''),
                    'estimation.wav',
                )
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                torchaudio.save(output_path, source_hat, config['dataset']['sample_rate'])

                source_path = os.path.join(
                    output_dir,
                    batch['utt_id'][0],
                    batch['mixture_filename'][0].replace('.wav', ''),
                    'source.wav',
                )
                enroll_path = os.path.join(
                    output_dir,
                    batch['utt_id'][0],
                    batch['mixture_filename'][0].replace('.wav', ''),
                    'enroll.wav',
                )
                background_path = os.path.join(
                    output_dir,
                    batch['utt_id'][0],
                    batch['mixture_filename'][0].replace('.wav', ''),
                    'background.wav',
                )
                mixture_path = os.path.join(
                    output_dir,
                    batch['utt_id'][0],
                    batch['mixture_filename'][0].replace('.wav', ''),
                    'mixture.wav',
                )
                os.makedirs(os.path.dirname(source_path), exist_ok=True)
                os.makedirs(os.path.dirname(enroll_path), exist_ok=True)
                os.makedirs(os.path.dirname(background_path), exist_ok=True)
                os.makedirs(os.path.dirname(mixture_path), exist_ok=True)
                
                torchaudio.save(source_path, batch['source'], config['dataset']['sample_rate'])
                torchaudio.save(enroll_path, batch['enroll'], config['dataset']['sample_rate'])
                torchaudio.save(background_path, batch['background'], config['dataset']['sample_rate'])
                torchaudio.save(mixture_path, batch['mixture'], config['dataset']['sample_rate'])
            
            all_output_metrics = calculate_metrics(batch['mixture_rescaled'], batch['source_rescaled'], source_hat)
            if predicter_type == "ECAPAMLP":
                results.append({
                    'filename': batch['mixture_filename'][0],
                    **all_output_metrics,
                    'alpha': alpha_true.item(),
                    'alpha_hat': alpha.item(),
                })
            else:
                results.append({
                    'filename': batch['mixture_filename'][0],
                    **all_output_metrics,
                    'alpha': alpha.item(),
                    'alpha_hat': None,
                })
    
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(output_dir, 'metrics_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f'Saved metrics results to {results_csv_path}')
    return results_df

def main():
    args = parse_args()
    config_path = args.config
    error = 0.0

    if "noisy" in config_path:
        task = "noisy"
    else:
        task = "clean"
    output_dir = f'/test_results/{task}_{args.t_predicter}'

    print(f'Generating samples using FlowTSE model...')
    generate_samples_from_testset(
        config_path,
        FlowTSELightningModule,
        output_dir,
        predicter_type=args.t_predicter,
        error_range=[-error, error],  # Adjust the error range as needed
        save_audio=True
    )

if __name__ == '__main__':
    main()
