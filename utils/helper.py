import pandas as pd
import torch

def get_primary_speaker_id(filepath: str) -> str:
    """
    Given a path like:
        "/path/to/fileid_12345_spkid_00054-11429-04842.wav"
    or
        "/path/to/fileid_12345_spkid_07406.wav"
    This function returns:
        "00054"  (for multiple speakers)
    or
        "07406"  (for single speaker)
    """
    spkid_part = filepath.split("spkid_")[1]
    spkid_part = spkid_part.replace(".wav", "")
    primary_speaker_id = spkid_part.split("-")[0]

    return primary_speaker_id

def add_speaker_id(df):
    df['speaker_id'] = df['clean_file_path'].apply(get_primary_speaker_id)
    return df

def label_speaker_id(df):
    add_speaker_id(df)
    unique_speakers = df['speaker_id'].unique()
    speaker2idx = {speaker: idx for idx, speaker in enumerate(unique_speakers)}
    df['label'] = df['speaker_id'].map(speaker2idx)
    return df

def sample_mixing_ratio_by_snr_range(source, background, snr_range):
    """
    Given a source and background signal, this function calculates the mixing ratio
    based on the desired SNR range.

    source: Tensor of shape (batch_size, num_samples)
    background: Tensor of shape (batch_size, num_samples)
    snr_range: Tuple of (min_snr, max_snr) in dB
    """
    # Calculate the power of the source and background signals
    source_rms = torch.mean(source ** 2, dim=-1).sqrt()
    background_rms = torch.mean(background ** 2, dim=-1).sqrt()
    num_samples = source.size(0)

    # Calculate the desired SNR in linear scale
    snr_db = torch.rand(num_samples, device=source.device) * (snr_range[1] - snr_range[0]) + snr_range[0]
    snr_sqrt = torch.sqrt(10 ** (snr_db / 10))

    # Calculate the mixing ratio
    # mixing_ratio = source_power / (source_power + background_power * snr_sqrt)
    mixing_ratio = snr_sqrt * background_rms / (source_rms + snr_sqrt * background_rms)
    return mixing_ratio

def calc_mixing_ratio_by_signal(mixture, source, background):
    """
    Given a normalized source and background signal, this function calculates the mixing ratio.
    """
    eps = 1e-8
    # mixture_m_background = mixture - background
    # source_m_background = source - background
    # source_m_background_power = torch.mean(source_m_background ** 2, dim=-1)
    # mixture_m_background_power = torch.mean(mixture_m_background ** 2, dim=-1)
    # mixing_ratio = torch.sqrt(mixture_m_background_power / source_m_background_power)
    mixing_ratio = (mixture - background) / (source - background + eps)
    return mixing_ratio

# Example usage:
if __name__ == "__main__":
    # df = pd.read_csv('/work/hdd/bdql/thsieh/TGIF-Dataset/tr/metadata.csv')
    # df = label_speaker_id(df)
    # print(df)

    # Example usage of sample_mixing_ratio_by_snr_range
    from torchmetrics.functional.audio.snr import signal_noise_ratio
    import torchaudio
    from data.datasets import LibriMixInformed
    from tqdm import tqdm
    import yaml

    with open('config/config_FlowTSE.yaml', 'r') as f:
        config = yaml.safe_load(f)

    test_dataset = LibriMixInformed(
        csv_dir=config['dataset']['test_dir'],
        librimix_meta_dir=config['dataset']['librimix_meta_dir'],
        task=config['dataset']['task'],
        sample_rate=config['dataset']['sample_rate'],
        n_src=config['dataset']['n_src'],
        segment=3,
        segment_aux=3,
    )
    all_mixing_ratios = []
    for i, batch in enumerate(tqdm(test_dataset)):
        if i > 200:
            break
        source = batch['source']
        background = batch['background']

        # test batch processing
        source = source.unsqueeze(0).repeat(3, 1)
        background = background.unsqueeze(0).repeat(3, 1)

        snr_range = (-15.0, 25.0)
        mixing_ratio = sample_mixing_ratio_by_snr_range(source, background, snr_range)
        mixing_ratio = mixing_ratio.unsqueeze(1)
        mixture = mixing_ratio * source + (1 - mixing_ratio) * background
        # snr = signal_noise_ratio(mixture, mixing_ratio * source).item()

        # all_mixing_ratios.append(mixing_ratio.item())
    # all_mixing_ratios = torch.tensor(all_mixing_ratios)
    # Example usage of calc_mixing_ratio_by_signal
    # source_ = source / source.abs().max()
    # background_ = background / background.abs().max()
    # alpha = torch.rand(1).item()
    # mixture = alpha * source_ + (1 - alpha) * background_
    # mixing_ratio = calc_mixing_ratio_by_signal(mixture, source_, background_)