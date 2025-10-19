import argparse
import os
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from asteroid.engine.optimizers import make_optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LambdaLR, SequentialLR

from flow_matching.path import CondOTProbPath
from flow_matching.solver.ode_solver import ODESolver

from utils import neg_sisdr_loss_wrapper
from data.datasets import get_dataloaders
from models.t_predicter import TPredicter
from utils.transforms import istft_torch
from utils.helper import sample_mixing_ratio_by_snr_range


def parse_args():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--config', default='config/config.yaml', help='Path to the config file.')
    args = parser.parse_args()
    return args


def parse_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


class LightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = TPredicter(
            **config['model']
        )
        self.config = config
        self.save_hyperparameters(config)  # Saves config for checkpointing
        self.neg_si_sdr = neg_sisdr_loss_wrapper

    def forward(self, x, enrollment):
        return self.model(x, enrollment, aug=False)

    def training_step(self, batch, batch_idx):
        source = batch['source_rescaled']
        background = batch['background_rescaled']
        enrollment = batch['enroll']
        batch_size = source.size(0)
        path = CondOTProbPath()

        alpha = torch.rand((batch_size, ), device=source.device)

        path_sample = path.sample(t=alpha, x_0=background, x_1=source)
        mixture = path_sample.x_t
        alpha_hat = self.model(mixture, enrollment, aug=self.config['train'].get('spectral_aug', False))
        loss = torch.nn.functional.mse_loss(alpha_hat, alpha)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        source = batch['source_rescaled']
        background = batch['background_rescaled']
        enrollment = batch['enroll']
        path = CondOTProbPath()

        if self.config['dataset']['snr_range'] is None:
            # alpha = torch.rand((batch_size, ), device=source.device)
            alpha = torch.rand((1, ), device=source.device)
        else:
            alpha = batch['alpha'].squeeze(1)

        path_sample = path.sample(t=alpha, x_0=background, x_1=source)
        mixture = path_sample.x_t
        alpha_hat = self.model(mixture, enrollment, aug=False)
        loss = torch.nn.functional.mse_loss(alpha_hat, alpha)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=mixture.size(0))

        return loss

    def configure_optimizers(self):
        optimizer = make_optimizer(self.model.parameters(), **self.config['optim'])
        if self.config['scheduler']['type'] == 'ReduceLROnPlateau':
            scheduler = {
                'scheduler': ReduceLROnPlateau(
                    optimizer=optimizer,
                    factor=self.config['scheduler']['lr_reduce_factor'],
                    patience=self.config['scheduler']['lr_reduce_patience']
                ),
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1,
                'strict': True,
            }
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        
        elif self.config['scheduler']['type'] == 'CosineAnnealingLR':
            # Warm-up scheduler
            def warmup_lambda(epoch):
                if epoch < self.config['scheduler']['warmup_epochs']:
                    return epoch / self.config['scheduler']['warmup_epochs']
                return 1.0

            warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

            # Cosine annealing scheduler
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.config['scheduler']['t_max'],
                eta_min=self.config['scheduler']['eta_min']
            )

            # Combine warm-up and cosine annealing
            scheduler = {
                'scheduler': SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[self.config['scheduler']['warmup_epochs']]
                ),
                'interval': 'epoch',
                'frequency': 1
            }

            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        else:
            return optimizer
    

class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.rank = int(os.environ.get('RANK', 0))

    def setup(self, stage=None):
        self.train_loader, self.val_loader = get_dataloaders(
            self.config,
            is_ddp=False,
            world_size=self.world_size,
            rank=self.rank
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader


def main():
    args = parse_args()
    config = parse_config(args.config)
    pl.seed_everything(config['seed'])
    torch.set_float32_matmul_precision('medium') # set to 'medium' or 'high'

    # Create necessary directories
    os.makedirs(config['train']['log_dir'], exist_ok=True)
    os.makedirs(config['checkpoint']['dir'], exist_ok=True)

    # Instantiate the LightningModule
    if config['checkpoint']['load_weights_only'] and config['checkpoint']['resume']:
        model = LightningModule.load_from_checkpoint(
            config['checkpoint']['resume'],
            config=config,
            strict=True
        )
    else:
        model = LightningModule(config)

    # Instantiate the DataModule
    data_module = DataModule(config)

    # Callbacks
    callbacks = []

    # EarlyStopping
    if config['early_stopping']['enabled']:
        early_stopping_callback = EarlyStopping(
            monitor=config['early_stopping']['monitor'],
            patience=config['early_stopping']['patience'],
            verbose=config['early_stopping']['verbose'],
            mode=config['early_stopping']['mode'],
            min_delta=config['early_stopping']['delta'],
        )
        callbacks.append(early_stopping_callback)

    # ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['checkpoint']['dir'],
        filename=config['checkpoint']['ckpt_name'],
        save_top_k=config['checkpoint']['save_best'],
        save_last=config['checkpoint']['save_last'],
        verbose=config['checkpoint']['verbose'],
        monitor=config['checkpoint']['monitor'],
        mode=config['checkpoint']['mode'],
    )
    callbacks.append(checkpoint_callback)

    # LearningRateMonitor
    if config['train']['log_lr']:
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)

    # Logger
    tb_logger = TensorBoardLogger(
        save_dir=config['train']['log_dir'],
        name='lightning_logs',
        version='0',
    )

    # Determine if DDP should be used
    ddp_config = config.get('ddp', {})
    use_ddp = ddp_config.get('use_ddp', False)
    num_nodes = ddp_config.get('num_nodes', 1)
    num_gpus = ddp_config.get('num_gpus', torch.cuda.device_count())
    strategy = ddp_config.get('strategy', 'ddp')
    strategy = DDPStrategy(find_unused_parameters=False)

    if use_ddp and num_gpus > 1:
        # Explicitly set DDP strategy and related arguments
        trainer = pl.Trainer(
            max_epochs=config['train']['num_epochs'],
            accelerator='gpu',
            devices=num_gpus,
            num_nodes=num_nodes,
            strategy=strategy,
            accumulate_grad_batches=config['train']['accumulation_steps'],
            callbacks=callbacks,
            default_root_dir=config['train']['log_dir'],
            logger=tb_logger,
            log_every_n_steps=config['train']['log_interval'],
            precision=config['train']['precision'],
            detect_anomaly=config['train']['detect_anomaly'],
            gradient_clip_val=config['train']['gradient_clip_val'],
            limit_train_batches=config['train']['limit_train_batches'],
            limit_val_batches=config['train']['limit_val_batches'],
            # overfit_batches=config['train']['overfit_batches'],
        )
    else:
        # Single GPU or CPU
        trainer = pl.Trainer(
            max_epochs=config['train']['num_epochs'],
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            accumulate_grad_batches=config['train']['accumulation_steps'],
            callbacks=callbacks,
            default_root_dir=config['train']['log_dir'],
            logger=tb_logger,
            log_every_n_steps=config['train']['log_interval'],
            precision=config['train']['precision'],
            gradient_clip_val=config['train']['gradient_clip_val'],
            detect_anomaly=config['train']['detect_anomaly'],
            limit_train_batches=config['train']['limit_train_batches'],
            limit_val_batches=config['train']['limit_val_batches'],
            # overfit_batches=config['train']['overfit_batches'],
        )

    # Resume from checkpoint if specifiedo
    ckpt_path = config['checkpoint']['resume'] if config['checkpoint']['resume'] else None

    # Fit the model
    if config['checkpoint']['load_weights_only']:
        trainer.fit(model, datamodule=data_module)
    else:
        trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == '__main__':
    main()