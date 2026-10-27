"""
Occlusion-aware training of SMIRK (src/occlusion_trainer.py) on clean faces + pre-generated occlusions.

    python train_occlusion.py configs/config_occlusion_finetune.yaml --paths paths.yaml train.log_path=logs/finetune

paths.yaml gives the data folders (written by the Kaggle notebook):
    data:
      clean_dirs: [.../clean_faces, ...]               clean_meta.csv + landmarks.npz
      image_dirs: {expw: .../images, affectnet: ...}   256x256 crops
      occluded_dirs: {expw: .../occluded_expw, ...}    occluded_meta.csv + images.bin + masks.bin
      fan_files: {expw: .../expw_fan.npz, ...}         FAN landmarks (optional, extract_fan_landmarks.py)
      exclude_files: [.../exclude_train.csv]           faces to leave out (optional, find_duplicates.py)

Kaggle sessions stop after 12 h: a checkpoint (model, optimizers, schedulers, position) is written every
train.save_every_iters iterations and before train.time_limit_hours; restart with resume_checkpoint=<checkpoint_last.pt>.
Weights only (same format as SMIRK_em1.pt, for the evaluation benchmark) are written to model_<epoch>.pt / model_final.pt.

Runs are reproducible with the config seed: the faces of every epoch, the augmentations (worker seeds) and the random
parts of the training (masking, cycle path) depend only on the seed and the epoch; a resumed run skips the batches of
the interrupted epoch that were already trained (GPU kernels are not fully deterministic, so results still differ
slightly between runs).
"""
import os
import sys
import time
import random
import argparse
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from src.occlusion_trainer import OcclusionTrainer
from datasets.occluded_face_dataset import OccludedFaceDataset


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--paths', default=None, help='YAML with the data folders')
    args, overrides = parser.parse_known_args()
    config = OmegaConf.load(args.config)
    if args.paths:
        config = OmegaConf.merge(config, OmegaConf.load(args.paths))
    OmegaConf.set_struct(config, True)
    return OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))


class EpochSampler(torch.utils.data.Sampler):
    """num_samples random faces per epoch (without replacement, as RandomSampler), fixed by the seed and the epoch;
    set_epoch(epoch, skip) also skips the first faces (resuming in the middle of an epoch)."""

    def __init__(self, size, num_samples, seed):
        self.size, self.num_samples, self.seed = size, min(num_samples or size, size), seed
        self.epoch, self.skip = 0, 0

    def set_epoch(self, epoch, skip=0):
        self.epoch, self.skip = epoch, skip

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed * 100003 + self.epoch)
        return iter(torch.randperm(self.size, generator=generator)[self.skip:self.num_samples].tolist())

    def __len__(self):
        return max(self.num_samples - self.skip, 0)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)  # also the base seed of the data loader workers (worker_init_fn)


def make_loader(config, split, samples=None, shuffle=True):
    dataset = OccludedFaceDataset(list(config.data.clean_dirs), dict(config.data.image_dirs), dict(config.data.occluded_dirs),
                                  split, image_size=config.image_size,
                                  train_scale=(config.train.train_scale_min, config.train.train_scale_max),
                                  test_scale=config.train.test_scale, p_occluded=config.occlusion.p_occluded,
                                  max_occluded_area=config.occlusion.max_occluded_area,
                                  fan_files=dict(config.data.get('fan_files') or {}),
                                  exclude_files=list(config.data.get('exclude_files') or []))
    sampler = EpochSampler(len(dataset), samples, config.seed) if shuffle else None
    return torch.utils.data.DataLoader(dataset, batch_size=config.train.batch_size, sampler=sampler, drop_last=True,
                                       num_workers=config.train.num_workers, pin_memory=True,
                                       worker_init_fn=lambda worker: np.random.seed(torch.initial_seed() % 2 ** 32))


def main():
    config = parse_config()
    log_path = config.train.log_path
    for sub in ['', 'train_images', 'val_images']:
        os.makedirs(os.path.join(log_path, sub), exist_ok=True)
    OmegaConf.save(config, os.path.join(log_path, 'config.yaml'))

    seed_everything(config.seed)
    train_loader = make_loader(config, 'train', samples=config.train.samples_per_epoch)
    val_loader = make_loader(config, 'val', shuffle=False)
    iters_per_epoch = len(train_loader)
    print(f'train faces: {len(train_loader.dataset)} ({iters_per_epoch} iterations / epoch), val faces: {len(val_loader.dataset)}, '
          f'left out as duplicates of evaluation faces: {train_loader.dataset.excluded} train / {val_loader.dataset.excluded} val')

    trainer = OcclusionTrainer(config).to(config.device)
    if config.resume:  # initial weights, e.g. pretrained_models/SMIRK_em1.pt
        trainer.load_model(config.resume, load_fuse_generator=config.load_fuse_generator, load_encoder=config.load_encoder, device=config.device)
    trainer.create_base_encoder()  # the initial encoder (pretrained SMIRK), frozen: occlusion.teacher=base, regularization
    trainer.apply_freezing()
    print('trained parameters:', {k: f'{v / 1e6:.2f}M' for k, v in trainer.trainable_parameters().items()})

    start_epoch, start_step, state = 0, 0, None
    if config.resume_checkpoint:
        state = trainer.load_checkpoint_model(config.resume_checkpoint, config.device)
        start_epoch, start_step = state['epoch'], state['step']
        print(f'resuming from {config.resume_checkpoint}: epoch {start_epoch}, step {start_step}')

    checkpoint_path = os.path.join(log_path, 'checkpoint_last.pt')
    started, total_iters = time.time(), 0
    for epoch in range(start_epoch, config.train.num_epochs):
        trainer.configure_optimizers(iters_per_epoch)
        if state is not None:
            trainer.load_training_state(state)
            state = None

        first = start_step if epoch == start_epoch else 0
        seed_everything(config.seed * 1000 + epoch * 100 + (1 if first else 0))  # a resumed epoch gets other random numbers
        train_loader.sampler.set_epoch(epoch, skip=first * config.train.batch_size)
        for step, batch in enumerate(tqdm(train_loader, desc=f'epoch {epoch}', initial=first, total=iters_per_epoch), start=first):
            if step >= iters_per_epoch:
                break
            trainer.set_freeze_status(config, step, epoch)
            batch = {k: v.to(config.device) for k, v in batch.items()}
            outputs = trainer.step(batch, step, phase='train')
            total_iters += 1

            if step % config.train.visualize_every == 0:
                with torch.no_grad():
                    trainer.save_visualizations(trainer.create_visualizations(batch, outputs),
                                                os.path.join(log_path, 'train_images', f'{epoch}_{step}.jpg'), show_landmarks=True)

            out_of_time = (time.time() - started) / 3600 > config.train.time_limit_hours
            smoke_done = config.train.max_iters and total_iters >= config.train.max_iters
            if (step + 1) % config.train.save_every_iters == 0 or out_of_time or smoke_done:
                trainer.save_checkpoint(checkpoint_path, epoch, step + 1)
            if out_of_time or smoke_done:
                print(f'stopping after {total_iters} iterations ({"time limit" if out_of_time else "max_iters"}); '
                      f'resume with resume_checkpoint={checkpoint_path}')
                trainer.save_model(trainer.state_dict(), os.path.join(log_path, 'model_last.pt'))
                return

        # validation (losses only; the occlusion benchmark gives the real evaluation)
        for step, batch in enumerate(val_loader):
            if step >= config.train.val_batches:
                break
            batch = {k: v.to(config.device) for k, v in batch.items()}
            outputs = trainer.step(batch, step, phase='val')
            if step == 0:
                with torch.no_grad():
                    trainer.save_visualizations(trainer.create_visualizations(batch, outputs),
                                                os.path.join(log_path, 'val_images', f'{epoch}.jpg'), show_landmarks=True)

        trainer.save_checkpoint(checkpoint_path, epoch + 1, 0)
        trainer.save_model(trainer.state_dict(), os.path.join(log_path, f'model_{epoch}.pt'))

    trainer.save_model(trainer.state_dict(), os.path.join(log_path, 'model_final.pt'))
    print('training finished:', os.path.join(log_path, 'model_final.pt'))


if __name__ == '__main__':
    sys.exit(main())
