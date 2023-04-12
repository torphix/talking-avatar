import argparse
import logging
import math
import os
import torch.nn as nn
import random
from pathlib import Path
from data import AudioVideoFrameDataset
import accelerate
import datasets
import torchvision
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
import json
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available


if is_wandb_available():
    import wandb


class Trainer:
    def __init__(self, args, accelerator, weight_dtype):
        self.args = self.args
        logging_dir = os.path.join(args.output_dir, args.logging_dir)
        accelerator_project_config = ProjectConfiguration(
            total_limit=args.checkpoints_total_limit
        )
        self.accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )
        self.weight_dtype = weight_dtype
        self.logger = get_logger(__name__, log_level="INFO")
        self._init_model()
        self._init_optimizer()
        self._init_dataloader()
        self._init_scheduler()

    def _init_model(self):
        # Load model
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.vae = AutoencoderKL.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="vae", revision=self.args.revision
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            self.args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=self.args.non_ema_revision,
        )
        # Freeze self.vae and text_encoder
        self.vae.requires_grad_(False)

        # Create EMA for the unet.
        if self.args.use_ema:
            ema_unet = UNet2DConditionModel.from_pretrained(
                self.args.pretrained_model_name_or_path, subfolder="unet", revision=self.args.revision
            )
            ema_unet = EMAModel(
                ema_unet.parameters(),
                model_cls=UNet2DConditionModel,
                model_config=ema_unet.config,
            )

        if self.args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    self.logger.warn(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly"
                )

    def _init_optimizer(self):
        # Initialize the optimizer
        if self.args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                )

            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        self.optimizer = optimizer_cls(
            self.unet.parameters(),
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )

    def _init_dataloader(self):
        # Preprocessing the datasets.
        train_transforms = transforms.Compose(
            [
                transforms.Resize(
                    self.args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(self.args.resolution)
                if self.args.center_crop
                else transforms.RandomCrop(self.args.resolution),
                transforms.RandomHorizontalFlip()
                if self.args.random_flip
                else transforms.Lambda(lambda x: x),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        # Load Data
        train_dataset = AudioVideoFrameDataset(self.args.data_dir, train_transforms)

        # DataLoaders creation:
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )

    def _init_scheduler(self):
        # Scheduler and math around the number of training steps.
        self.overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.args.gradient_accumulation_steps
        )
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.num_train_epochs * num_update_steps_per_epoch
            self.overrode_max_train_steps = True

        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.args.gradient_accumulation_steps,
        num_training_steps=self.args.max_train_steps * self.args.gradient_accumulation_steps,
    )
        
    def _init_accelerator(self):
        # Prepare everything with our `accelerator`.
        self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler
        )
        # Override unet in layer
        self.unet.conv_in = nn.Conv2d(
            16,
            self.unet.conv_in.out_channels,
            kernel_size=self.unet.conv_in.kernel_size,
            padding=self.unet.conv_in.padding,
            dtype=self.unet.conv_in.weight.dtype,
        )
        self.unet.encoder_hid_proj = nn.Linear(1024, 768)

        if self.args.use_ema:
            self.ema_unet.to(self.accelerator.device)

        # For mixed precision training we cast the text_encoder and self.vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move text_encode and self.vae to gpu and cast to weight_dtype
        self.vae.to(self.accelerator.device, dtype=weight_dtype)
        self.unet.to(self.accelerator.device)
        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.args.gradient_accumulation_steps
        )
        if self.overrode_max_train_steps:
            self.args.max_train_steps = self.args.num_train_epochs * self.num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.args.num_train_epochs = math.ceil(self.args.max_train_steps / self.num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.accelerator.is_main_process:
            tracker_config = dict(vars(self.args))
            self.accelerator.init_trackers(self.args.tracker_project_name, tracker_config)


    def _load_checkpoint(self):
        # Potentially load in the weights and states from a previous save
        if self.args.resume_from_checkpoint:
            if self.args.resume_from_checkpoint != "latest":
                path = os.path.basename(self.args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(self.args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{self.args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                self.args.resume_from_checkpoint = None
            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                self.accelerator.load_state(os.path.join(self.args.output_dir, path))
                global_step = int(path.split("-")[1])

                resume_global_step = global_step * self.args.gradient_accumulation_steps
                first_epoch = global_step // self.num_update_steps_per_epoch
                resume_step = resume_global_step % (
                    self.num_update_steps_per_epoch * self.args.gradient_accumulation_steps
                )
            return global_step, first_epoch, resume_step
        
    def compute_snr(self, timesteps):
            """
            Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
            """
            alphas_cumprod = self.noise_scheduler.alphas_cumprod
            sqrt_alphas_cumprod = alphas_cumprod**0.5
            sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

            # Expand the tensors.
            # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
            sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
                timesteps
            ].float()
            while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
                sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
            alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
                device=timesteps.device
            )[timesteps].float()
            while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
                sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
            sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

            # Compute SNR.
            snr = (alpha / sigma) ** 2
            return snr


    def train(self):
        # Train!
        total_batch_size = (
            self.args.train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )

        self.logger.info("***** Running training *****")
        self.logger.info(f"  Num examples = {len(self.train_dataset)}")
        self.logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        self.logger.info(f"  Instantaneous batch size per device = {self.args.train_batch_size}")
        self.logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        self.logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        self.logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
        global_step = 0
        first_epoch = 0

        if self.args.resume_from_checkpoint:
            global_step, first_epoch, resume_step = self._load_checkpoint()
            

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(
            range(global_step, self.args.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
        )
        progress_bar.set_description("Steps")
        last_step_check = 0
        for epoch in range(first_epoch, self.args.num_train_epochs):
            unet.train()
            train_loss = 0.0
            for step, batch in enumerate(self.train_dataloader):
                # Skip steps until we reach the resumed step
                if (
                    self.args.resume_from_checkpoint
                    and epoch == first_epoch
                    and step < resume_step
                ):
                    if step % self.args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                with self.accelerator.accumulate(unet):
                    # Convert images to latent space
                    latents = self.vae.encode(
                        batch["target_frame"].to(self.weight_dtype)
                    ).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor
                    # Encode id image
                    id_latents = self.vae.encode(
                        batch["id_frame"].to(self.weight_dtype)
                    ).latent_dist.sample()
                    id_latents = id_latents * self.vae.config.scaling_factor
                    # Encode motion images
                    motion_latents = torch.cat(
                        [
                            self.vae.encode(l.to(self.weight_dtype)).latent_dist.sample()
                            for l in batch["motion_frames"]
                        ],
                        1,
                    )
                    motion_latents = motion_latents * self.vae.config.scaling_factor
                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    if self.args.noise_offset:
                        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                        noise += self.args.noise_offset * torch.randn(
                            (latents.shape[0], latents.shape[1], 1, 1),
                            device=latents.device,
                        )

                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0,
                        self.noise_scheduler.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    )
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
                    # Add the identity frame to the noisy latents
                    noisy_latents = torch.cat(
                        [
                            noisy_latents,
                            id_latents,
                            motion_latents,
                        ],
                        dim=1,
                    )
                    # Get the text embedding for conditioning
                    encoder_hidden_states = batch["audio_frame"].to(self.weight_dtype)

                    # Get the target for loss depending on the prediction type
                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(
                            f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
                        )
                    # Predict the noise residual and compute loss
                    model_pred = unet(
                        noisy_latents, timesteps, encoder_hidden_states
                    ).sample

                    if self.args.snr_gamma is None:
                        loss = F.mse_loss(
                            model_pred.float(), target.float(), reduction="mean"
                        )
                    else:
                        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                        # This is discussed in Section 4.2 of the same paper.
                        snr = self.compute_snr(timesteps)
                        mse_loss_weights = (
                            torch.stack(
                                [snr, self.args.snr_gamma * torch.ones_like(timesteps)], dim=1
                            ).min(dim=1)[0]
                            / snr
                        )
                        # We first calculate the original loss. Then we mean over the non-batch dimensions and
                        # rebalance the sample-wise losses with their respective loss weights.
                        # Finally, we take the mean of the rebalanced loss.
                        loss = F.mse_loss(
                            model_pred.float(), target.float(), reduction="none"
                        )
                        loss = (
                            loss.mean(dim=list(range(1, len(loss.shape))))
                            * mse_loss_weights
                        )
                        loss = loss.mean()

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = self.accelerator.gather(loss.repeat(self.args.train_batch_size)).mean()
                    train_loss += avg_loss.item() / self.args.gradient_accumulation_steps

                    # Backpropagate
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(unet.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    if self.args.use_ema:
                        self.ema_unet.step(unet.parameters())
                    progress_bar.update(1)
                    global_step += 1
                    self.accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0

                    if global_step % self.args.checkpointing_steps == 0:
                        if self.accelerator.is_main_process:
                            save_path = os.path.join(
                                self.args.output_dir, f"checkpoint-{global_step}"
                            )
                            self.accelerator.save_state(save_path)
                            self.logger.info(f"Saved state to {save_path}")

                logs = {
                    "step_loss": loss.detach().item(),
                    "lr": self.lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)

                if global_step >= self.args.max_train_steps:
                    break
                # Validation
                if (
                global_step % self.args.check_val_every_n_steps == 0 and global_step != 0
                ):
                    if global_step == last_step_check:
                        continue
                    else:
                        last_step_check = global_step
                    if self.args.use_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        self.ema_unet.store(unet.parameters())
                        self.ema_unet.copy_to(unet.parameters())
                    print('Running validation')
                    self.log_validation(
                        self.vae,
                        unet,
                        self.noise_scheduler,
                        ['/home/ubuntu/data/vox/processed/datapoints/id06209/6oI-FJQS9V0/00027/00027_0.json'],
                        self.train_transforms,
                        self.args,
                        self.accelerator,
                        self.weight_dtype,
                        global_step,
                    )
                    if self.args.use_ema:
                        # Switch back to the original UNet parameters.
                        self.ema_unet.restore(unet.parameters())

            # Create the pipeline using the trained modules and save it.
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                unet = self.accelerator.unwrap_model(unet)
                if self.args.use_ema:
                    self.ema_unet.copy_to(unet.parameters())

                pipeline = StableDiffusionPipeline.from_pretrained(
                    self.args.pretrained_model_name_or_path,
                    text_encoder=self.text_encoder,
                    vae=self.vae,
                    unet=unet,
                    revision=self.args.revision,
                )
                pipeline.save_pretrained(self.args.output_dir)

                if self.args.push_to_hub:
                    upload_folder(
                        repo_id=self.repo_id,
                        folder_path=self.args.output_dir,
                        commit_message="End of training",
                        ignore_patterns=["step_*", "epoch_*"],
                )

            self.accelerator.end_training()



    def log_validation(
            self, vae, unet, noise_scheduler, val_files, train_transforms, args, accelerator, weight_dtype, global_step
    ):
        self.logger.info("Running validation... ")
        with torch.no_grad():
            for datapoint in val_files:
                # Prepare datapoint
                with open(datapoint, 'r') as f:
                    data = json.load(f)
                audio_frame = torch.from_numpy(np.load(data["audio_frame"])).to(accelerator.device).unsqueeze(0)
                video_frame = torch.from_numpy(np.load(data["video_frame"]))
                # Randomly select a frame from the video frames directory
                dir_path = "/".join(data["video_frame"].split("/")[:-1])
                id_frame = f'{dir_path}/{random.choice(os.listdir(dir_path))}'
                id_frame = torch.from_numpy(np.load(id_frame))
                # Get motion frames
                if data["motion_frame"]:
                    motion_frames = [
                        torch.from_numpy(np.load(frame)) for frame in data["motion_frame"]
                    ]
                else:
                    motion_frames = [id_frame, id_frame]
                # Apply transforms
                if train_transforms:
                    video_frame = train_transforms(video_frame/255).to(accelerator.device).to(weight_dtype).unsqueeze(0)
                    id_frame = train_transforms(id_frame/255).to(accelerator.device).to(weight_dtype).unsqueeze(0)
                    motion_frames = [train_transforms(frame/255).to(accelerator.device).to(weight_dtype).unsqueeze(0) for frame in motion_frames]
                # Prepare for input
                video_latent = vae.encode(video_frame.to(weight_dtype)).latent_dist.sample()
                video_latent = video_latent * vae.config.scaling_factor
                # Encode id image
                id_latents = vae.encode(
                    id_frame.to(weight_dtype)
                ).latent_dist.sample()
                id_latents = id_latents * vae.config.scaling_factor
                # Encode motion images
                motion_latents = torch.cat(
                    [
                        vae.encode(l.to(weight_dtype)).latent_dist.sample()
                        for l in motion_frames
                    ],
                    1,
                )
                motion_latents = motion_latents * vae.config.scaling_factor
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(video_latent)
                timesteps = torch.tensor([50]).long().to(accelerator.device)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(video_latent, noise, timesteps)
                # Get the text embedding for conditioning
                encoder_hidden_states = audio_frame.to(weight_dtype)
                for t in range(timesteps):
                    # Add the identity frame to the noisy latents
                    inputs = torch.cat(
                        [
                            noisy_latents,
                            id_latents,
                            motion_latents,
                        ],
                        dim=1,
                    )
                    # Input into model
                    noise_pred = unet(
                            inputs, timesteps, encoder_hidden_states
                    ).sample
                    noisy_latents = noise_scheduler.step(noise_pred, t, noisy_latents).prev_sample.to(weight_dtype)
            # Decode the latents
            noisy_latents = 1 / vae.config.scaling_factor * noisy_latents
            image = vae.decode(noisy_latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
            image = image.cpu().float()
            # Save the image
            os.makedirs('results', exist_ok=True)
            torchvision.utils.save_image(image, f'results/{global_step}_image.png')
        torch.cuda.empty_cache()

