import PIL
import torch
import inspect
import torchaudio
from tqdm import tqdm
from typing import Union, List, Dict, Any, Optional
from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers import (
    DiffusionPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
)
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from diffusers.schedulers.scheduling_k_dpm_2_discrete import KarrasDiffusionSchedulers
from diffusers.utils import is_accelerate_available, is_accelerate_version, randn_tensor


class DiffusedAvatarPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        audio_feature_processor: Wav2Vec2Processor,
        audio_feature_extractor: Wav2Vec2ForCTC,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            audio_feature_processor=audio_feature_processor,
            audio_feature_extractor=audio_feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_sequential_cpu_offload
    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        Note that offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.14.0"):
            from accelerate import cpu_offload
        else:
            raise ImportError(
                "`enable_sequential_cpu_offload` requires `accelerate v0.14.0` or higher"
            )

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            cpu_offload(cpu_offloaded_model, device)

        if self.safety_checker is not None:
            cpu_offload(
                self.safety_checker, execution_device=device, offload_buffers=True
            )

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.enable_model_cpu_offload
    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError(
                "`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher."
            )

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        hook = None
        for cpu_offloaded_model in [self.text_encoder, self.unet, self.vae]:
            _, hook = cpu_offload_with_hook(
                cpu_offloaded_model, device, prev_module_hook=hook
            )

        if self.safety_checker is not None:
            _, hook = cpu_offload_with_hook(
                self.safety_checker, device, prev_module_hook=hook
            )

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    @property
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._execution_device
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline._encode_prompt
    def _encode_audio(
        self,
        audio: torch.FloatTensor,
        sampling_rate: int,
        video_fps: int,
        device: torch.device,
    ) -> List[torch.FloatTensor]:
        r"""
        Args:
            audio (:obj:`torch.FloatTensor` or :obj:`List[torch.FloatTensor]` or :obj:`str` or :obj:`List[str]`):
                The audio to encode. Can be a single audio or a list of audio. If a list of audio is provided, the
                batch size will be set to the length of the list. If a single audio is provided, the batch size will be 1
            sampling_rate (:obj:`int`):
                The sampling rate of the audio.
            device (:obj:`torch.device`):
                Which device to use for the encoding.
        """
        # Resample audio
        with torch.no_grad():
            audio = self.audio_feature_processor(
                audio, return_tensors="pt", sampling_rate=sampling_rate
            ).input_values
            audio_features = self.audio_feature_extractor(
                audio.squeeze(0).to(device), output_hidden_states=True
            ).hidden_states[-1]
        # Split audio
        n_chunks = (audio.shape[-1] / sampling_rate) * video_fps
        # Split array into chunks
        audio_chunks = torch.chunk(audio_features, int(n_chunks), dim=1)
        return audio_chunks

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start

    def prepare_inputs(
        self,
        identity_image,
        timestep,
        dtype,
        device,
        generator=None,
        motion_images=[],
    ):
        if not isinstance(identity_image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(identity_image)}"
            )

        # Prepare identity image
        identity_image = identity_image.to(device=device).to(dtype=dtype)
        identity_image_latent = self.vae.encode(identity_image).latent_dist.sample()
        identity_image_latent = self.vae.config.scaling_factor * identity_image_latent
        # Prepare motion images
        motion_images_latents = []
        for motion_image in motion_images:
            motion_image = motion_image.to(device=device).to(dtype=dtype)
            motion_image_latent = self.vae.encode(motion_image).latent_dist.sample()
            motion_image_latent = self.vae.config.scaling_factor * motion_image_latent
            motion_images_latents.append(motion_image_latent)

        # Get Noise
        noise = randn_tensor(
            identity_image_latent.shape, device=identity_image.device, dtype=identity_image.dtype
        )
        return torch.cat(motion_images_latents, dim=1), identity_image_latent, noise

    @torch.no_grad()
    def __call__(
        self,
        audio: torch.FloatTensor,
        identity_image: Union[torch.FloatTensor, PIL.Image.Image],
        sampling_rate: int = 16000,
        video_fps: int = 30,
        batch_size: int = 1,
        num_inference_steps: Optional[int] = 50,
        device="cuda",
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            audio (`torch.FloatTensor` of shape :obj:`(batch_size, num_channels, num_samples)`):
                The audio input. The audio should be normalized to a range between -1 and 1. The audio should be
                mono-channel and sampled at 16kHz.
            identity_image (`torch.FloatTensor` of shape :obj:`(batch_size, 3, 256, 256)` or :obj:`PIL.Image.Image`):
                The identity image. The image should be normalized to a range between -1 and 1. The image should be
                RGB and have a resolution of 256x256.
            sampling_rate (`int`, *optional*, defaults to 16000):
                The sampling rate of the audio input. The audio input will be resampled to this sampling rate.
            video_fps (`int`, *optional*, defaults to 30):
                The number of frames per second of the generated video.
            batch_size (`int`, *optional*, defaults to 1):
                Number of audio chunks to process at once
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter will be modulated by `strength`.
        """

        num_images_per_prompt = 1
        guidance_scale = 7.5
        do_classifier_free_guidance = True

        # 1. Preprocess audio
        audio_embeddings = self._encode_audio(
            audio,
            sampling_rate,
            video_fps,
            device,
        )
        # 2. Preprocess image
        identity_image = self.image_processor.preprocess(identity_image)

        # 3. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps, 1.0, device
        )
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 4. Prepare latent variables
        (
            motion_images_latents,
            identity_image_latent,
            noise_latent,
        ) = self.prepare_inputs(
            identity_image,
            latent_timestep,
            self.unet.dtype,
            device,
            None,
            [identity_image, identity_image],
        )

        # 5. Iterate over audio chunks TODO change this into batched inference
        output_images = []
        for audio_embedding in tqdm(audio_embeddings, "Decoding Audio", leave=False):
            output_image_latent = self.inference_one_chunk(
                audio_embedding,
                identity_image_latent,
                motion_images_latents,
                noise_latent,
                num_inference_steps,
                timesteps,
            )
            motion_images_latents = torch.cat([motion_images_latents[:, 4:], output_image_latent], dim=1)
            image = self.decode_latents(output_image_latent).to(self.unet.dtype)
            image = self.image_processor.postprocess(
                image, output_type='pil'
            )
            output_images.append(image)
            noise_latent = randn_tensor(
                noise_latent.shape,
                device=noise_latent.device,
                dtype=noise_latent.dtype,
            )
        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()
        return output_images

    def inference_one_chunk(
        self,
        audio_embedding: torch.FloatTensor,
        identity_image_latent: torch.FloatTensor,
        motion_images_latents: torch.FloatTensor,
        noise_latent: torch.FloatTensor,
        num_inference_steps: int = 50,
        timesteps: Optional[torch.Tensor] = None,
    ):
        do_classifier_free_guidance = True
        guidance_scale = 10
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = self.scheduler.scale_model_input(noise_latent, t)
                # Add identity image and motion frames
                inputs = torch.cat(
                    [
                        latent_model_input,
                        identity_image_latent,
                        motion_images_latents,
                    ],
                    dim=1,
                )
                # predict the noise residual
                noise_pred = self.unet(
                    inputs,
                    t,
                    encoder_hidden_states=audio_embedding,
                    cross_attention_kwargs=None,
                ).sample

                # # perform guidance
                # if do_classifier_free_guidance:
                #     noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                #     noise_pred = noise_pred_uncond + guidance_scale * (
                #         noise_pred_text - noise_pred_uncond
                #     )
                # compute the previous noisy sample x_t -> x_t-1
                noise_latent = self.scheduler.step(
                    noise_pred, t, noise_latent
                ).prev_sample
                progress_bar.update(1)
        return noise_latent


from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
import torch.nn as nn

# Load scheduler, tokenizer and models.
noise_scheduler = DDPMScheduler.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
)
vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="unet",
)
unet.conv_in = nn.Conv2d(
    16,
    unet.conv_in.out_channels,
    kernel_size=unet.conv_in.kernel_size,
    padding=unet.conv_in.padding,
    dtype=unet.conv_in.weight.dtype,
)
unet.encoder_hid_proj = nn.Linear(1024, 768)

# Freeze vae and text_encoder
vae.requires_grad_(False)
audio_feature_processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
)
audio_feature_extractor = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
)


pipe = DiffusedAvatarPipeline(
    scheduler=noise_scheduler,
    vae=vae,
    unet=unet,
    audio_feature_processor=audio_feature_processor,
    audio_feature_extractor=audio_feature_extractor,
).to("cuda")

audio = torch.rand(1, 16000)
identity_image = torch.rand(1, 3, 256, 256)

out = pipe(
    audio,
    identity_image,
    sampling_rate=16000,
    video_fps=25,
    batch_size=1,
    num_inference_steps=50,
    device="cuda",
)

print(len(out))
# Test pipeline
# Use pipeline in validation testing
# Need to pad audio features to match video fps
# Test trainer code
# Scale up dataset
# Process dataset
# Train on dataset
