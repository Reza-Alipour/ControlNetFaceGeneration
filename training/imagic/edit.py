import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
from utils import *

import accelerate
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, Dataset, DatasetDict
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils import load_image
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, Adafactor

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.24.0.dev0")

logger = get_logger(__name__)


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    if args.load_unet_from_local:
        unet = UNet2DConditionModel.from_pretrained(args.unet_local_path)
    else:
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
        )

    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path, 
                                                     token=args.hub_read_token, 
                                                     revision=args.controlnet_load_revision)
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = ControlNetModel.from_unet(unet, revision='english')

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]

                    sub_dir = "controlnet"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.requires_grad_(False)
    unet.requires_grad_(False)
    vae.eval()
    text_encoder.eval()
    controlnet.eval()
    unet.eval()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.use_adafactor:
        optimizer_class = Adafactor
    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    elif args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params = []
    params_lowlr = []
    for name, param in unet.named_parameters():
    #     if (name.find('attn1') > 0 or name.find('attn2') > 0) and (name.find('to_out') > 0) and (
    #             name.find('attn1') > 0 or (name.find('attn2') > 0 and name.find('.weight') > 0)):
    #         params.append(param)
    #     else:
    #         params_lowlr.append(param)
    #         param.requires_grad = False
        if(name.startswith('up_blocks')):
            params.append(param)

    if args.unet_layer == 'only1': # 116 layers
        params_to_optimize = [
            {'params': params[38:154], 'lr': args.learning_rate},
        ]
    elif args.unet_layer == 'only2': # 116 layers
        params_to_optimize = [
            {'params': params[154:270], 'lr': args.learning_rate},
        ]
    elif args.unet_layer == 'only3': # 114 layers
        params_to_optimize = [
            {'params': params[270:], 'lr': args.learning_rate},
        ]
    elif args.unet_layer == '1and2': # 232 layers
        params_to_optimize = [
            {'params': params[38:270], 'lr': args.learning_rate},
        ]
    elif args.unet_layer == '2and3': # 230 layers
        params_to_optimize = [
            {'params': params[154:], 'lr': args.learning_rate},
        ]

    train_dataset = make_train_dataset(args, tokenizer, accelerator)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    controlnet, train_dataloader = accelerator.prepare(
        controlnet, train_dataloader
    )
    for step, batch in enumerate(train_dataloader):
        pass

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # ----------------------------------- Imagic ---------------------------------- #
    logger.info('************* Imagic ***********')
    if args.load_finetune_from_local:
        logger.info('Loading embeddings from local ...')
        orig_emb = torch.load(os.path.join(args.finetune_path, 'orig_emb.pt'))
        emb = torch.load(os.path.join(args.finetune_path, 'emb.pt'))
    else:
        # init_image = batch["pixel_values"][0].to(dtype=weight_dtype)
        init_latent = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
        init_latent = init_latent * vae.config.scaling_factor

        if args.controlnet_load_revision == 'mlin' or args.controlnet_load_revision is None:
            orig_emb = text_encoder(batch["input_ids"], attention_mask=batch["attention_mask"])[0]
        else:
            orig_emb = text_encoder(batch["input_ids"])[0]
        emb = orig_emb.clone()
        torch.save(orig_emb, os.path.join(args.output_dir, 'orig_emb.pt'))
        torch.save(emb, os.path.join(args.output_dir, 'emb.pt'))

        # 1. Optimize the embedding
        logger.info('1. Optimize the embedding')
        unet.eval()
        emb.requires_grad = True
        lr = 0.001
        it = args.embedding_optimize_it # 500
        opt = torch.optim.Adam([emb], lr=lr)
        history = []

        pbar = tqdm(
            range(it),
            initial=0,
            desc="Optimize Steps",
            disable=not accelerator.is_local_main_process,
        )
        global_step = 0
        # wandb.define_metric("embedding/loss", step_metric="optimize_step")

        for i in pbar:
            opt.zero_grad()
            
            noise = torch.randn_like(init_latent)
            bsz = init_latent.shape[0]
            t_enc = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=init_latent.device)
            t_enc = t_enc.long()
            z = noise_scheduler.add_noise(init_latent, noise, t_enc)

            controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)

            down_block_res_samples, mid_block_res_sample = controlnet(
                z,
                t_enc,
                encoder_hidden_states=emb,
                controlnet_cond=controlnet_image,
                return_dict=False,
            )

            # Predict the noise residual
            pred_noise = unet(
                z,
                t_enc,
                encoder_hidden_states=emb,
                down_block_additional_residuals=[
                    sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                ],
                mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
            ).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(init_latent, noise, t_enc)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            loss = F.mse_loss(pred_noise.float(), target.float(), reduction="mean")
            
            loss.backward()
            logs = {"embedding/loss": loss.detach().item()}
            global_step += 1
            accelerator.log(logs)
            pbar.set_postfix({"loss": loss.item()})
            history.append(loss.item())
            opt.step()
            opt.zero_grad()

        # 2. Finetune the model
        logger.info('2. Finetune the model')
        emb.requires_grad = False
        unet.requires_grad_(True)
        unet.train()

        lr = 5e-5
        it = args.model_finetune_it # 1000
        opt = torch.optim.Adam(params_to_optimize, lr=lr)
        history = []

        # wandb.define_metric("finetune/loss", step_metric="finetune_step")

        pbar = tqdm(
            range(it),
            initial=0,
            desc="Finetune Steps",
            disable=not accelerator.is_local_main_process,
        )
        global_step = 0
        for i in pbar:
            opt.zero_grad()
            
            noise = torch.randn_like(init_latent)
            bsz = init_latent.shape[0]
            t_enc = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=init_latent.device)
            t_enc = t_enc.long()
            z = noise_scheduler.add_noise(init_latent, noise, t_enc)

            controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)

            down_block_res_samples, mid_block_res_sample = controlnet(
                z,
                t_enc,
                encoder_hidden_states=emb,
                controlnet_cond=controlnet_image,
                return_dict=False,
            )

            # Predict the noise residual
            pred_noise = unet(
                z,
                t_enc,
                encoder_hidden_states=emb,
                down_block_additional_residuals=[
                    sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                ],
                mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
            ).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(init_latent, noise, t_enc)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            loss = F.mse_loss(pred_noise.float(), target.float(), reduction="mean")
            
            loss.backward()
            logs = {"finetune/loss": loss.detach().item()}
            global_step += 1
            accelerator.log(logs)
            pbar.set_postfix({"loss": loss.item()})
            history.append(loss.item())
            opt.step()
            opt.zero_grad()

    # 3. Generate Images    
    logger.info("3. Running validation... ")
    image_logs = []

    unet.eval()
    controlnet = accelerator.unwrap_model(controlnet)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    # Reconstruct original image
    cond = load_image(args.condition_image_path)
    with torch.autocast("cuda"):
        image = pipeline(
                image=cond, prompt_embeds=emb, num_inference_steps=20, generator=generator
            ).images[0]
        image.save(f'{args.output_dir}/reconstruct_original_mask.jpg')
        image_logs.append(
                {"image": image, "mask": cond, "name": 'reconstruct_original_mask'}
        )

    cond = load_image(args.mask_path)
    with torch.autocast("cuda"):
        image = pipeline(
                image=cond, prompt_embeds=emb, num_inference_steps=20, generator=generator
            ).images[0]
        image.save(f'{args.output_dir}/reconstruct_new_mask.jpg')
        image_logs.append(
                {"image": image, "mask": cond, "name": 'reconstruct_new_mask'}
        )

    # Interpolate the embedding
    for num_inference_steps in args.num_inference_steps:
        for alpha in args.alpha:
            new_emb = alpha * orig_emb + (1 - alpha) * emb

            with torch.autocast("cuda"):
                image = pipeline(
                        image=cond, prompt_embeds=new_emb, num_inference_steps=num_inference_steps, generator=generator
                    ).images[0]
                image.save(f'{args.output_dir}/image_{num_inference_steps}_{alpha}.jpg')
                image_logs.append(
                    {"image": image, "mask": cond, "name": f'{num_inference_steps}_{alpha}'}
                )

    for tracker in accelerator.trackers:
        formatted_images = []

        for log in image_logs:
            image = log["image"]
            mask = log["mask"]
            name = log["name"]

            formatted_images.append(wandb.Image(image, caption=name))
            formatted_images.append(wandb.Image(mask, caption=name))

        tracker.log({"results": formatted_images})

    # Create the pipeline using using the trained modules and save it.
    logger.info('Saving the unet model.')
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        unet.save_pretrained(f'{args.output_dir}/unet')

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                token=args.hub_token,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == '__main__':
    args = parse_args()
    main(args)



