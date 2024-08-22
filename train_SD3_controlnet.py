#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import contextlib
import inspect
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
# from torchmetrics.functional import structural_similarity_index_measure as ssim
# from ignite.metrics import psnr
from pytorch_msssim import ssim, ms_ssim
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    StableDiffusion3ControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)

from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
from diffusers.models.transformers import SD3Transformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available,USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module, randn_tensor
from diffusers.image_processor import VaeImageProcessor


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.30.0.dev0")

logger = get_logger(__name__)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def log_validation(
    vae, text_encoder,text_encoder2,text_encoder3, tokenizer, tokenizer2, tokenizer3, transformer, controlnet, args, accelerator, weight_dtype, step, is_final_validation=False
):
    logger.info("Running validation... ")

    if not is_final_validation:
        controlnet = accelerator.unwrap_model(controlnet)
    else:
        controlnet = SD3ControlNetModel.from_pretrained(args.output_dir, torch_dtype=weight_dtype)

    pipeline = StableDiffusion3ControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=text_encoder2,
        tokenizer_2=tokenizer2,
        text_encoder_3=text_encoder3,
        tokenizer_3=tokenizer3,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if len(args.validation_image) == len(args.validation_prompt)==len(args.validation_negative_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
        validation_negative_prompts = args.validation_negative_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
        validation_negative_prompts = args.validation_negative_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
        validation_negative_prompts = args.validation_negative_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` `args.validation_prompt` and `args.validation_negative_prompt` should be checked in `parse_args`"
        )

    image_logs = []
    inference_ctx = contextlib.nullcontext() if is_final_validation else torch.autocast("cuda")

    for validation_prompt, validation_negative_prompt, validation_image in zip(validation_prompts, validation_negative_prompts, validation_images):
        validation_image = Image.open(validation_image).convert("RGB")

        images = []

        for _ in range(args.num_validation_images):
            with inference_ctx:
                image = pipeline(
                    validation_prompt,negative_prompt=validation_negative_prompt, control_image=validation_image, num_inference_steps=args.num_inference_steps, generator=generator,width=args.resolution,height=args.resolution,
                ).images[0]

            images.append(image)

        image_logs.append(
            {"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt, "validation_negative_prompt": validation_negative_prompt}
        )

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_negative_prompts = log["validation_negative_prompt"]
                validation_image = log["validation_image"]

                formatted_images = []

                formatted_images.append(np.asarray(validation_image))

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images("prompt:"+validation_prompt+"\n negative prompt:"+validation_negative_prompts, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_negative_prompt = log["validation_negative_prompt"]
                validation_image = log["validation_image"]

                formatted_images.append(wandb.Image(validation_image, caption="Controlnet conditioning"))

                for image in images:
                    image = wandb.Image(image, caption="prompt:"+validation_prompt+"\n negative prompt:"+validation_negative_prompt)
                    formatted_images.append(image)

            tracker.log({tracker_key: formatted_images})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

        return image_logs

def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    model_description = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion",
        "stable-diffusion-diffusers",
        "text-to-image",
        "diffusers",
        "controlnet",
        "diffusers-training",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=256,
        help=(
            "Maximum sequence length to use with the `prompt`."
        ),
    )
    parser.add_argument(
        "--control_guidance_start",
        type=Union[float, List[float]],
        default=0.0,
        help=(
            "The percentage of total steps at which the ControlNet starts applying."
        ),
    )
    parser.add_argument(
        "--control_guidance_end",
        type=Union[float, List[float]],
        default=1.0,
        help=(
            "The percentage of total steps at which the ControlNet stops applying."
        ),
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=0.5,
        help=(
            "Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)."
            "`guidance_scale` is defined as `w` of equation 2. of "
            "[Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >"
            "1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,"
            "usually at the expense of lower image quality."
        ),
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=28,
        help=(
            "The number of denoising steps. More denoising steps usually lead to a higher quality image at the"
            "expense of slower inference."
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--joint_attention_kwargs",
        type=Dict[str, Any],
        default=None,
        help=(
            "A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under"
            "`self.processor` in"
            "diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py)."
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--prompt_column",
        type=str,
        default="prompt",
        help="The column of the dataset containing a prompt",
    )
    parser.add_argument(
        "--prompt2_column",
        type=str,
        default="prompt2",
        help="The column of the dataset containing a prompt",
    )
    parser.add_argument(
        "--prompt3_column",
        type=str,
        default="prompt3",
        help="The column of the dataset containing a prompt",
    )
    parser.add_argument(
        "--negative_prompt_column",
        type=str,
        default="negative_prompt",
        help="The column of the dataset containing a negative prompt",
    )
    parser.add_argument(
        "--negative_prompt2_column",
        type=str,
        default="negative_prompt2",
        help="The column of the dataset containing a negative prompt2",
    )
    parser.add_argument(
        "--negative_prompt3_column",
        type=str,
        default="negative_prompt3",
        help="The column of the dataset containing a negative prompt3",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0.5,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_negative_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of negative prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default="1.0",
        help=(
            "The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added"
            "to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set"
            "the corresponding scale as a list."
        ),
    )   
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--lambda_l1",
        type=float,
        default="1.0",
        help=(
            "The weight of the L1 loss in the controlnet loss. The L1 loss is computed between the"
            " controlnet conditioning image and the generated image."
        ),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args

# Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline._get_clip_prompt_embeds
def get_clip_prompt_embeds(
    text_input_ids: torch.LongTensor, #shape (batch_size, sequence_length)
    text_encoder: CLIPTextModelWithProjection,
    device: Optional[torch.device],
    num_images_per_prompt: int = 1,
    clip_skip: Optional[int] = None,
):
    # device = device or self._execution_device
    batch_size = text_input_ids.shape[0]
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
    pooled_prompt_embeds = prompt_embeds[0]

    if clip_skip is None:
        prompt_embeds = prompt_embeds.hidden_states[-2]
    else:
        prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
    pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds, pooled_prompt_embeds

# Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline._get_t5_prompt_embeds
def get_t5_prompt_embeds(
    text_input_ids: torch.LongTensor, #shape (batch_size, sequence_length)
    device: Optional[torch.device],
    text_encoder: T5EncoderModel = None,
    num_images_per_prompt: int = 1,
    dtype: Optional[torch.dtype] = None,
    tokenizer_max_length: int = 77,
    joint_attention_dim: int = 64,
):
    batch_size = text_input_ids.shape[0]

    if text_encoder is None:
        return torch.zeros(
            (
                batch_size * num_images_per_prompt,
                tokenizer_max_length,
                joint_attention_dim,
            ),
            device=device,
            dtype=dtype,
        )

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds

# Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline.encode_prompt
def encode_prompt(
    prompt: torch.LongTensor, #shape (batch_size, sequence_length)
    text_encoders: List[Union[CLIPTextModelWithProjection, T5EncoderModel]],
    device: Optional[torch.device] = None,
    num_images_per_prompt: int = 1,
    do_classifier_free_guidance: bool = True,
    prompt_2: torch.LongTensor=None,
    prompt_3: torch.LongTensor=None,
    negative_prompt: Optional[torch.LongTensor] = None,
    negative_prompt_2: Optional[torch.LongTensor] = None,
    negative_prompt_3: Optional[torch.LongTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    clip_skip: Optional[int] = None,
    max_sequence_length: Optional[int] = 256,
    tokenizer_max_length: Optional[int] = 77,
    lora_scale: Optional[float] = None,
    joint_attention_dim: int = 64,
):
    # device = device or self._execution_device
    if prompt is not None:
        batch_size = prompt.shape[0]
    else:
        batch_size = prompt_embeds.shape[0]

    if prompt_embeds is None:
        prompt_embed, pooled_prompt_embed = get_clip_prompt_embeds(
            text_input_ids=prompt,
            text_encoder=text_encoders[0],
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            clip_skip=clip_skip,
        )
        prompt_2_embed, pooled_prompt_2_embed = get_clip_prompt_embeds(
            text_input_ids=prompt_2,
            text_encoder=text_encoders[1],
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            clip_skip=clip_skip,
        )
        clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

        t5_prompt_embed = get_t5_prompt_embeds(
            text_input_ids=prompt_3,
            text_encoder=text_encoders[2],
            num_images_per_prompt=num_images_per_prompt,
            tokenizer_max_length=tokenizer_max_length,
            joint_attention_dim=joint_attention_dim,
            device=device,
        )

        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
        )

        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
        pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

    if do_classifier_free_guidance and negative_prompt_embeds is None:
        if prompt is not None and type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif batch_size != negative_prompt.shape[0]:
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {negative_prompt.shape[0]}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )

        negative_prompt_embed, negative_pooled_prompt_embed = get_clip_prompt_embeds(
            negative_prompt,
            text_encoder=text_encoders[0],
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            clip_skip=None,
        )
        negative_prompt_2_embed, negative_pooled_prompt_2_embed = get_clip_prompt_embeds(
            negative_prompt_2,
            text_encoder=text_encoders[1],
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            clip_skip=None,
        )
        negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)

        t5_negative_prompt_embed = get_t5_prompt_embeds(
            prompt=negative_prompt_3,
            text_encoder=text_encoders[2],
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        negative_clip_prompt_embeds = torch.nn.functional.pad(
            negative_clip_prompt_embeds,
            (0, t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1]),
        )

        negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
        negative_pooled_prompt_embeds = torch.cat(
            [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
        )

    if text_encoders[0] is not None:
        if USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(text_encoders[0], lora_scale)

    if text_encoders[1] is not None:
        if USE_PEFT_BACKEND:
            # Retrieve the original scale by scaling back the LoRA layers
            unscale_lora_layers(text_encoders[1], lora_scale)

    return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds



def make_train_dataset(args, tokenizers, accelerator, do_classifier_free_guidance, joint_attention_dim):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        if args.train_data_dir is not None:
            dataset = load_dataset(
                args.train_data_dir,
                cache_dir=args.cache_dir,
                trust_remote_code=True
            )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )
        
    if args.conditioning_image_column is None:
        conditioning_image_column = column_names[1]
        logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
    else:
        conditioning_image_column = args.conditioning_image_column
        if conditioning_image_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_column` value '{args.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.prompt_column is None:
        prompt_column = column_names[2]
        logger.info(f"prompt column defaulting to {prompt_column}")
    else:
        prompt_column = args.prompt_column
        if prompt_column not in column_names:
            raise ValueError(
                f"`--prompt_column` value '{args.prompt_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )
        
    if args.prompt2_column is None:
        prompt2_column = column_names[3]
        logger.info(f"prompt2 column defaulting to {prompt2_column}")
    else:
        prompt2_column = args.prompt2_column
        if prompt2_column not in column_names:
            # raise ValueError(
            #     f"`--prompt2_column` value '{args.prompt2_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            # )
            prompt2_column=prompt_column
        
    if args.prompt3_column is None:
        prompt3_column = column_names[4]
        logger.info(f"prompt3 column defaulting to {prompt3_column}")
    else:
        prompt3_column = args.prompt3_column
        if prompt3_column not in column_names:
            # raise ValueError(
            #     f"`--prompt3_column` value '{args.prompt3_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            # )
            prompt3_column=prompt_column
                
    if args.negative_prompt_column is None:
        negative_prompt_column = column_names[5]
        logger.info(f"negative prompt column defaulting to {negative_prompt_column}")
    else:
        negative_prompt_column = args.negative_prompt_column
        if negative_prompt_column not in column_names:
            raise ValueError(
                f"`--negative_prompt_column` value '{args.negative_prompt_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.negative_prompt2_column is None:
        negative_prompt2_column = column_names[6]
        logger.info(f"negative prompt2 column defaulting to {negative_prompt2_column}")
    else:
        negative_prompt2_column = args.negative_prompt2_column
        if negative_prompt2_column not in column_names:
            # raise ValueError(
            #     f"`--negative_prompt2_column` value '{args.negative_prompt2_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            # )
            negative_prompt2_column = negative_prompt_column

    if args.negative_prompt3_column is None:
        negative_prompt3_column = column_names[7]
        logger.info(f"negative prompt3 column defaulting to {negative_prompt3_column}")
    else:
        negative_prompt3_column = args.negative_prompt3_column
        if negative_prompt3_column not in column_names:
            # raise ValueError(
            #     f"`--negative_prompt3_column` value '{args.negative_prompt3_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            # )
            negative_prompt3_column=negative_prompt_column

    def tokenize_prompts(examples, is_negative:bool, is_train=True):
        prompts = []
        prompts2 = []
        prompts3 = []
        for i in range(len(examples[prompt_column])):
            if is_negative:
                prompt=examples[negative_prompt_column][i]
                prompt = prompt or ""
                prompt2=examples[negative_prompt2_column][i]
                prompt2 = prompt2 or prompt
                prompt3=examples[negative_prompt3_column][i]
                prompt3 = prompt3 or prompt

                # normalize str to list
                prompt = [prompt] if isinstance(prompt, str) else prompt
                prompt2 = (
                    [prompt2] if isinstance(prompt2, str) else prompt2
                )
                prompt3 = (
                    [prompt3] if isinstance(prompt3, str) else prompt3
                )
            else:
                prompt=examples[prompt_column][i]
                prompt = [prompt] if isinstance(prompt, str) else prompt

                prompt2=examples[prompt2_column][i]
                prompt2 = prompt2 or prompt
                prompt2 = [prompt2] if isinstance(prompt2, str) else prompt2

                prompt3=examples[prompt3_column][i]
                prompt3 = prompt3 or prompt
                prompt3 = [prompt3] if isinstance(prompt3, str) else prompt3

            if random.random() < args.proportion_empty_prompts:
                prompts.append("")
                prompts2.append("")
                prompts3.append("")
            elif isinstance(prompt, str) and isinstance(prompt2, str) and isinstance(prompt3, str) :
                prompts.append(prompt)
                prompts2.append(prompt2)
                prompts3.append(prompt3)
            elif isinstance(prompt, (list, np.ndarray)) and isinstance(prompt2, (list, np.ndarray)) and isinstance(prompt3, (list, np.ndarray)):
                # take a random caption if there are multiple
                prompts.append(random.choice(prompt) if is_train else prompt[0])
                prompts2.append(random.choice(prompt2) if is_train else prompt2[0])
                prompts3.append(random.choice(prompt3) if is_train else prompt3[0])
            else:
                raise ValueError(
                    f"prompt column `{prompt_column}` should contain either strings or lists of strings."
                )
        prompt_inputs = tokenizers[0](
            prompts, max_length=tokenizers[0].model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        prompt_inputs_ids = prompt_inputs.input_ids
        untruncated_prompt_ids = tokenizers[0](prompts, padding="longest", return_tensors="pt").input_ids
        if untruncated_prompt_ids.shape[-1] >= prompt_inputs_ids.shape[-1] and not torch.equal(prompt_inputs_ids, untruncated_prompt_ids):
            removed_text = tokenizers[0].batch_decode(untruncated_prompt_ids[:, args.max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {args.max_sequence_length} tokens: {removed_text}"
            )

        prompt2_inputs = tokenizers[1](
            prompts2, max_length=tokenizers[1].model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        prompt2_inputs_ids = prompt2_inputs.input_ids
        untruncated_prompt2_ids = tokenizers[1](prompts2, padding="longest", return_tensors="pt").input_ids
        if untruncated_prompt2_ids.shape[-1] >= prompt2_inputs_ids.shape[-1] and not torch.equal(prompt2_inputs_ids, untruncated_prompt2_ids):
            removed_text = tokenizers[1].batch_decode(untruncated_prompt2_ids[:, args.max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {args.max_sequence_length} tokens: {removed_text}"
            )

        prompt3_inputs = tokenizers[2](
            prompts3,
            padding="max_length",
            max_length=args.max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt3_inputs_ids = prompt3_inputs.input_ids
        untruncated_prompt3_ids = tokenizers[2](prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_prompt3_ids.shape[-1] >= prompt3_inputs_ids.shape[-1] and not torch.equal(prompt3_inputs_ids, untruncated_prompt3_ids):
            removed_text = tokenizers[2].batch_decode(untruncated_prompt3_ids[:, tokenizers[2].model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {args.max_sequence_length} tokens: {removed_text}"
            )

        return prompt_inputs_ids, prompt2_inputs_ids, prompt3_inputs_ids

    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            #TODO: ADD DATA AUGMENTATION
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        images = [image_transforms(image) for image in images]

        conditioning_images = [image.convert("RGB") for image in examples[conditioning_image_column]]
        conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]

        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_images
        examples["prompt"], examples["prompt2"], examples["prompt3"] = tokenize_prompts(examples, is_negative=False)
        examples["negative_prompt"], examples["negative_prompt2"],examples["negative_prompt3"] = tokenize_prompts(examples, is_negative=True)

        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    prompt = torch.stack([example["prompt"] for example in examples])
    prompt2 = torch.stack([example["prompt2"] for example in examples])
    prompt3 = torch.stack([example["prompt3"] for example in examples])

    negative_prompt = torch.stack([example["negative_prompt"] for example in examples])
    negative_prompt2 = torch.stack([example["negative_prompt2"] for example in examples])
    negative_prompt3 = torch.stack([example["negative_prompt3"] for example in examples])

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "prompt": prompt,
        "prompt2": prompt2,
        "prompt3": prompt3,
        "negative_prompt": negative_prompt,
        "negative_prompt2": negative_prompt2,
        "negative_prompt3": negative_prompt3,
    }

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

def prepare_image(
    image,
    width,
    height,
    batch_size,
    num_images_per_prompt,
    device,
    dtype,
    image_processor,
    do_classifier_free_guidance=False,
    guess_mode=False,
):
    if isinstance(image, torch.Tensor):
        pass
    else:
        image = image_processor.preprocess(image, height=height, width=width)
    image_batch_size = image.shape[0]

    if image_batch_size == 1:
        repeat_by = batch_size
    else:
        # image batch size is the same as prompt batch size
        repeat_by = num_images_per_prompt

    image = image.repeat_interleave(repeat_by, dim=0)

    image = image.to(device=device, dtype=dtype)

    if do_classifier_free_guidance and not guess_mode:
        image = torch.cat([image] * 2)

    return image

# Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3.StableDiffusion3Pipeline.prepare_latents
def prepare_latents(
    batch_size,
    num_channels_latents,
    height,
    width,
    dtype,
    device,
    vae_scale_factor,
    generator=None,
    latents=None,
):
    if latents is not None:
        return latents.to(device=device, dtype=dtype)

    shape = (
        batch_size,
        num_channels_latents,
        int(height) // vae_scale_factor,
        int(width) // vae_scale_factor,
    )

    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    return latents

def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

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
    do_classifier_free_guidance=args.guidance_scale > 1

    # Load the tokenizer
    #TODO: make sure there is fp16 revision
    tokenizer=CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )
    tokenizer2=CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        use_fast=False,
    )
    tokenizer3=T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_3",
        revision=args.revision,
        use_fast=False,
    )

    # import correct text encoder class
    # Load scheduler and models
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    text_encoder3 = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=args.revision, variant=args.variant
    )
    transformer=SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    controlnet = SD3ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)

    vae_scale_factor = (2 ** (len(vae.config.block_out_channels) - 1))
    tokenizer_max_length = (
            tokenizer.model_max_length if tokenizer is not None else 77
        )
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

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
                load_model = SD3ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder2.requires_grad_(False)
    text_encoder3.requires_grad_(False)
    controlnet.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            transformer.enable_xformers_memory_efficient_attention()
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

    if unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = controlnet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataset = make_train_dataset(args, [tokenizer, tokenizer2, tokenizer3], accelerator, do_classifier_free_guidance,transformer.config.joint_attention_dim)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    # controlnet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder2.to(accelerator.device, dtype=weight_dtype)
    text_encoder3.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    height = width = args.resolution

    # align format for control guidance
    if not isinstance(args.control_guidance_start, list) and isinstance(args.control_guidance_end, list):
        control_guidance_start = len(args.control_guidance_end) * [args.control_guidance_start]
    elif not isinstance(args.control_guidance_end, list) and isinstance(args.control_guidance_start, list):
        control_guidance_end = len(args.control_guidance_start) * [args.control_guidance_end]
    elif not isinstance(args.control_guidance_start, list) and not isinstance(args.control_guidance_end, list):
        mult = len(controlnet.nets) if isinstance(controlnet, SD3MultiControlNetModel) else 1
        control_guidance_start, control_guidance_end = (
            mult * [args.control_guidance_start],
            mult * [args.control_guidance_end],
        )

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                control_image = prepare_image(
                    image=batch["conditioning_pixel_values"],
                    width=width,
                    height=height,
                    batch_size=args.train_batch_size,
                    num_images_per_prompt=1,
                    device=accelerator.device,
                    dtype=transformer.dtype,
                    image_processor=image_processor,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=False,
                )
                height, width = control_image.shape[-2:]
                
                control_image = vae.encode(control_image).latent_dist.sample()
                control_image = control_image * vae.config.scaling_factor

                (
                    prompt_embeds,
                    negative_prompt_embeds,
                    pooled_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = encode_prompt(
                    prompt=batch["prompt"],
                    prompt_2=batch["prompt2"],
                    prompt_3=batch["prompt3"],
                    negative_prompt=batch["negative_prompt"],
                    negative_prompt_2=batch["negative_prompt2"],
                    negative_prompt_3=batch["negative_prompt3"],
                    text_encoders=[text_encoder, text_encoder2, text_encoder3],
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    device=accelerator.device,
                    max_sequence_length=args.max_sequence_length,
                    tokenizer_max_length=tokenizer_max_length,
                )
                if do_classifier_free_guidance:
                    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                    pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

                controlnet_pooled_projections = torch.zeros_like(pooled_prompt_embeds)
                
                # 4. Prepare timesteps
                timesteps, num_inference_steps = retrieve_timesteps(scheduler, args.num_inference_steps, accelerator.device)
                num_warmup_steps = max(len(timesteps) - num_inference_steps * scheduler.order, 0)
                num_timesteps = len(timesteps)

                # 5. Prepare latent variables
                # num_channels_latents = transformer.config.in_channels
                # latents = prepare_latents(
                #     total_batch_size,
                #     num_channels_latents,
                #     height,
                #     width,
                #     prompt_embeds.dtype,
                #     accelerator.device,
                #     vae_scale_factor=vae_scale_factor
                # )
                latents=vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()

                # Create tensor stating which controlnets to keep
                controlnet_keep = []
                for i in range(len(timesteps)):
                    keeps = [
                        1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                        for s, e in zip(control_guidance_start, control_guidance_end)
                    ]
                    controlnet_keep.append(keeps[0] if isinstance(controlnet, SD3ControlNetModel) else keeps)
                # 7. Denoising loop
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                    timestep = t.expand(latent_model_input.shape[0])

                    if isinstance(controlnet_keep[i], list):
                        cond_scale = [c * s for c, s in zip(args.controlnet_conditioning_scale, controlnet_keep[i])]
                    else:
                        controlnet_cond_scale = args.controlnet_conditioning_scale
                        if isinstance(controlnet_cond_scale, list):
                            controlnet_cond_scale = controlnet_cond_scale[0]
                        cond_scale = controlnet_cond_scale * controlnet_keep[i]

                    # controlnet(s) inference
                    control_block_samples = controlnet(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=controlnet_pooled_projections,
                        joint_attention_kwargs=args.joint_attention_kwargs,
                        controlnet_cond=control_image,
                        conditioning_scale=cond_scale,
                        return_dict=False,
                    )[0]

                    noise_pred = transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        block_controlnet_hidden_states=[sample.to(dtype=weight_dtype) for sample in control_block_samples],
                        joint_attention_kwargs=args.joint_attention_kwargs,
                        return_dict=False,
                    )[0]

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents_dtype = latents.dtype
                    latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                    if latents.dtype != latents_dtype:
                        if torch.backends.mps.is_available():
                            # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                            latents = latents.to(latents_dtype)


                latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor

                model_pred = vae.decode(latents, return_dict=False)[0]
                # image = image_processor.postprocess(image, output_type="pil")
                # print(model_pred)

                target=batch["pixel_values"].to(dtype=weight_dtype)
                
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                # ssim_loss = 1-ssim(noise_pred.float(), target.float(), data_range=255, size_average=True)
                # l1_loss = F.l1_loss(model_pred.float(), target.float())


                # loss = ssim_loss + args.lambda_l1 * l1_loss


                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        image_logs = log_validation(
                            vae,
                            text_encoder,
                            text_encoder2,
                            text_encoder3,
                            tokenizer,
                            tokenizer2,
                            tokenizer3,
                            transformer,
                            controlnet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = unwrap_model(controlnet)
        controlnet.save_pretrained(args.output_dir)

        # Run a final round of validation.
        image_logs = None
        if args.validation_prompt is not None:
            image_logs = log_validation(
                vae=vae,
                text_encoder=text_encoder,
                text_encoder2=text_encoder2,
                text_encoder3=text_encoder3,
                tokenizer=tokenizer,
                tokenizer2=tokenizer2,
                tokenizer3=tokenizer3,
                transformer=transformer,
                controlnet=None,
                args=args,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                step=global_step,
                is_final_validation=True,
            )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
