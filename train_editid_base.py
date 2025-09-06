import argparse
import itertools
import logging
import math
import os
import random
from safetensors.torch import save_file
import datasets
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from copy import deepcopy
import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from einops import rearrange

from flux.util import (
    SamplingOptions,
    load_ae,
    load_clip,
    load_flow_model,
    load_flow_model_quintized,
    load_t5,
)
import gc

import cv2
import insightface
import torch
import torch.nn as nn
from facexlib.parsing import init_parsing_model
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from huggingface_hub import hf_hub_download, snapshot_download
from insightface.app import FaceAnalysis
from safetensors.torch import load_file
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import normalize, resize

from eva_clip import create_model_and_transforms
from eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from editid.encoders_transformer import IDFormer, PerceiverAttentionCA
from editid.utils import img2tensor, tensor2img
from image_datasets.dataset import pulid_loader
from flux.sampling import get_schedule, prepare
from editid.utils import resize_numpy_image_long

if is_wandb_available():
    import wandb
logger = get_logger(__name__, log_level="INFO")


def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'


class Pulid_flux(torch.nn.Module):
    def __init__(self, dit, pulid_ca, pulid_encoder, device, weight_dtype=torch.bfloat16, onnx_provider='gpu'):
        super().__init__()
        self.transformer = dit
        self.device = device
        self.weight_dtype = weight_dtype

        self.pulid_encoder = pulid_encoder
        self.pulid_ca = pulid_ca

        # preprocessors
        # face align and parsing
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            device=self.device,
        )
        self.face_helper.face_parse = None
        self.face_helper.face_parse = init_parsing_model(model_name='bisenet', device=self.device)
        # clip-vit backbone
        model, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True)
        model = model.visual
        self.clip_vision_model = model.to(self.device, dtype=self.weight_dtype)
        eva_transform_mean = getattr(self.clip_vision_model, 'image_mean', OPENAI_DATASET_MEAN)
        eva_transform_std = getattr(self.clip_vision_model, 'image_std', OPENAI_DATASET_STD)
        if not isinstance(eva_transform_mean, (list, tuple)):
            eva_transform_mean = (eva_transform_mean,) * 3
        if not isinstance(eva_transform_std, (list, tuple)):
            eva_transform_std = (eva_transform_std,) * 3
        self.eva_transform_mean = eva_transform_mean
        self.eva_transform_std = eva_transform_std
        # antelopev2
        # snapshot_download('DIAMONIK7777/antelopev2', local_dir='models/antelopev2')
        providers = ['CPUExecutionProvider'] if onnx_provider == 'cpu' \
            else ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.app = FaceAnalysis(name='antelopev2', root='./models/insightface', providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.handler_ante = insightface.model_zoo.get_model(
            './models/insightface/antelopev2/glintr100.onnx',
            providers=providers)
        self.handler_ante.prepare(ctx_id=0)

        gc.collect()
        torch.cuda.empty_cache()

        self.load_pretrain()

        # other configs
        self.debug_img_list = []

    def components_to_device(self, device):
        # everything but pulid_ca
        self.face_helper.face_det = self.face_helper.face_det.to(device)
        self.face_helper.face_parse = self.face_helper.face_parse.to(device)
        self.clip_vision_model = self.clip_vision_model.to(device)
        self.pulid_encoder = self.pulid_encoder.to(device)

    def load_pretrain(self, pretrain_path=None, version='v0.9.0'):
        # -----------------------------------------------------------------------------------------------------------
        # hf_hub_download('guozinan/PuLID', f'pulid_flux_{version}.safetensors', local_dir='models')
        ckpt_path = f'models/pulid_flux_{version}.safetensors'
        if pretrain_path is not None:
            ckpt_path = pretrain_path
        state_dict = load_file(ckpt_path)
        state_dict_dict = {}
        for k, v in state_dict.items():
            module = k.split('.')[0]
            state_dict_dict.setdefault(module, {})
            new_k = k[len(module) + 1:]
            state_dict_dict[module][new_k] = v

        for module in state_dict_dict:
            print(f'loading from {module}')
            getattr(self, module).load_state_dict(state_dict_dict[module], strict=True)

        del state_dict
        del state_dict_dict

    def to_gray(self, img):
        x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        x = x.repeat(1, 3, 1, 1)
        return x

    # @torch.no_grad()
    def get_id_embedding(self, image, cal_uncond=False):
        """
        Args:
            image: numpy rgb image, range [0, 255]
        """
        # import pdb;
        # pdb.set_trace()
        self.face_helper.clean_all()
        self.debug_img_list = []
        image = image.cpu().numpy()
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # get insightface embeddings
        id_ante_embedding = None
        for size in [(size, size) for size in range(640, 256, -64)]:
            self.app.det_model.input_size = size  # 假设 self.app 是 InsightFace 模型
            face_info = self.app.get(image_bgr)
            if face_info:
                # 只选择面积最大的人脸
                face_info = \
                sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
                # id_ante_embedding = torch.from_numpy(face_info['embedding']).unsqueeze(0).to(device,
                #                                                                              dtype=torch.float32)
                id_ante_embedding = face_info["embedding"]
                self.debug_img_list.append(
                    image_bgr[
                    int(face_info['bbox'][1]): int(face_info['bbox'][3]),
                    int(face_info['bbox'][0]): int(face_info['bbox'][2]),
                    ]
                )
                break
        else:
            # 未检测到人脸，记录警告
            logging.warning("No face detected in the image after trying multiple scales")
            id_ante_embedding = None
        # # get antelopev2 embedding
        # face_info = self.app.get(image_bgr)
        # if len(face_info) > 0:
        #     face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[
        #         -1
        #     ]  # only use the maximum face
        #     id_ante_embedding = face_info['embedding']
        #     self.debug_img_list.append(
        #         image[
        #         int(face_info['bbox'][1]): int(face_info['bbox'][3]),
        #         int(face_info['bbox'][0]): int(face_info['bbox'][2]),
        #         ]
        #     )
        # else:
        #     id_ante_embedding = None

        # using facexlib to detect and align face
        self.face_helper.read_image(image_bgr)
        self.face_helper.get_face_landmarks_5(only_center_face=True)
        self.face_helper.align_warp_face()
        if len(self.face_helper.cropped_faces) == 0:
            raise RuntimeError('facexlib align face fail')
        align_face = self.face_helper.cropped_faces[0]
        # incase insightface didn't detect face
        if id_ante_embedding is None:
            print('fail to detect face using insightface, extract embedding on align face')
            id_ante_embedding = self.handler_ante.get_feat(align_face)

        id_ante_embedding = torch.from_numpy(id_ante_embedding).to(self.device, self.weight_dtype)
        if id_ante_embedding.ndim == 1:
            id_ante_embedding = id_ante_embedding.unsqueeze(0)

        # parsing
        input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
        input = input.to(self.device)
        parsing_out = self.face_helper.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)
        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        bg = sum(parsing_out == i for i in bg_label).bool()
        white_image = torch.ones_like(input)
        # only keep the face features
        face_features_image = torch.where(bg, white_image, self.to_gray(input))
        self.debug_img_list.append(tensor2img(face_features_image, rgb2bgr=False))

        # transform img before sending to eva-clip-vit
        face_features_image = resize(face_features_image, self.clip_vision_model.image_size, InterpolationMode.BICUBIC)
        face_features_image = normalize(face_features_image, self.eva_transform_mean, self.eva_transform_std)
        id_cond_vit, id_vit_hidden = self.clip_vision_model(
            face_features_image.to(self.weight_dtype), return_all_features=False, return_hidden=True, shuffle=False
        )
        id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)
        id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)

        id_cond = torch.cat([id_ante_embedding, id_cond_vit], dim=-1)

        #  上面的都是特征提取模块
        id_embedding = self.pulid_encoder(id_cond, id_vit_hidden)

        if not cal_uncond:
            return id_embedding, None

        id_uncond = torch.zeros_like(id_cond)
        id_vit_hidden_uncond = []
        for layer_idx in range(0, len(id_vit_hidden)):
            id_vit_hidden_uncond.append(torch.zeros_like(id_vit_hidden[layer_idx]))
        uncond_id_embedding = self.pulid_encoder(id_uncond, id_vit_hidden_uncond)

        return id_embedding, uncond_id_embedding

    def forward(self, noisy_model_input, timesteps, guidance, pooled_prompt_embeds, encoder_hidden_states, text_ids,
                latent_image_ids, id_embeddings):
        # 推理和前向是两件事
        noise_pred = self.transformer(
            img=noisy_model_input,
            img_ids=latent_image_ids,
            txt=encoder_hidden_states,
            txt_ids=text_ids,
            timesteps=timesteps,
            y=pooled_prompt_embeds,
            guidance=guidance,
            id=id_embeddings,
            id_weight=1.0,
            aggressive_offload=False)
        return noise_pred


def get_models(name: str, device: torch.device, offload: bool):
    t5 = load_t5(device, max_length=128)
    clip = load_clip(device)
    model = load_flow_model(name, device="cpu" if offload else device)
    # model.eval()
    ae = load_ae(name, device="cpu" if offload else device)
    return model, ae, t5, clip


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default="./EditIDv2/train_configs/editid_insert_multi_data.yaml",
        help="path to config",
    )
    # parser.add_argument("--fp8", action="store_true",default=False, help="use flux-dev-fp8 model")
    args = parser.parse_args()

    return args.config


def main():
    args = OmegaConf.load(parse_args())
    # is_schnell = args.model_name == "flux-schnell"
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
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
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # import pdb; pdb.set_trace()
    dit, vae, t5, clip = get_models(
        name=args.model_name, device=accelerator.device, offload=False)

    double_interval = 2
    single_interval = 4

    num_ca = 19 // double_interval + 38 // single_interval
    if 19 % double_interval != 0:
        num_ca += 1
    if 38 % single_interval != 0:
        num_ca += 1
    # id嵌入模块
    pulid_ca = nn.ModuleList([
        PerceiverAttentionCA().to(accelerator.device, torch.bfloat16) for _ in range(num_ca)
    ])

    dit.pulid_ca = pulid_ca
    dit.pulid_double_interval = double_interval
    dit.pulid_single_interval = single_interval

    # id融合模块
    pulid_encoder = IDFormer().to(accelerator.device, torch.bfloat16)

    editid = Pulid_flux(
        dit, pulid_ca, pulid_encoder,
        device=accelerator.device,
        weight_dtype=torch.bfloat16,
    )

    pulid_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    t5.requires_grad_(False)
    clip.requires_grad_(False)
    dit.requires_grad_(False)
    dit = dit.to(torch.float32)
    dit.train()

    for param in editid.pulid_ca.parameters():
        param.requires_grad = True

    print(sum([p.numel() for p in dit.parameters() if p.requires_grad]) / 1000000, 'M parameters')

    optimizer_cls = torch.optim.AdamW
    params_to_opt = itertools.chain(editid.pulid_ca.parameters())
    optimizer = optimizer_cls(
        params_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataloader = pulid_loader(**args.local_config)
    train_dataloader_len = len(train_dataloader)
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(train_dataloader_len / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    global_step = 0
    first_epoch = 0

    editid, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        editid, optimizer, deepcopy(train_dataloader), lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    num_update_steps_per_epoch = math.ceil(train_dataloader_len / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"test": None})

    timesteps = get_schedule(
        999,
        (1024 // 8) * (1024 // 8) // 4,
        shift=True,
    )
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    def load_checkpoint_and_resume():
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint != "latest":
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                dirs = os.listdir(args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1].split('.')[0]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                accelerator.print(
                    f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                args.resume_from_checkpoint = None
                initial_global_step = 0
                first_epoch = 0
                step_in_epoch = 0
            else:
                accelerator.print(f"Resuming from checkpoint {path}")
                ckpt_path = os.path.join(args.output_dir, path)
                state_dict = load_file(ckpt_path)
                state_dict_dict = {}
                for k, v in state_dict.items():
                    module = k.split('.')[0]
                    state_dict_dict.setdefault(module, {})
                    new_k = k[len(module) + 1:]
                    state_dict_dict[module][new_k] = v

                for module in state_dict_dict:
                    getattr(editid, module).load_state_dict(state_dict_dict[module], strict=True)

                # 假设检查点文件名包含 global_step，例如 checkpoint-1000.safetensors
                initial_global_step = int(path.split("-")[1].split('.')[0])
                first_epoch = 0  # 可根据需要恢复
                step_in_epoch = 0  # 可根据需要恢复

        else:
            initial_global_step = 0
            first_epoch = 0
            step_in_epoch = 0
        return initial_global_step, first_epoch, step_in_epoch

    global_step, first_epoch, step_in_epoch = load_checkpoint_and_resume()

    def check_nan(loss):
        if torch.isnan(loss):
            accelerator.print("Loss is NaN. Loading checkpoint and restarting training.")
            return True
        return False

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader, start=step_in_epoch):
            with accelerator.accumulate(editid):
                img = batch['images']
                prompts = batch['texts']
                with torch.no_grad():
                    x_1 = vae.encode(img.to(accelerator.device).to(torch.float32))
                    inp = prepare(t5=t5, clip=clip, img=x_1, prompt=prompts)
                    x_1 = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

                bs = img.shape[0]
                t = torch.tensor([timesteps[random.randint(0, 999)]]).to(accelerator.device)
                x_0 = torch.randn_like(x_1).to(accelerator.device)
                x_t = (1 - t) * x_1 + t * x_0
                bsz = x_1.shape[0]
                guidance_vec = torch.full((x_t.shape[0],), 1, device=x_t.device, dtype=x_t.dtype)

                with torch.no_grad():
                    id_image = batch['id_images']
                    if id_image is not None:
                        try:
                            id_image = resize_numpy_image_long(id_image, 1024)
                            id_embeddings, uncond_id_embeddings = editid.get_id_embedding(id_image)
                        except RuntimeError as e:
                            print("id image processing failed:", e)
                            id_embeddings = None
                            uncond_id_embeddings = None
                        # id_image = resize_numpy_image_long(id_image, 1024)
                        # id_embeddings, uncond_id_embeddings = editid.get_id_embedding(id_image)
                    else:
                        id_embeddings = None
                        uncond_id_embeddings = None

                torch.cuda.empty_cache()
                model_pred = editid(noisy_model_input=x_t.to(weight_dtype),
                                   latent_image_ids=inp['img_ids'].to(weight_dtype),
                                   encoder_hidden_states=inp['txt'].to(weight_dtype),
                                   text_ids=inp['txt_ids'].to(weight_dtype),
                                   pooled_prompt_embeds=inp['vec'].to(weight_dtype),
                                   timesteps=t.to(weight_dtype),
                                   guidance=guidance_vec.to(weight_dtype),
                                   id_embeddings=id_embeddings.to(weight_dtype),
                                   )

                loss = F.mse_loss(model_pred.float(), (x_0 - x_1).float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                if global_step > 1 and check_nan(loss):
                    # Load the checkpoint and restart training
                    global_step, first_epoch, step_in_epoch = load_checkpoint_and_resume()
                    progress_bar.close()
                    progress_bar = tqdm(range(0, args.max_train_steps),
                                        initial=global_step, desc="Steps",
                                        disable=not accelerator.is_local_main_process, )
                    break  # Exit the inner loop to restart training

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(dit.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

            # 在保存检查点时
            if global_step % args.checkpointing_steps == 0 and global_step != 0 and accelerator.is_local_main_process:
                # 处理检查点数量限制
                if args.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(args.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1].split('.')[0]))
                    if len(checkpoints) >= args.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]
                        accelerator.print(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        accelerator.print(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                            os.remove(removing_checkpoint)

                # 保存权重
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.safetensors")
                unwrapped_pulid = accelerator.unwrap_model(editid)

                # 合并 pulid_encoder 和 pulid_ca 的状态字典
                merged_state_dict = {}
                # pulid_encoder 权重
                encoder_state_dict = unwrapped_pulid.pulid_encoder.state_dict()
                merged_state_dict.update({f"pulid_encoder.{k}": v for k, v in encoder_state_dict.items()})
                # pulid_ca 权重（ModuleList）
                for idx, ca_module in enumerate(unwrapped_pulid.pulid_ca):
                    ca_state_dict = ca_module.state_dict()
                    merged_state_dict.update({f"pulid_ca.{idx}.{k}": v for k, v in ca_state_dict.items()})

                # 保存为 safetensors 格式
                save_file(merged_state_dict, save_path)
                accelerator.print(f"Saved state to {save_path}")

            # 在训练结束保存最终权重
            if global_step >= args.max_train_steps and accelerator.is_local_main_process:
                if (step + 1) % args.gradient_accumulation_steps != 0:
                    optimizer.step()
                    optimizer.zero_grad()

                # 保存最终权重
                save_path = os.path.join(args.output_dir, f"final-checkpoint-{global_step}.safetensors")
                unwrapped_pulid = accelerator.unwrap_model(editid)

                # 合并 pulid_encoder 和 pulid_ca 的状态字典
                merged_state_dict = {}
                # pulid_encoder 权重
                encoder_state_dict = unwrapped_pulid.pulid_encoder.state_dict()
                merged_state_dict.update({f"pulid_encoder.{k}": v for k, v in encoder_state_dict.items()})
                # pulid_ca 权重（ModuleList）
                for idx, ca_module in enumerate(unwrapped_pulid.pulid_ca):
                    ca_state_dict = ca_module.state_dict()
                    merged_state_dict.update({f"pulid_ca.{idx}.{k}": v for k, v in ca_state_dict.items()})

                # 保存为 safetensors 格式
                save_file(merged_state_dict, save_path)
                accelerator.print(f"Finally saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
        step_in_epoch = 0

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr
    main()

# accelerate launch --config_file default_config.yaml  --main_process_port 23456 train_flux_editid.py
# CUDA_VISIBLE_DEVICES=5,6 accelerate launch --config_file default_config.yaml --num_processes 2 train_flux_editid.py
