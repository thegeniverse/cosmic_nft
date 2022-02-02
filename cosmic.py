import gc
import logging
from typing import *

import torch
import torchvision
from PIL import Image
from upscaler.models import ESRGAN, ESRGANConfig
from geniverse.models import TamingDecoder


class CosmicNFT:
    def __init__(self, ):
        clip_model_name_list = [
            "ViT-B/32",
            "ViT-B/16",
        ]

        self.generator = TamingDecoder(
            device="cuda",
            clip_model_name_list=clip_model_name_list,
        )

        esrgan_model_name = "RealESRGAN_x4plus"
        tile = 256
        esrgan_config = ESRGANConfig(
            model_name=esrgan_model_name,
            tile=tile,
        )

        self.upscaler = ESRGAN(esrgan_config, )

        self.param_dict_list = [
            {
                "resolution": (400, 400),
                "lr": 0.3,
                "num_iterations": 30,
                "do_upscale": False,
            },
            {
                "resolution": (900, 900),
                "lr": 0.2,
                "num_iterations": 20,
                "do_upscale": True,
            },
        ]

    def optimize(
        self,
        prompt_list: List[List[str]],
        prompt_weight_list: List[List[float]],
        num_iterations: int,
        resolution: Tuple[int],
        cond_img=None,
        device: str = "cuda",
        lr: float = 0.5,
        loss_type="spherical_distance",
        num_augmentations: int = 16,
        aug_noise_factor: float = 0.11,
        num_accum_steps: int = 8,
        init_step: int = 0,
        do_upscale: bool = False,
    ):
        num_augmentations = max(1, int(num_augmentations / num_accum_steps))

        logging.debug(f"Using {num_augmentations} augmentations")
        logging.info(f"Using {num_accum_steps} accum steps")
        logging.info(
            f"Effective num crops of {num_accum_steps * num_augmentations}")

        assert loss_type in self.generator.supported_loss_types, f"ERROR! Loss type " \
            f"{loss_type} not recognized. " f"Only " \
            f"{' or '.join(self.generator.supported_loss_types)} supported."

        cond_img_size = cond_img.shape[2::]
        scale = (max(resolution) // 16 * 16) / max(cond_img_size)
        if scale != 1:
            img_resolution = [
                int(cond_img_size[0] * scale),
                int(cond_img_size[1] * scale)
            ]

            cond_img = torch.nn.functional.interpolate(
                cond_img,
                img_resolution,
                mode="bilinear",
            )

        latents = self.generator.get_latents_from_img(cond_img, )

        latents = latents.to(device)
        latents = torch.nn.Parameter(latents)

        optimizer = torch.optim.AdamW(
            params=[latents],
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.1,
        )

        for step in range(init_step, init_step + num_iterations):
            optimizer.zero_grad()

            logging.info(f"step {step}")

            for _num_accum in range(num_accum_steps):
                loss = 0

                img_rec = self.generator.get_img_from_latents(latents, )

                x_rec_stacked = self.generator.augment(
                    img_rec,
                    num_crops=num_augmentations,
                    noise_factor=aug_noise_factor,
                )

                img_logits_list = self.generator.get_clip_img_encodings(
                    x_rec_stacked, )

                for prompt, prompt_weight in zip(prompt_list,
                                                 prompt_weight_list):
                    text_logits_list = self.generator.get_clip_text_encodings(
                        prompt, )

                    for img_logits, text_logits in zip(img_logits_list,
                                                       text_logits_list):
                        text_logits = text_logits.clone().detach()
                        if loss_type == 'cosine_similarity':
                            clip_loss = -10 * torch.cosine_similarity(
                                text_logits, img_logits).mean()

                        if loss_type == "spherical_distance":
                            clip_loss = (text_logits - img_logits).norm(
                                dim=-1).div(2).arcsin().pow(2).mul(2).mean()

                        loss += prompt_weight * clip_loss

                loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            torch.cuda.empty_cache()
            gc.collect()

        return img_rec, latents

    def generate_nfts(
        self,
        num_nfts: int = 10,
    ):
        cond_img = torchvision.transforms.PILToTensor()(
            Image.open("cosmic.png"))[None, :]
        cond_img = (cond_img / 255.) * 2 - 1

        with open("cosmic.txt", "r") as f:
            prompt_list = f.readlines()

        prompt_weight_list = [
            1,
            1.5,
        ]

        for prompt in prompt_list:
            prompt_list = [
                "cosmic baloon",
                prompt,
            ]

            for _params_idx, auto_params in enumerate(self.param_dict_list):
                gen_img, latents = self.optimize(
                    prompt_list=prompt_list,
                    prompt_weight_list=prompt_weight_list,
                    num_iterations=auto_params["num_iterations"],
                    resolution=auto_params["resolution"],
                    cond_img=cond_img,
                    device="cuda",
                    lr=auto_params["lr"],
                    loss_type="spherical_distance",
                    num_augmentations=256,
                    aug_noise_factor=0.11,
                    num_accum_steps=4,
                    init_step=0,
                    do_upscale=auto_params["do_upscale"],
                )

            img_rec = self.upscaler.upscale(img_rec, )

            torchvision.transforms.ToPILImage()(gen_img[0]).save(
                f"results/{prompt}.png", )


if __name__ == "__main__":
    cosmic_nft = CosmicNFT()
    nft_img_list = cosmic_nft.generate_nfts()
