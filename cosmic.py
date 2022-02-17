import os
import gc
import logging
import subprocess
from typing import *
from datetime import datetime

import torch
import torchvision
from PIL import Image
from upscaler.models import ESRGAN, ESRGANConfig
from geniverse.models import TamingDecoder
from geniverse_hub import hub_utils


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

        self.auto_param_dict_list = [
            {
                "resolution": (256, 256),
                "lr": 0.05,
                "num_iterations": 30,
                "do_upscale": False,
                "num_crops": 128,
            },
            {
                "resolution": (512, 512),
                "lr": 0.08,
                "num_iterations": 20,
                "do_upscale": False,
                "num_crops": 128,
            },
            {
                "resolution": (640, 640),
                "lr": 0.08,
                "num_iterations": 20,
                "do_upscale": True,
                "num_crops": 64,
            },
        ]

        self.u2net = hub_utils.load_from_hub("u2net")

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
        results_dir: str = "results",
    ):
        num_augmentations = max(1, int(num_augmentations / num_accum_steps))

        logging.debug(f"Using {num_augmentations} augmentations")
        logging.info(f"Using {num_accum_steps} accum steps")
        logging.info(
            f"Effective num crops of {num_accum_steps * num_augmentations}")

        assert loss_type in self.generator.supported_loss_types, f"ERROR! Loss type " \
            f"{loss_type} not recognized. " f"Only " \
            f"{' or '.join(self.generator.supported_loss_types)} supported."

        if cond_img.max() > 1:
            cond_img = cond_img / 255.

        cond_img = cond_img.to(device, )
        cond_img_size = cond_img.shape[2::]
        scale = (max(resolution)) / max(cond_img_size)

        img_resolution = [
            int((cond_img_size[0] * scale) // 16 * 16),
            int((cond_img_size[1] * scale) // 16 * 16)
        ]

        if scale != 1:
            cond_img = torch.nn.functional.interpolate(
                cond_img,
                img_resolution,
                mode="bilinear",
            )

        norm_cond_img = cond_img * 2 - 1
        norm_cond_img = torch.nn.functional.interpolate(
            cond_img,
            (200, 200),
            mode="bilinear",
        )
        mask = self.u2net.get_img_mask(norm_cond_img, ).detach().clone()
        mask = torch.nn.functional.interpolate(
            mask,
            img_resolution,
            mode="bilinear",
        )
        mask = torchvision.transforms.functional.gaussian_blur(
            img=mask,
            kernel_size=[11, 11],
        )
        torchvision.transforms.ToPILImage()(mask[0]).save(
            f"{results_dir}/mask.png", )

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

                def scale_grad(grad, ):
                    grad_size = grad.shape[2:4]

                    grad_mask = torch.nn.functional.interpolate(
                        mask,
                        grad_size,
                        mode="nearest",
                    )

                    grad.data = grad.data * grad_mask

                    return grad

                img_rec_hook = latents.register_hook(scale_grad, )

                img_rec = img_rec * mask + cond_img * (1 - mask)
                if step == 0:
                    torchvision.transforms.ToPILImage()(cond_img[0]).save(
                        f"{results_dir}/{step:04d}.png", )

                torchvision.transforms.ToPILImage()(img_rec[0]).save(
                    f"{results_dir}/{(step+1):04d}.png", )

                x_rec_stacked = self.generator.augment(
                    img_rec,
                    num_crops=num_augmentations,
                    noise_factor=aug_noise_factor,
                )

                img_logits_list = self.generator.get_clip_img_encodings(
                    x_rec_stacked, )

                # with torch.no_grad():
                #     cond_img_stacked = self.generator.augment(
                #         cond_img,
                #         num_crops=num_augmentations,
                #         noise_factor=aug_noise_factor,
                #     ).detach().clone()

                # loss += -10 * torch.cosine_similarity(
                #     x_rec_stacked,
                #     cond_img_stacked,
                # ).mean()

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
            img_rec_hook.remove()

            torch.cuda.empty_cache()
            gc.collect()

        if do_upscale:
            img_rec = self.upscaler.upscale(img_rec, ).to(
                torch.float32, device) / 255.

        return img_rec, latents

    def generate_nfts(
        self,
        num_nfts: int = 10,
    ):
        cond_img = torchvision.transforms.PILToTensor()(
            Image.open("cosmic.png"))[None, :]

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

            for _params_idx, auto_params in enumerate(
                    self.auto_param_dict_list):
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

    def generate_nfts_from_prompt(
        self,
        prompt_list: str = "",
        num_nfts: int = 10,
        cond_img: Image.Image = None,
        auto: bool = True,
        param_dict: Dict[str, Any, ] = None,
    ) -> List[Image.Image]:
        filename = f"{'-'.join(['_'.join(prompt.split()) for prompt in prompt_list], )}-{'_'.join(str(datetime.now()).split())}"
        results_dir = os.path.join("results", filename)
        os.makedirs(
            results_dir,
            exist_ok=True,
        )

        if cond_img is None:
            cond_img = Image.open("cosmic.png")

        cond_img = torchvision.transforms.PILToTensor()(cond_img, )[None, :]

        prompt_weight_list = [1 for _ in range(len(prompt_list))]

        if auto or param_dict is None:
            param_dict_list = self.auto_param_dict_list
        else:
            param_dict_list = [
                param_dict,
            ]

        nft_list = []
        for _ in range(num_nfts):
            init_step = 0
            for params_idx, auto_params in enumerate(param_dict_list):
                gen_img, _latents = self.optimize(
                    prompt_list=prompt_list,
                    prompt_weight_list=prompt_weight_list,
                    num_iterations=auto_params["num_iterations"],
                    resolution=auto_params["resolution"],
                    cond_img=cond_img,
                    device="cuda",
                    lr=auto_params["lr"],
                    loss_type="cosine_similarity",
                    num_augmentations=auto_params["num_crops"],
                    aug_noise_factor=0.11,
                    num_accum_steps=4,
                    init_step=init_step,
                    do_upscale=auto_params["do_upscale"],
                    results_dir=results_dir,
                )
                init_step += auto_params["num_iterations"]
                cond_img = gen_img.detach().clone()

            gen_img_pil = torchvision.transforms.ToPILImage()(gen_img[0])

            nft_list.append(gen_img_pil, )

            gen_img_pil.save(f"results/{filename}.png", )

            fps = 10
            cmd = ("ffmpeg -y "
                   "-r 8 "
                   f"-pattern_type glob -i '{results_dir}/0*.png' "
                   "-vcodec libx264 "
                   f"-crf {fps} "
                   "-pix_fmt yuv420p "
                   "-vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' "
                   f"results/{filename}.mp4;")

            subprocess.check_call(cmd, shell=True)

        return nft_list


if __name__ == "__main__":
    cosmic_nft = CosmicNFT()
    nft_img_list = cosmic_nft.generate_nfts()
