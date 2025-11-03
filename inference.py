# the original file is inference02.py
# this is the modified file to receive two inputs (one for the image and one for the mask) and generate the output image
"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os
from typing import List
import torch
import glob
import re

from typing import Callable, List, Optional, Union

from diffusers import DiffusionPipeline, DDIMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline, StableDiffusionPipelineOutput
import torch
from torchvision import transforms

from mask_encoder2 import MaskEncoder
from PIL import Image


class MaskAndImageDiffusionPipeline(object):
    def __init__(self, pipeline, mask_encoder, mask_path,num_of_assets, *args, **kwargs):
        self.mask_encoder = mask_encoder
        self.pipeline = pipeline
        self.mask_path = mask_path
        instance_mask_path = glob.glob(os.path.join(self.mask_path, f"mask*.png"))
        print('load image ', instance_mask_path)
        masks=len(instance_mask_path) #added for many masks
        self.curr_mask = []
       # self.strategy = strategy
        self.num_of_assets = num_of_assets

        for i in range(0, masks):
            curr_mask = Image.open(instance_mask_path[i])

            self.mask_transforms = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )
            curr_mask = self.mask_transforms(curr_mask)[0, None, None, ...]
            self.curr_mask.append(curr_mask)

        self.curr_mask = torch.cat(self.curr_mask)
        # inject empty tokens for mask conditioning based on the strategy
        # if self.strategy == 2:
        #     new_masks = torch.zeros((len(self.curr_mask),) + curr_mask.shape[1:]).float()
        #     new_masks[tokens_ids_to_use] = example["instance_masks"]
        #     example["instance_masks"] = new_masks
        
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        strategy: int = 0, 
        num_of_assets: int = 0,
        
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height = height or self.pipeline.unet.config.sample_size * self.pipeline.vae_scale_factor
        width = width or self.pipeline.unet.config.sample_size * self.pipeline.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.pipeline.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.pipeline._execution_device
        
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds = self.pipeline._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # if self.strategy == 2:
        #     assets = [int(re.findall("\d", asset)[0]) for asset in re.findall("<asset\d>", prompt[0])]
        #     new_masks = torch.zeros((self.num_of_assets,) + self.curr_mask[0].shape).float()
        #     if len(assets) == 1:
        #         new_masks[asset] = self.curr_mask[0] #leo
        #         # new_masks[asset] = self.curr_mask[asset] 
        #     else:
        #         for idx, asset in enumerate(assets):
        #             new_masks[idx] = self.curr_mask[idx]
        #     self.curr_mask = new_masks
       
        # if self.strategy == 3:
        #     assets = [int(re.findall("\d", asset)[0]) for asset in re.findall("<asset\d>", prompt[0])]
        #     new_masks = torch.zeros((self.num_of_assets,) + self.curr_mask[0].shape).float()
        #     if len(assets) == 1:
        #         new_masks[asset] = self.curr_mask[0]
        #         # new_masks[asset] = self.curr_mask[asset]
        #     else:
        #         for idx, asset in enumerate(assets):
        #             new_masks[asset] = self.curr_mask[idx]
        #     self.curr_mask = new_masks

        # if self.strategy == 4:
        #     assets = [int(re.findall("\d", asset)[0]) for asset in re.findall("<asset\d>", prompt[0])]
        #     new_masks = torch.zeros((self.num_of_assets,) + self.curr_mask[0].shape).float()
        #     if len(assets) == 1:
        #         new_masks[0] = self.curr_mask[0]
        #     else:
        #         for idx, asset in enumerate(assets):
        #             new_masks[asset] = self.curr_mask[idx]

        #     self.curr_mask = new_masks

        # inject mask conditioning
        if hasattr(self, 'mask_encoder'):
            curr_mask_tensor = self.curr_mask[None].to(device)
            mask_embeds = self.mask_encoder(curr_mask_tensor).type(prompt_embeds.dtype)

            if len(prompt_embeds) > len(mask_embeds):
                neg_mask_embeds = torch.zeros_like(mask_embeds)
                mask_embeds = torch.cat([neg_mask_embeds, mask_embeds])
            prompt_embeds = self.mask_encoder.merge(prompt_embeds, mask_embeds)

        # 4. Prepare timesteps
        self.pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.pipeline.scheduler.timesteps


        # 5. Prepare latent variables
        num_channels_latents = self.pipeline.unet.in_channels
        latents = self.pipeline.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.pipeline.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.pipeline.scheduler.order
        with self.pipeline.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.pipeline.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.pipeline.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.pipeline.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.pipeline.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # 8. Post-processing
        image = self.pipeline.decode_latents(latents)

        # 9. Run safety checker
        image, has_nsfw_concept = self.pipeline.run_safety_checker(image, device, prompt_embeds.dtype)

        # 10. Convert to PIL
        if output_type == "pil":
            image = self.pipeline.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


class BreakASceneInference:
    def __init__(self):
        self._parse_args()
        self._load_pipeline()

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_path", type=str, required=True)
        parser.add_argument(
            "--prompt", type=str, default="a photo of <asset0> at the beach"
        )
        parser.add_argument("--output_path", type=str, default="outputs/result1.jpg")

        parser.add_argument("--mask_path", type=str, default="") #added for many masks

        parser.add_argument("--AssetNumber", type=int, default=0)

        parser.add_argument("--device", type=str, default="cuda")
       
        parser.add_argument("--strategy", type=int, default=0)
        parser.add_argument("--num_of_assets", type=int, default=0)

        self.args = parser.parse_args()

    def _load_pipeline(self):
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.args.model_path,
            torch_dtype=torch.float16,
        )
        self.pipeline.to(self.args.device)  

        #mask_encoder_s = torch.load(os.path.join(self.args.model_path, "mask_encoder/mask_encoder.pth"))
        mask_encoder = MaskEncoder()
        #mask_encoder.load_state_dict(mask_encoder_s)
        mask_encoder = mask_encoder.to(self.args.device)

        self.my_pipeline = MaskAndImageDiffusionPipeline(
            # unet=self.pipeline.unet,
            # text_encoder=self.pipeline.text_encoder,
            # tokenizer=self.pipeline.tokenizer,
            # scheduler=self.pipeline.scheduler,
            # feature_extractor=self.pipeline.feature_extractor,
            # safety_checker=self.pipeline.safety_checker,
            # vae=self.pipeline.vae,
            pipeline=self.pipeline,
            mask_encoder=mask_encoder,
            mask_path=self.args.mask_path, #added for many masks
            #strategy=self.args.strategy,
            num_of_assets=self.args.num_of_assets,
            #mask_path2=self.args.mask_path2,

        )
        #self.my_pipeline = self.my_pipeline.to(device="cuda")
        #self.my_pipeline.torch_dtype = torch.float16

        self.pipeline.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.pipeline.to(self.args.device)

    @torch.no_grad()
    def infer_and_save(self, prompts):
        # images = self.my_pipeline(prompts).images
        # images[0].save(self.args.output_path)
        
        import re
        sanitized_prompt = re.sub(r'[\/:*?"<>|]', '_', self.args.prompt)
        
        for i in range(0,5):    
            images = self.my_pipeline(prompts).images
            #images[i].save(self.args.output_path)
            
            #images[0].save(os.path.join(self.args.output_path, f"img{i}.jpg"))

            images[0].save(os.path.join(self.args.output_path, f"{sanitized_prompt}_{i}.jpg"))

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    break_a_scene_inference = BreakASceneInference()
    break_a_scene_inference.infer_and_save(
        prompts=[break_a_scene_inference.args.prompt]
    )
    