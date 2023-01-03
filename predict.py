import os
from typing import List

import torch
from diffusers import UnCLIPPipeline
from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")

        model_id = "kakaobrain/karlo-v1-alpha"
        MODEL_CACHE = "diffusers-cache"
        self.pipe = UnCLIPPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to("cuda")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="a high-resolution photograph of a big red frog on a green leaf",
        ),
        num_images_per_prompt: int = Input(
            description="Number of images to output", choices=[1, 4], default=1
        ),
        prior_num_inference_steps: int = Input(
            description="The number of denoising steps for the prior. More denoising steps usually lead to a higher quality image at the expense of slower inference",
            default=25,
        ),
        decoder_num_inference_steps: int = Input(
            description="The number of denoising steps for the decoder. More denoising steps usually lead to a higher quality image at the expense of slower inference",
            default=25,
        ),
        super_res_num_inference_steps: int = Input(
            description="The number of denoising steps for super resolution. More denoising steps usually lead to a higher quality image at the expense of slower inference",
            default=7,
        ),
        prior_guidance_scale: float = Input(
            description="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality",
            default=4.0,
        ),
        decoder_guidance_scale: float = Input(
            description="Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality",
            default=8.0,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        generator = torch.Generator("cuda").manual_seed(seed)

        output = self.pipe(
            prompt=[prompt] * num_images_per_prompt,
            prior_num_inference_steps=prior_num_inference_steps,
            decoder_num_inference_steps=decoder_num_inference_steps,
            super_res_num_inference_steps=super_res_num_inference_steps,
            prior_guidance_scale=prior_guidance_scale,
            decoder_guidance_scale=decoder_guidance_scale,
            generator=generator,
        )
        output_paths = []

        for i, sample in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
