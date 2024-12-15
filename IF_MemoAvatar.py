import os
import torch
import numpy as np
import torchaudio
from PIL import Image
import logging
from tqdm import tqdm
import time
from contextlib import contextmanager
from packaging import version

import folder_paths
import comfy.model_management
from diffusers import FlowMatchEulerDiscreteScheduler

from memo.pipelines.video_pipeline import VideoPipeline
from memo.utils.audio_utils import extract_audio_emotion_labels, preprocess_audio, resample_audio
from memo.utils.vision_utils import preprocess_image, tensor_to_video
from memo_model_manager import MemoModelManager

logger = logging.getLogger("memo")

class IF_MemoAvatar:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "audio": ("STRING", {"default": ""}),
                "reference_net": ("MODEL",),
                "diffusion_net": ("MODEL",),
                "vae": ("VAE",),
                "image_proj": ("IMAGE_PROJ",),
                "audio_proj": ("AUDIO_PROJ",),
                "emotion_classifier": ("EMOTION_CLASSIFIER",),
                "resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "num_frames_per_clip": ("INT", {"default": 16, "min": 1, "max": 32}),
                "fps": ("INT", {"default": 30, "min": 1, "max": 60}),
                "inference_steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg_scale": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 100.0}),
                "seed": ("INT", {"default": 42}),
                "output_name": ("STRING", {"default": "memo_video"})
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "status")
    FUNCTION = "generate"
    CATEGORY = "ImpactFrames💥🎞️/MemoAvatar"

    def __init__(self):
        self.device = comfy.model_management.get_torch_device()
        # Use bfloat16 if available, fallback to float16
        self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        
        # Initialize model manager and get paths
        self.model_manager = MemoModelManager()
        self.paths = self.model_manager.get_model_paths()
        
        # Verify xformers availability
        if hasattr(torch.backends, 'xformers'):
            import xformers
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning("xFormers 0.0.16 may have issues. Consider updating to 0.0.17+")

    def generate(self, image, audio, reference_net, diffusion_net, vae, image_proj, audio_proj, 
                emotion_classifier, resolution=512, num_frames_per_clip=16, fps=30, 
                inference_steps=20, cfg_scale=3.5, seed=42, output_name="memo_video"):
        try:
            # Memory optimizations
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
                autocast = torch.cuda.amp.autocast
            else:
                @contextmanager
                def autocast():
                    yield

            num_init_past_frames = 2
            num_past_frames = 16

            # Better error handling for audio
            if "wav" not in audio.lower():
                logger.warning("MEMO might not generate full-length video for non-wav audio file.")

            # Save input image temporarily
            temp_dir = folder_paths.get_temp_directory()
            temp_image = os.path.join(temp_dir, f"ref_image_{time.time()}.png")
            
            try:
                # Convert ComfyUI image format to PIL
                if isinstance(image, torch.Tensor):
                    image = image.cpu().numpy()
                if image.ndim == 4:
                    image = image[0]
                image = Image.fromarray((image * 255).astype(np.uint8))
                image.save(temp_image)
                face_models_path = self.paths["face_models"]
                print(f"face_models path: {face_models_path}")
                # Process image with our models
                pixel_values, face_emb = preprocess_image(
                    face_analysis_model=self.paths["face_models"],
                    image_path=temp_image,
                    image_size=resolution,
                )
            finally:
                if os.path.exists(temp_image):
                    os.remove(temp_image)
    
            # Set up audio cache directory
            cache_dir = os.path.join(folder_paths.get_temp_directory(), "memo_audio_cache")
            os.makedirs(cache_dir, exist_ok=True)

            # Process audio path
            input_audio_path = os.path.join(folder_paths.get_input_directory(), audio)
            resampled_path = os.path.join(cache_dir, f"{os.path.splitext(os.path.basename(audio))[0]}-16k.wav")
            resampled_path = resample_audio(input_audio_path, resampled_path)

            # Process audio
            audio_emb, audio_length = preprocess_audio(
                wav_path=resampled_path,
                num_generated_frames_per_clip=num_frames_per_clip,
                fps=fps,
                wav2vec_model=self.paths["wav2vec"],
                vocal_separator_model=self.paths["vocal_separator"],
                cache_dir=cache_dir,
                device=str(self.device)
            )

            # Extract emotion 
            audio_emotion, num_emotion_classes = extract_audio_emotion_labels(
                model=self.paths["memo_base"],
                wav_path=resampled_path,
                emotion2vec_model=self.paths["emotion2vec"],
                audio_length=audio_length,
                device=str(self.device)
            )

            # Model optimizations
            vae.requires_grad_(False).eval()
            reference_net.requires_grad_(False).eval()
            diffusion_net.requires_grad_(False).eval()
            image_proj.requires_grad_(False).eval()
            audio_proj.requires_grad_(False).eval()

            # Enable memory efficient attention
            if hasattr(torch.backends, 'xformers'):
                reference_net.enable_xformers_memory_efficient_attention()
                diffusion_net.enable_xformers_memory_efficient_attention()

            # Create pipeline with optimizations
            noise_scheduler = FlowMatchEulerDiscreteScheduler()
            with torch.inference_mode():
                pipeline = VideoPipeline(
                    vae=vae,
                    reference_net=reference_net,
                    diffusion_net=diffusion_net,
                    scheduler=noise_scheduler,
                    image_proj=image_proj,
                )
                pipeline.to(device=self.device, dtype=self.dtype)

            # Generate video frames with memory optimizations
            video_frames = []
            num_clips = audio_emb.shape[0] // num_frames_per_clip
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            for t in tqdm(range(num_clips), desc="Generating video clips"):
                # Clear cache at the start of each iteration
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                if len(video_frames) == 0:
                    past_frames = pixel_values.repeat(num_init_past_frames, 1, 1, 1)
                    past_frames = past_frames.to(dtype=pixel_values.dtype, device=pixel_values.device)
                    pixel_values_ref_img = torch.cat([pixel_values, past_frames], dim=0)
                else:
                    past_frames = video_frames[-1][0]
                    past_frames = past_frames.permute(1, 0, 2, 3)
                    past_frames = past_frames[0 - num_past_frames:]
                    past_frames = past_frames * 2.0 - 1.0
                    past_frames = past_frames.to(dtype=pixel_values.dtype, device=pixel_values.device)
                    pixel_values_ref_img = torch.cat([pixel_values, past_frames], dim=0)

                pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)
                
                # Process audio in smaller chunks if needed
                audio_tensor = (
                    audio_emb[
                        t * num_frames_per_clip : min(
                            (t + 1) * num_frames_per_clip, audio_emb.shape[0]
                        )
                    ]
                    .unsqueeze(0)
                    .to(device=audio_proj.device, dtype=audio_proj.dtype)
                )
                
                with torch.inference_mode():
                    audio_tensor = audio_proj(audio_tensor)

                    audio_emotion_tensor = audio_emotion[
                        t * num_frames_per_clip : min(
                            (t + 1) * num_frames_per_clip, audio_emb.shape[0]
                        )
                    ]

                    pipeline_output = pipeline(
                        ref_image=pixel_values_ref_img,
                        audio_tensor=audio_tensor,
                        audio_emotion=audio_emotion_tensor,
                        emotion_class_num=num_emotion_classes,
                        face_emb=face_emb,
                        width=resolution,
                        height=resolution,
                        video_length=num_frames_per_clip,
                        num_inference_steps=inference_steps,
                        guidance_scale=cfg_scale,
                        generator=generator,
                    )

                video_frames.append(pipeline_output.videos)

            video_frames = torch.cat(video_frames, dim=2)
            video_frames = video_frames.squeeze(0)
            video_frames = video_frames[:, :audio_length]

            # Save video
            timestamp = time.strftime('%Y%m%d-%H%M%S')
            video_name = f"{output_name}_{timestamp}.mp4"
            output_dir = folder_paths.get_output_directory()
            video_path = os.path.join(output_dir, video_name)

            tensor_to_video(video_frames, video_path, input_audio_path, fps=fps)
            return (video_path, f"✅ Video saved as {video_name}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            return (folder_paths.get_output_directory(), f"❌ Error: {str(e)}")

# Node mappings
NODE_CLASS_MAPPINGS = {
    "IF_MemoAvatar": IF_MemoAvatar
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IF_MemoAvatar": "IF MemoAvatar 🗣️"
}