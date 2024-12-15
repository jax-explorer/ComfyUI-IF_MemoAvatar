import os
import folder_paths
from huggingface_hub import hf_hub_download
import logging

logger = logging.getLogger("memo")

class MemoModelManager:
    def __init__(self):
        self.models_base = folder_paths.models_dir
        self.setup_paths()
        self.ensure_model_structure()

    def setup_paths(self):
        """Initialize all required paths"""
        # Main paths
        self.memo_dir = os.path.join(self.models_base, "checkpoints", "memo")
        self.face_dir = os.path.join(self.memo_dir, "misc", "face_analysis")
        self.face_models_path = self.face_dir
        self.vocal_dir = os.path.join(self.memo_dir, "misc", "vocal_separator")
        self.wav2vec_dir = os.path.join(self.models_base, "wav2vec")
        self.emotion2vec_dir = os.path.join(self.models_base, "emotion2vec")
        self.vae_path = os.path.join(self.models_base, "vae", "sd-vae-ft-mse")
        
        # Create directories
        os.makedirs(self.memo_dir, exist_ok=True)
        os.makedirs(self.face_models_path, exist_ok=True)
        os.makedirs(self.vocal_dir, exist_ok=True)
        os.makedirs(self.wav2vec_dir, exist_ok=True)
        os.makedirs(self.emotion2vec_dir, exist_ok=True)
        
        # Set environment variables
        os.environ["MEMO_FACE_MODELS"] = self.face_models_path
        os.environ["MEMO_VOCAL_MODEL"] = os.path.join(self.vocal_dir, "Kim_Vocal_2.onnx")

    def ensure_model_structure(self):
        """Ensure all models are in place"""
        self._ensure_face_analysis_models()
        self._ensure_vocal_separator()
        self._ensure_wav2vec_models()
        self._ensure_emotion2vec_models()
        self._ensure_emotion_classifier()

    def _ensure_face_analysis_models(self):
        """Ensure face analysis models are present"""
        required_models = [
            "1k3d68.onnx",
            "2d106det.onnx",
            "face_landmarker_v2_with_blendshapes.task",
            "genderage.onnx",
            "glintr100.onnx",
            "scrfd_10g_bnkps.onnx"
        ]
        
        for model in required_models:
            target_path = os.path.join(self.face_dir, model)
            if not os.path.exists(target_path):
                try:
                    hf_hub_download(
                        repo_id="memoavatar/memo",
                        filename=f"misc/face_analysis/models/{model}",
                        local_dir=self.face_dir,
                        local_dir_use_symlinks=False
                    )
                except Exception as e:
                    logger.warning(f"Failed to download face analysis model {model}: {e}")

    def _ensure_vocal_separator(self):
        """Ensure vocal separator model is present"""
        target_path = os.path.join(self.vocal_dir, "Kim_Vocal_2.onnx")
        if not os.path.exists(target_path):
            try:
                hf_hub_download(
                    repo_id="memoavatar/memo",
                    filename="misc/vocal_separator/Kim_Vocal_2.onnx",
                    local_dir=self.vocal_dir,
                    local_dir_use_symlinks=False
                )
            except Exception as e:
                logger.warning(f"Failed to download vocal separator: {e}")

    def _ensure_wav2vec_models(self):
        """Ensure wav2vec models are present"""
        wav2vec_files = [
            "config.json",
            "preprocessor_config.json",
            "pytorch_model.bin",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "vocab.json"
        ]
        
        wav2vec_path = os.path.join(self.wav2vec_dir, "facebook", "wav2vec2-base-960h")
        os.makedirs(wav2vec_path, exist_ok=True)
        
        for file in wav2vec_files:
            target_path = os.path.join(wav2vec_path, file)
            if not os.path.exists(target_path):
                try:
                    hf_hub_download(
                        repo_id="facebook/wav2vec2-base-960h",
                        filename=file,
                        local_dir=wav2vec_path,
                        local_dir_use_symlinks=False
                    )
                except Exception as e:
                    logger.warning(f"Failed to download wav2vec file {file}: {e}")

    def _ensure_emotion2vec_models(self):
        """Ensure emotion2vec models are present"""
        emotion2vec_files = [
            "config.yaml",
            "model.pt",
            "tokens.txt"
        ]
        
        emotion2vec_path = os.path.join(self.emotion2vec_dir, "emotion2vec_plus_large")
        os.makedirs(emotion2vec_path, exist_ok=True)
        
        for file in emotion2vec_files:
            target_path = os.path.join(emotion2vec_path, file)
            if not os.path.exists(target_path):
                try:
                    hf_hub_download(
                        repo_id="emotion2vec/emotion2vec_plus_large",
                        filename=file,
                        local_dir=emotion2vec_path,
                        local_dir_use_symlinks=False
                    )
                except Exception as e:
                    logger.warning(f"Failed to download emotion2vec file {file}: {e}")

    def _ensure_emotion_classifier(self):
        """Ensure emotion classifier model is present"""
        classifier_dir = os.path.join(self.memo_dir, "misc", "audio_emotion_classifier")
        os.makedirs(classifier_dir, exist_ok=True)
        
        model_path = os.path.join(classifier_dir, "diffusion_pytorch_model.safetensors")
        if not os.path.exists(model_path):
            try:
                hf_hub_download(
                    repo_id="memoavatar/memo",
                    filename="misc/audio_emotion_classifier/diffusion_pytorch_model.safetensors",
                    local_dir=classifier_dir,
                    local_dir_use_symlinks=False
                )
            except Exception as e:
                logger.warning(f"Failed to download emotion classifier model: {e}")

    def get_model_paths(self):
        """Return dictionary of model paths"""
        wav2vec_path = os.path.join(self.wav2vec_dir, "facebook", "wav2vec2-base-960h")
        return {
            "memo_base": self.memo_dir,
            "face_models": self.face_models_path,
            "vocal_separator": os.path.join(self.vocal_dir, "Kim_Vocal_2.onnx"),
            "wav2vec": wav2vec_path,
            "emotion2vec": os.path.join(self.emotion2vec_dir, "emotion2vec_plus_large"),
            "vae": self.vae_path
        }