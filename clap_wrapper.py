import sys

import torch
from huggingface_hub import hf_hub_download
from pathlib import Path
from transformers import ClapModel, ClapProcessor

from config import config

models = dict()
LOCAL_PATH = "./emotional/clap-htsat-fused"
if not Path(LOCAL_PATH).joinpath('pytorch_model.bin').exists():
    if config.mirror.lower() == "openi":
        import openi
        openi.model.download_model(
            "Stardust_minus/Bert-VITS2", 'clap-htsat-fused', "./emotional"
        )
    else:
        hf_hub_download(
            'laion/clap-htsat-fused', 'pytorch_model.bin', local_dir=LOCAL_PATH, local_dir_use_symlinks=False
        )
processor = ClapProcessor.from_pretrained(LOCAL_PATH)


def get_clap_audio_feature(audio_data, device=config.bert_gen_config.device):
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"
    if device not in models.keys():
        if config.webui_config.fp16_run:
            models[device] = ClapModel.from_pretrained(
                LOCAL_PATH, torch_dtype=torch.float16
            ).to(device)
        else:
            models[device] = ClapModel.from_pretrained(LOCAL_PATH).to(device)
    with torch.no_grad():
        inputs = processor(
            audios=audio_data, return_tensors="pt", sampling_rate=48000
        ).to(device)
        emb = models[device].get_audio_features(**inputs).float()
    return emb.T


def get_clap_text_feature(text, device=config.bert_gen_config.device):
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"
    if device not in models.keys():
        if config.webui_config.fp16_run:
            models[device] = ClapModel.from_pretrained(
                LOCAL_PATH, torch_dtype=torch.float16
            ).to(device)
        else:
            models[device] = ClapModel.from_pretrained(LOCAL_PATH).to(device)
    with torch.no_grad():
        inputs = processor(text=text, return_tensors="pt").to(device)
        emb = models[device].get_text_features(**inputs).float()
    return emb.T
