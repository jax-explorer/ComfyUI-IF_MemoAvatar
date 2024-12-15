# Migration from original github

1. fix runtime error
2. Automatic downloads are being fix....(WIP)
3. There are also some parameter tuning issues, under normal development. (WIP)
4. Optimize compatibility with different images

online run: https://www.comfyonline.app/explore/791e2d71-eaaf-48da-8f99-9b4f7d657ff2


====================================================================

# ComfyUI-IF_MemoAvatar
Memory-Guided Diffusion for Expressive Talking Video Generation


![demo](https://github.com/user-attachments/assets/f11caaf4-2345-4e69-b6b4-ffa895116f29)

#ORIGINAL REPO
**MEMO: Memory-Guided Diffusion for Expressive Talking Video Generation**
<br>
[Longtao Zheng](https://ltzheng.github.io)\*,
[Yifan Zhang](https://scholar.google.com/citations?user=zuYIUJEAAAAJ)\*,
[Hanzhong Guo](https://scholar.google.com/citations?user=q3x6KsgAAAAJ)\,
[Jiachun Pan](https://scholar.google.com/citations?user=nrOvfb4AAAAJ),
[Zhenxiong Tan](https://scholar.google.com/citations?user=HP9Be6UAAAAJ),
[Jiahao Lu](https://scholar.google.com/citations?user=h7rbA-sAAAAJ),
[Chuanxin Tang](https://scholar.google.com/citations?user=3ZC8B7MAAAAJ),
[Bo An](https://personal.ntu.edu.sg/boan/index.html),
[Shuicheng Yan](https://scholar.google.com/citations?user=DNuiPHwAAAAJ)
<br>
_[Project Page](https://memoavatar.github.io) | [arXiv](https://arxiv.org/abs/2412.04448) | [Model](https://huggingface.co/memoavatar/memo)_

This repository contains the example inference script for the MEMO-preview model. The gif demo below is compressed. See our [project page](https://memoavatar.github.io) for full videos.

<div style="width: 100%; text-align: center;">
    <img src="assets/demo.gif" alt="Demo GIF" style="width: 100%; height: auto;">
</div>

# ComfyUI-IF_MemoAvatar
Memory-Guided Diffusion for Expressive Talking Video Generation

## Overview
This is a ComfyUI implementation of MEMO (Memory-Guided Diffusion for Expressive Talking Video Generation), which enables the creation of expressive talking avatar videos from a single image and audio input.

## Features
- Generate expressive talking head videos from a single image
- Audio-driven facial animation
- Emotional expression transfer
- High-quality video output
![thorium_XMBCG9kbGn](https://github.com/user-attachments/assets/7d19178f-870c-429b-988e-1335fe1ba8f9)

## Installation

git clone the repo to your custom_nodes folder and then 
```bash
cd ComfyUI-IF_MemoAvatar
pip install -r requirements.txt
```
I removed xformers from the file because it needs a particular combination of pytorch on windows to work

if you are on linux you can just run 
```bash
pip install xformers 
```
for windows users if you don't have xformers on your env 
```bash
pip show xformers 
```
follow this guide to install a good comfyui environment if you don't see any version install the latest following this free guide

https://ko-fi.com/post/Installing-Triton-and-Sage-Attention-Flash-Attenti-P5P8175434

### Model Files
The models will automatically download to the following locations in your ComfyUI installation:

```bash
models/checkpoints/memo/
├── audio_proj/
├── diffusion_net/
├── image_proj/
├── misc/
│ ├── audio_emotion_classifier/
│ ├── face_analysis/
│ └── vocal_separator/
└── reference_net/
models/wav2vec/
models/vae/sd-vae-ft-mse/
models/emotion2vec/emotion2vec_plus_large/

```

Copy the faceanalisys/models models from the folder directly into faceanalisys 
just until I make sure don't just move then duplicate them cos
HF will detect empty and download them every time 
If you don't see a `models.json` or errors out create one yourself this is the content 
```bash
{
  "detection": [
    "scrfd_10g_bnkps"
  ],
  "recognition": [
    "glintr100"
  ],
  "analysis": [
    "genderage",
    "2d106det",
    "1k3d68"
  ]
}
```
and a `version.txt` containing 
`0.7.3`

![yW8hDQhnhM](https://github.com/user-attachments/assets/1c11e940-2da3-4d43-9453-cc1be06942c3)

<img src="https://count.getloli.com/get/@IF_MemoAvatar_comfy?theme=moebooru" alt=":IF_MemoAvatar_comfy" />

