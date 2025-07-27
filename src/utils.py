import io
import os
import json
import random
from prettytable import PrettyTable

import qrcode
from pyzbar.pyzbar import decode

from sklearn import metrics
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
import cv2

import torch
import torchvision.transforms as tforms
from bm3d import bm3d_rgb

from diffusers import DDIMInverseScheduler
from datasets import load_dataset

from skimage.metrics import structural_similarity as ssim
import lpips

from pytorch_fid.fid_score import *
from compressai.zoo import bmshj2018_hyperprior, cheng2020_anchor #bmshj2018_factorized

# ====================================================================================================
# [Global variables]
device = "cuda"
hw_latent = 64
shape = (1, 4, hw_latent, hw_latent)
w_seed = 7433 # TREE :)
RADIUS = 14

# [Center-aware design] 44x44 center area
start = 10
end = 54 # 64-10 = hw_latent-start
center_slice = (slice(None), slice(None), slice(start, end), slice(start, end))

# [Tree-Ring] hyperparameters
w_channel = 3
TREE_WATERMARK_CHANNEL = [w_channel]
# [RingID] hyperparameters
RADIUS_CUTOFF = 3
USE_ROUNDER_RING = True
HETER_WATERMARK_CHANNEL = [0]
RING_WATERMARK_CHANNEL = [w_channel]
RINGID_WATERMARK_CHANNEL = sorted(HETER_WATERMARK_CHANNEL + RING_WATERMARK_CHANNEL) # [0,3]
# [HSQR] hyperparameters
HSQR_WATERMARK_CHANNEL = [w_channel]
assert TREE_WATERMARK_CHANNEL == HSQR_WATERMARK_CHANNEL, "HSQR and Tree-Ring have the same channel in the paper."
qr_version = 1 # 21x21 module(cell)
box_size = 2 # HSQR
delta = 0

# [Task - Identification] hyperparameters
wm_capacity = 2**(RADIUS-RADIUS_CUTOFF)
assert wm_capacity == 2048

# ====================================================================================================
# [Watermark Masks]
# circle mask
def circle_mask(size: int, r=16, x_offset=0, y_offset=0):
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    # y = y[::-1]
    # This original tree-ring code is wrong since (0,0) is on the top-left corner. (not bottom-left)
    # Plus, RingID's Y-center adjustment with -1 offset is done with this.
    return ((x - x0)**2 + (y-y0)**2)<= r**2

# ring mask
class RounderRingMask:
    def __init__(self, size=65, r_out=RADIUS):
        assert size >= 3
        self.size = size
        self.r_out = r_out

        num_rings = r_out
        zero_bg_freq = torch.zeros(size, size)
        center = size // 2
        center_x, center_y = center, center
        # center_x, center_y = center + x_offset, center - y_offset

        ring_vector = torch.tensor([(200 - i*4) * (-1)**i for i in range(num_rings)])
        zero_bg_freq[center_x, center_y:center_y+num_rings] = ring_vector
        zero_bg_freq = zero_bg_freq[None, None, ...]
        self.ring_vector_np = ring_vector.numpy()

        res = torch.zeros(360, size, size)
        res[0] = zero_bg_freq
        for angle in range(1, 360):
            zero_bg_freq_rot = tforms.functional.rotate(zero_bg_freq, angle=angle)
            res[angle] = zero_bg_freq_rot

        res = res.numpy()
        self.res = res
        self.pure_bg = np.zeros((size, size))
        for x in range(size):
            for y in range(size):
                values, count = np.unique(res[:, x, y],  return_counts=True)
                if len(count) > 2:
                    self.pure_bg[x, y] = values[count == max(count[values!=0])][0]
                elif len(count) == 2:
                    self.pure_bg[x, y] = values[values!=0][0]
        
    def get_ring_mask(self, r_out, r_in):
        # get mask from pure_bg
        assert r_out <= self.r_out
        if r_in - 1 < 0:
            right_end = 0  # None, to take the center
        else:
            right_end = r_in - 1
        cand_list = self.ring_vector_np[r_out-1:right_end:-1]
        mask = np.isin(self.pure_bg, cand_list)
        if self.size % 2:
            mask = mask[:self.size-1, :self.size-1]  # [64, 64]
        return mask

if USE_ROUNDER_RING:
    mask_obj = RounderRingMask(size=65, r_out=RADIUS)
    def ring_mask(size=64, r_out=RADIUS, r_in=RADIUS_CUTOFF):
        assert size == 64
        return mask_obj.get_ring_mask(r_out=r_out, r_in=r_in)  
else:
    def ring_mask(size=64, r_out=RADIUS, r_in=RADIUS_CUTOFF):
        outer_mask = circle_mask(size=size, r=r_out)
        inner_mask = circle_mask(size=size, r=r_in)
        return outer_mask & (~(inner_mask))

# [Tree-Ring] Circular mask (full-bodied center)
single_channel_tree_watermark_mask = torch.tensor(circle_mask(size=shape[-1], r=RADIUS)) # (64,64)
# [RingID] Ring-watermark mask (empty center)
single_channel_ring_watermark_mask = torch.tensor(ring_mask(size=shape[-1], r_out=RADIUS, r_in=RADIUS_CUTOFF)) # (64,64)
# [RingID] Heterogeneous-watermark mask
if len(HETER_WATERMARK_CHANNEL) > 0:
    single_channel_heter_watermark_mask = torch.tensor(ring_mask(size=shape[-1], r_out=RADIUS, r_in=RADIUS_CUTOFF)) # (64,64)
    heter_watermark_region_mask = single_channel_heter_watermark_mask.unsqueeze(0).repeat(len(HETER_WATERMARK_CHANNEL), 1, 1).to(device) # (C_H, 64,64)

# [get_distance - input] watermark_region_mask for detection process
watermark_region_mask_tree = single_channel_tree_watermark_mask[None, ...].to(device) # (1,64,64) - cuda
watermark_region_mask_ringid = torch.stack([
    single_channel_ring_watermark_mask if idx in RING_WATERMARK_CHANNEL else single_channel_heter_watermark_mask 
    for idx in RINGID_WATERMARK_CHANNEL]).to(device)  # # (C_R+C_H,64,64) - cuda
watermark_region_mask_hstr = torch.stack([
    single_channel_heter_watermark_mask, 
    single_channel_tree_watermark_mask]).to(device) # # (C_R+C_H,64,64) - cuda

# [입력 마스크 준비] inject_wm의 입력 변수
tree_masks = torch.zeros(shape, dtype=torch.bool) # (1,4,64,64)
ringid_masks = torch.zeros(shape, dtype=torch.bool) # (1,4,64,64)
tree_masks[:, TREE_WATERMARK_CHANNEL] = single_channel_tree_watermark_mask # (64,64)
ringid_masks[:, RING_WATERMARK_CHANNEL] = single_channel_ring_watermark_mask # (64,64)
ringid_masks[:, HETER_WATERMARK_CHANNEL] = single_channel_heter_watermark_mask # (64,64)

# ====================================================================================================
# [Random Seed]
def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)

# [Generation] Sampling zT ~ N(0,I)
@torch.no_grad()
def get_random_latents(pipe, batch_size=1, gen_seed=None, resolution=512):
    if gen_seed:
        g = torch.Generator(device=pipe.device).manual_seed(gen_seed)
        return pipe.prepare_latents(batch_size, pipe.unet.in_channels, resolution, resolution, pipe.unet.dtype, pipe.device, g) # (1,4,64,64)
    return pipe.prepare_latents(batch_size, pipe.unet.in_channels, resolution, resolution, pipe.unet.dtype, pipe.device, None) # (1,4,64,64)

# [Text Dataset]
def get_text_dataset(dataset_id):
    if dataset_id == "coco":
        with open("text_dataset/coco/meta_data.json") as f:
            dataset = json.load(f)
        meta_data = dataset["annotations"] # (dataset["images"], dataset["annotations"])
        prompt_key = "caption"
        gt_folder = "text_dataset/coco/ground_truth"
    elif dataset_id == "Gustavo":
        with open("text_dataset/Gustavo/prompts.json") as f:
            meta_data = json.load(f)
        prompt_key = "Prompt"
        gt_folder = None
    elif dataset_id == "DB1k":
        with open("text_dataset/DiffusionDB/metadata_1k.json") as f:
            meta_data = json.load(f)
        prompt_key = "prompt"
        gt_folder = None
    return meta_data, prompt_key, gt_folder

# [Fourier transforms]
def fft(input_tensor):
    assert len(input_tensor.shape) == 4
    return torch.fft.fftshift(torch.fft.fft2(input_tensor), dim=(-1, -2))

def ifft(input_tensor):
    assert len(input_tensor.shape) == 4
    return torch.fft.ifft2(torch.fft.ifftshift(input_tensor, dim=(-1, -2)))

@torch.no_grad()
def rfft(input_tensor):
    assert len(input_tensor.shape) == 4
    return torch.fft.fftshift(torch.fft.rfft2(input_tensor, dim=(-2,-1)), dim=-2)

@torch.no_grad()
def irfft(input_tensor):
    assert len(input_tensor.shape) == 4
    return torch.fft.irfft2(torch.fft.ifftshift(input_tensor, dim=-2), dim=(-2,-1), s=(input_tensor.shape[-2],input_tensor.shape[-2]))

# ====================================================================================================
# [Watermark Patterns]
# SFW - Symmetric Fourier Watermark Enforcement (TR -> HSTR)
@torch.no_grad()
def enforce_hermitian_symmetry(freq_tensor):
    B, C, H, W = freq_tensor.shape # fftshifted frequency (complex tensor) - center (32,32)
    assert H == W, "H != W"
    freq_tensor = freq_tensor.clone()
    freq_tensor_tmp = freq_tensor.clone()
    # DC point (no imaginary)
    freq_tensor[:, :, H//2, W//2] = torch.real(freq_tensor_tmp[:, :, H//2, W//2])
    if H % 2 == 0: # Even
        # Nyquist Points (no imaginary)
        freq_tensor[:, :, 0, 0] = torch.real(freq_tensor_tmp[:, :, 0, 0])
        freq_tensor[:, :, H//2, 0] = torch.real(freq_tensor_tmp[:, :, H//2, 0])  # (32, 0)
        freq_tensor[:, :, 0, W//2] = torch.real(freq_tensor_tmp[:, :, 0, W//2])  # (0, 32)
    
        # Nyquist axis - conjugate
        freq_tensor[:, :, 0, 1:W//2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, 0, W//2+1:], dims=[2]))
        freq_tensor[:, :, H//2, 1:W//2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, H//2, W//2+1:], dims=[2]))
        freq_tensor[:, :, 1:H//2, 0] = torch.conj(torch.flip(freq_tensor_tmp[:, :, H//2+1:, 0], dims=[2]))
        freq_tensor[:, :, 1:H//2, W//2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, H//2+1:, W//2], dims=[2]))
        # Square quadrants - conjugate
        freq_tensor[:, :, 1:H//2, 1:W//2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, H//2+1:, W//2+1:], dims=[2, 3]))
        freq_tensor[:, :, H//2+1:, 1:W//2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, 1:H//2, W//2+1:], dims=[2, 3]))
    else: # Odd
        # Nyquist axis - conjugate
        freq_tensor[:, :, H//2, 0:W//2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, H//2, W//2+1:], dims=[2]))
        freq_tensor[:, :, 0:H//2, W//2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, H//2+1:, W//2], dims=[2]))
        # Square quadrants - conjugate
        freq_tensor[:, :, 0:H//2, 0:W//2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, H//2+1:, W//2+1:], dims=[2, 3]))
        freq_tensor[:, :, H//2+1:, 0:W//2] = torch.conj(torch.flip(freq_tensor_tmp[:, :, 0:H//2, W//2+1:], dims=[2, 3]))
    return freq_tensor

# TR - tree-ring pattern: constant ring values from normal distribution N(0,1).
@torch.no_grad()
def make_Fourier_treering_pattern(pipe, shape, w_seed=999999, resolution=512, 
        hs=False, center=False, heter=False):
    assert shape[-1] == shape[-2] # 64==64
    device = pipe.device
    g = torch.Generator(device=device).manual_seed(w_seed)
    gt_init = pipe.prepare_latents(1, pipe.unet.in_channels, resolution, resolution, pipe.unet.dtype, device, g) # (1,4,64,64)
    # [HSTR] center-aware design
    if center:
        watermarked_latents_fft = fft(torch.zeros(shape, device=device)) # (1,4,64,64) complex64
        gt_patch_tmp = fft(gt_init[center_slice]).clone().detach() # (1,4,44,44) complex64
        center_len = gt_patch_tmp.shape[-1] // 2 # 22
        for radius in range(center_len-1, 0, -1): # [21,20,...,1]
            tmp_mask = torch.tensor(circle_mask(size=shape[-1], r=radius)) # (64,64)
            for j in range(watermarked_latents_fft.shape[1]): # GT : all channel Tree-Ring
                watermarked_latents_fft[:, j, tmp_mask] = gt_patch_tmp[0, j, center_len, center_len + radius].item() # Use (22,22+radius) element.
        if heter: # Gaussian noise key (Heterogenous watermark in RingID)
            watermarked_latents_fft[:, HETER_WATERMARK_CHANNEL, start:end, start:end] = gt_patch_tmp[:, HETER_WATERMARK_CHANNEL] # (1,1,44,44) complex64
    # [Original Tree-Ring]
    else:
        watermarked_latents_fft = fft(gt_init) # (1,4,64,64)
        # constant ring values chosen from a Gaussian distribution.
        gt_patch_tmp = watermarked_latents_fft.clone().detach()
        center_len = shape[-1] // 2 # 32
        for radius in range(center_len-1, 0, -1): # [31,30,...,1]
            tmp_mask = torch.tensor(circle_mask(size=shape[-1], r=radius))
            for j in range(watermarked_latents_fft.shape[1]): # GT : all channel Tree-Ring
                watermarked_latents_fft[:, j, tmp_mask] = gt_patch_tmp[0, j, center_len, center_len + radius].item() # Use (32,32+radius) element.
    # [Hermitian Symmetric Fourier] HSTR or TR
    if hs: 
        return enforce_hermitian_symmetry(watermarked_latents_fft)
    return watermarked_latents_fft # (1,4,64,64) complex64

# RI - ringid pattern
@torch.no_grad()
def make_Fourier_ringid_pattern(pipe, shape, key_value_combination,
        radius, radius_cutoff, ring_width=1, 
        ring_watermark_channel=RING_WATERMARK_CHANNEL, heter_watermark_channel=HETER_WATERMARK_CHANNEL, 
        heter_watermark_region_mask=None, w_seed=999999,
        hs=False):
    if ring_width != 1:
        raise NotImplementedError(f'Proposed watermark generation only implemented for ring width = 1.')
    if len(key_value_combination) != (RADIUS - RADIUS_CUTOFF):
        raise ValueError('Mismatch between #key values and #slots')
    if len(shape) != 4:
        raise ValueError(f'Invalid shape for initial latent: {shape}')
    device = pipe.device
    watermarked_latents_fft = fft(torch.zeros(shape, device=device))
    radius_list = [this_radius for this_radius in range(radius, radius_cutoff, -1)]
    # put ring
    for radius_index in range(len(radius_list)):
        this_r_out = radius_list[radius_index]
        this_r_in = this_r_out - ring_width
        mask = torch.tensor(ring_mask(size=shape[-1], r_out=this_r_out, r_in=this_r_in)).to(device).to(torch.float64)
        for batch_index in range(shape[0]):
            for channel_index in range(len(ring_watermark_channel)):
                watermarked_latents_fft[batch_index, ring_watermark_channel[channel_index]].real = \
                    (1 - mask) * watermarked_latents_fft[batch_index, ring_watermark_channel[channel_index]].real + mask * key_value_combination[radius_index][channel_index]
                watermarked_latents_fft[batch_index, ring_watermark_channel[channel_index]].imag = \
                    (1 - mask) * watermarked_latents_fft[batch_index, ring_watermark_channel[channel_index]].imag + mask * key_value_combination[radius_index][channel_index]
    # put noise or zeros
    if len(heter_watermark_channel) > 0:
        assert len(heter_watermark_channel) == len(heter_watermark_region_mask)
        heter_watermark_region_mask = heter_watermark_region_mask.to(torch.float64)
        w_type = 'noise'
        if w_type == 'noise':
            g = torch.Generator(device=device).manual_seed(w_seed)
            w_content = fft(torch.randn(shape, device=device, generator=g))  # [N, c, h, w]
            # w_content = fft(torch.randn(shape, device=device))  # [N, c, h, w]
        elif w_type == 'zeros':
            w_content = fft(torch.zeros(shape, device=device))  # [N, c, h, w]
        else:
            raise NotImplementedError
        for batch_index in range(shape[0]):
            for channel_id, channel_mask in zip(heter_watermark_channel, heter_watermark_region_mask):
                watermarked_latents_fft[batch_index, channel_id].real = \
                    (1 - channel_mask) * watermarked_latents_fft[batch_index, channel_id].real + channel_mask * w_content[batch_index][channel_id].real
                watermarked_latents_fft[batch_index, channel_id].imag = \
                    (1 - channel_mask) * watermarked_latents_fft[batch_index, channel_id].imag + channel_mask * w_content[batch_index][channel_id].imag
    return watermarked_latents_fft

# HSQR - hermitian symmetric QR pattern
class QRCodeGenerator:
    def __init__(self, box_size=2, border=1, qr_version=1):
        self.qr = qrcode.QRCode(version=qr_version, box_size=box_size, border=border,
            error_correction = qrcode.constants.ERROR_CORRECT_H)
    
    def make_qr_tensor(self, data, filename='qrcode.png', save_img=False):
        self.qr.add_data(data)
        self.qr.make(fit=True)
        img = self.qr.make_image(fill_color="black", back_color="white")
        if save_img:
            img.save(filename)
        self.clear()
        img_array = np.array(img)
        tensor = torch.from_numpy(img_array)
        return tensor.clone().detach() # boolean (h,w)
    
    def clear(self):
        self.qr.clear()

qr_generator = QRCodeGenerator(box_size=box_size, border=0, qr_version=qr_version)
@torch.no_grad()
def make_hsqr_pattern(idx: int):
    data = f"HSQR{idx % 10000}"
    qr_tensor = qr_generator.make_qr_tensor(data=data) # (42,42) boolean tensor
    qr_tensor = qr_tensor.repeat(len(HSQR_WATERMARK_CHANNEL), 1, 1) # (c_wm,42,42) boolean tensor
    return qr_tensor # (c_wm,42,42) boolean tensor

# ====================================================================================================
# [Inject Watermarks]
@torch.no_grad()
def inject_wm(inverted_latent, w_pattern, w_mask, cut_real=True, center=False, device="cuda"):
    assert len(w_pattern.shape) == 4
    assert len(w_mask.shape) == 4
    batch_size = inverted_latent.shape[0]
    w_mask = w_mask.repeat(batch_size, 1, 1, 1)

    inverted_latent = inverted_latent.to(device)
    w_pattern = w_pattern.to(device)
    w_mask = w_mask.to(device)
    # inject watermarks in fourier space
    # center 옵션에 따른 마스킹 처리
    if center:
        center_latent_fft = fft(inverted_latent[center_slice]) # (N,4,44,44) complex64
        # 워터마크 삽입
        temp_mask = w_mask[center_slice] # (N,4,44,44) boolean
        temp_pattern = w_pattern[center_slice] # (N,4,44,44) complex64
        center_latent_fft[temp_mask] = temp_pattern[temp_mask].clone() # (N,4,44,44) complex64
        # IFFT 및 원래 위치로 복원
        center_latent_ifft = ifft(center_latent_fft) # (N,4,44,44)
        center_latent_ifft = center_latent_ifft.real if cut_real or center_latent_ifft.imag.abs().max() < 1e-3 else center_latent_ifft
        # 원본 텐서에 복원
        inverted_latent = inverted_latent.clone()
        inverted_latent[center_slice] = center_latent_ifft
        inverted_latent_fft = None
    else:
        # 기존 로직 유지
        inverted_latent_fft = fft(inverted_latent) # complex64
        inverted_latent_fft[w_mask] = w_pattern[w_mask].clone()
        inverted_latent = ifft(inverted_latent_fft) # complex64
        inverted_latent = inverted_latent.real if cut_real or inverted_latent.imag.abs().max() < 1e-3 else inverted_latent
        # if cut_real: # enforcing to discard imaginary part regardless of its values.
        #     inverted_latent = inverted_latent.real # float32
        # else:
        #     if inverted_latent.imag.abs().max() < 1e-3: # discard numerical error in imaginary part.
        #         inverted_latent = inverted_latent.real # float32
        #     else:
        #         raise
    # hot fix to prevent out of bounds values. will "properly" fix this later
    inverted_latent[inverted_latent == float("Inf")] = 4
    inverted_latent[inverted_latent == float("-Inf")] = -4
    return inverted_latent, inverted_latent_fft # float32, complex64

@torch.no_grad()
def qr_abs(boolean_tensor, input_tensor, delta=0): # boolean → qr_abs tensor
    return torch.where(boolean_tensor, input_tensor.abs() + delta, -input_tensor.abs() - delta)

@torch.no_grad()
def inject_hsqr(inverted_latent, qr_tensor, center=False, device="cuda"): # (N,4,64,64) -> (N,4,64,64)
    assert len(qr_tensor.shape) == 4 # (N,c_wm,42,42)
    inverted_latent = inverted_latent.to(device)
    qr_tensor = qr_tensor.to(device)
    qr_pix_len = qr_tensor.shape[-1]    # 42
    qr_pix_half = (qr_pix_len + 1) // 2 # 21
    qr_left = qr_tensor[:, :, :, :qr_pix_half]    # (N,c_wm,42,21) boolean
    qr_right = qr_tensor[:, :, :, qr_pix_half:]   # (N,c_wm,42,21) boolean
    if center:
        # rfft
        center_latent_rfft = rfft(inverted_latent[center_slice]) # (N,4,44,44) -> # (N,4,44,23) complex64
        center_real_batch = center_latent_rfft.real # (N,4,44,23) f32
        center_imag_batch = center_latent_rfft.imag # (N,4,44,23) f32
        real_slice = (slice(None), HSQR_WATERMARK_CHANNEL, slice(1, 1+qr_pix_len), slice(1, 1+qr_pix_half))
        imag_slice = (slice(None), HSQR_WATERMARK_CHANNEL, slice(1, 1+qr_pix_len), slice(1, 1+qr_pix_half))
        #center=True  [:,[3], 1:43,1:22] (N,1,42,21)
        center_real_batch[real_slice] = qr_abs(qr_left, center_real_batch[real_slice], delta=delta) # (N,c_wm,42,21)
        center_imag_batch[imag_slice] = qr_abs(qr_right, center_imag_batch[imag_slice], delta=delta) # (N,c_wm,42,21)
        center_latent_ifft = irfft(torch.complex(center_real_batch, center_imag_batch)) # (N,4,44,44) f32
        inverted_latent = inverted_latent.clone()
        inverted_latent[center_slice] = center_latent_ifft
        return inverted_latent # (N,4,64,64)
    else:
        # Coordinates for HSQR injection
        center_row = inverted_latent.shape[-2] // 2 # 32
        row_start = center_row - qr_pix_half + (1 if qr_pix_len % 2 else 0) # if odd length QR, plus 1
        row_end = center_row + qr_pix_half
        col_end_left = 1 + qr_pix_half
        col_end_right = qr_pix_half if qr_pix_len % 2 else col_end_left # if odd length QR, shortend by 1 pix
        # rfft
        latent_rfft = rfft(inverted_latent) # (N,4,64,64) -> (N,4,64,33) complex64
        real_batch = latent_rfft.real # (N,4,64,33) f32
        imag_batch = latent_rfft.imag # (N,4,64,33) f32
        # Inject HSQR
        # [row_start 11 = 32-21 : row_end 53 = 32+21] / [col_start 1 : col_end_left 22 = 1+21 = col_end_right 22]
        real_slice = (slice(None), HSQR_WATERMARK_CHANNEL, slice(row_start, row_end), slice(1, col_end_left))
        imag_slice = (slice(None), HSQR_WATERMARK_CHANNEL, slice(row_start, row_end), slice(1, col_end_right))
        #center=False [:,[3],11:53,1:22] (N,1,42,21)
        real_batch[real_slice] = qr_abs(qr_left, real_batch[real_slice], delta=delta) # (N,c_wm,42,21)
        imag_batch[imag_slice] = qr_abs(qr_right, imag_batch[imag_slice], delta=delta) # (N,c_wm,42,21)
        return irfft(torch.complex(real_batch, imag_batch)) # (N,4,64,64)

# ====================================================================================================

# ====================================================================================================
# [image attacks]
class RandomCropWithOriginalPosition:
    def __init__(self, crop_size, original_size, fill=0):
        self.crop_size = crop_size
        self.original_size = original_size
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        top = random.randint(0, h - self.crop_size)
        left = random.randint(0, w - self.crop_size)
        cropped_img = img.crop((left, top, left + self.crop_size, top + self.crop_size))
        padded_img = Image.new(img.mode, (self.original_size, self.original_size), self.fill)
        padded_img.paste(cropped_img, (left, top))
        return padded_img

vaeb = bmshj2018_hyperprior(quality=3, pretrained=True).to("cuda").eval()
vaec = cheng2020_anchor(quality=3, pretrained=True).to("cuda").eval()
@torch.no_grad()
def image_distortion(img1, img2, seed, 
                     brightness_factor = None, 
                     contrast_factor = None, 
                     jpeg_ratio = None, 
                     gaussian_blur_r = None, 
                     gaussian_std = None, 
                     bm3d_sigma = None,
                     vaeb_quality = None,
                     vaec_quality = None,
                     center_crop_area_ratio = None,
                     random_crop_area_ratio = None,
                     ):
    if brightness_factor is not None:
        if img1 is not None:
            img1 = tforms.ColorJitter(brightness=brightness_factor)(img1)
        img2 = tforms.ColorJitter(brightness=brightness_factor)(img2)
    if contrast_factor is not None:
        if img1 is not None:
            img1 = ImageEnhance.Contrast(img1).enhance(contrast_factor)
        img2 = ImageEnhance.Contrast(img2).enhance(contrast_factor)
    if jpeg_ratio is not None:
        if img1 is not None:
            buf = io.BytesIO()
            img1.save(buf, format='JPEG', quality=jpeg_ratio)
            img1 = Image.open(buf)
        buf2 = io.BytesIO()
        img2.save(buf2, format='JPEG', quality=jpeg_ratio)
        img2 = Image.open(buf2)
    if gaussian_blur_r is not None:
        if img1 is not None:
            img1 = Image.fromarray(cv2.GaussianBlur(np.array(img1), (gaussian_blur_r, gaussian_blur_r), 1))
        img2 = Image.fromarray(cv2.GaussianBlur(np.array(img2), (gaussian_blur_r, gaussian_blur_r), 1))
    if gaussian_std is not None:
        img_shape = np.array(img1).shape
        g_noise = np.random.normal(0, gaussian_std, img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        if img1 is not None:
            img1 = Image.fromarray(np.clip(np.array(img1) + g_noise, 0, 255))
        img2 = Image.fromarray(np.clip(np.array(img2) + g_noise, 0, 255))
    if bm3d_sigma is not None:
        if img1 is not None:
            img1 = Image.fromarray((np.clip(bm3d_rgb(np.array(img1) / 255, bm3d_sigma), 0, 1) * 255).astype(np.uint8))
        img2 = Image.fromarray((np.clip(bm3d_rgb(np.array(img2) / 255, bm3d_sigma), 0, 1) * 255).astype(np.uint8))
    if vaeb_quality is not None:
        assert vaeb_quality == 3, "Only quality 3 is supported for VAE-B"
        img_transforms = tforms.Compose([tforms.Resize((512,512)), tforms.ToTensor()])
        if img1 is not None:
            enc1 = vaeb.compress(img_transforms(img1).unsqueeze(0).to("cuda"))
            dec1 = vaeb.decompress(enc1['strings'], enc1['shape'])
            img1 = tforms.ToPILImage()(dec1['x_hat'].squeeze())
        enc2 = vaeb.compress(img_transforms(img2).unsqueeze(0).to("cuda"))
        dec2 = vaeb.decompress(enc2['strings'], enc2['shape'])
        img2 = tforms.ToPILImage()(dec2['x_hat'].squeeze())
    if vaec_quality is not None:
        assert vaec_quality == 3, "Only quality 3 is supported for VAE-C"
        img_transforms = tforms.Compose([tforms.Resize((512,512)), tforms.ToTensor()])
        if img1 is not None:
            enc1 = vaec.compress(img_transforms(img1).unsqueeze(0).to("cuda"))
            dec1 = vaec.decompress(enc1['strings'], enc1['shape'])
            img1 = tforms.ToPILImage()(dec1['x_hat'].squeeze())
        enc2 = vaec.compress(img_transforms(img2).unsqueeze(0).to("cuda"))
        dec2 = vaec.decompress(enc2['strings'], enc2['shape'])
        img2 = tforms.ToPILImage()(dec2['x_hat'].squeeze())
    if center_crop_area_ratio is not None:
        crop_len = int(512 * (center_crop_area_ratio ** 0.5))
        padding_left = (512 - crop_len) // 2
        padding_right = (512 - crop_len) - padding_left
        center_crop_transforms = tforms.Compose([
            tforms.CenterCrop(size=(crop_len, crop_len)),
            tforms.Pad(padding=(padding_left, padding_left, padding_right, padding_right), fill=0),])
        if img1 is not None:
            img1 = center_crop_transforms(img1)
        img2 = center_crop_transforms(img2)
    if random_crop_area_ratio is not None:
        crop_len = int(512 * (random_crop_area_ratio ** 0.5))
        random_crop_transforms = RandomCropWithOriginalPosition(crop_len, 512, fill=0)
        if img1 is not None:
            img1 = random_crop_transforms(img1)
        img2 = random_crop_transforms(img2)
    return [img1, img2]

# ====================================================================================================
# [Detecting Process] pillow_image → torch_image → encoded_latent → ddim_inversion
def transform_img(image, resolution=512):
    tform = tforms.Compose([tforms.Resize((resolution,resolution)), tforms.ToTensor()])
    image = tform(image)
    return 2.0 * image - 1.0

@torch.no_grad()
def pil2latent(pipe, image_pil):
    images = [image_pil] if isinstance(image_pil, Image.Image) else image_pil
    image_tensor = torch.stack([transform_img(image) for image in images]).to(pipe.unet.dtype).to(pipe.device)
    image_latent = pipe.vae.encode(image_tensor).latent_dist.mode() * pipe.vae.config.scaling_factor #latent_scaling_factor
    return image_latent

@torch.no_grad()
def ddim_invert(pipe, image_pil, invert_prompt="", invert_guidance=0):
    invert_prompt_list = [""] * len(image_pil) if isinstance(image_pil, list) and invert_prompt == "" else invert_prompt
    # change to inversion scheduler
    curr_scheduler = pipe.scheduler
    pipe.scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    # ddim inversion
    image_latent = pil2latent(pipe, image_pil)
    inverted_latent = pipe(prompt=invert_prompt_list, latents=image_latent, guidance_scale=invert_guidance, num_inference_steps=50, output_type="latent").images
    # revert to original scheduler
    pipe.scheduler = curr_scheduler
    return inverted_latent # (N,4,64,64)

# [Detection] Get distance metric
@torch.no_grad()
def get_distance(tensor1, tensor2, mask, channel=RINGID_WATERMARK_CHANNEL, 
                p=1, mode='complex', channel_min=False, center=False):
    if tensor1.shape != tensor2.shape:
        raise ValueError(f'Shape mismatch during eval: {tensor1.shape} vs {tensor2.shape}')
    if mode not in ['complex', 'real', 'imag']:
        raise NotImplementedError(f'Eval mode not implemented: {mode}')
    
    def calc_diff(t1, t2, m):
        if mode == 'complex':
            diff = torch.abs(t1 - t2)
        elif mode == 'real':
            diff = torch.abs(t1.real - t2.real)
        else:  # 'imag'
            diff = torch.abs(t1.imag - t2.imag)
        return diff if m is None else diff[m]
    
    if center:
        temp_tensor1 = tensor1[center_slice].clone() # 1,4,64,64 -> 1,4,44,44
        temp_tensor2 = tensor2[center_slice].clone() # 1,4,44,44
        temp_mask = mask[None, ...][center_slice][0].clone() # (C_R+C_H,64,64) -> (C_R+C_H,44,44)
        if not channel_min: # C_H=0. Only non-hetero watermarked channels C_R.
            diff = calc_diff(temp_tensor1[0][channel], temp_tensor2[0][channel], temp_mask) # (C_R,44,44) masked.
            return torch.norm(diff, p=p).item() / torch.sum(temp_mask) if p != 1 else torch.mean(diff).item()    
        else:
            assert p == 1
            diff = calc_diff(temp_tensor1[0][channel], temp_tensor2[0][channel], None)  # (C_R+C_H,44,44) unmasked.
            l1_list = [torch.mean(diff[i][temp_mask[i]]).item() for i in range(len(channel))]
            if channel == RINGID_WATERMARK_CHANNEL: # [0,3]
                return min(l1_list)
            else:
                raise NotImplementedError
    else:
        if not channel_min:
            diff = calc_diff(tensor1[0][channel], tensor2[0][channel], mask) # (C_R,64,64) masked.
            return torch.norm(diff, p=p).item() / torch.sum(mask) if p != 1 else torch.mean(diff).item()
        else:
            diff = calc_diff(tensor1[0][channel], tensor2[0][channel], None)  # (C_R+C_H,64,64) unmasked.
            l1_list = [torch.mean(diff[i][mask[i]]).item() for i in range(len(channel))]
            
            if len(RING_WATERMARK_CHANNEL) > 1 and len(HETER_WATERMARK_CHANNEL) > 0:
                ring_indices = [i for i, c in enumerate(RINGID_WATERMARK_CHANNEL) if c in RING_WATERMARK_CHANNEL]
                heter_indices = [i for i, c in enumerate(RINGID_WATERMARK_CHANNEL) if c in HETER_WATERMARK_CHANNEL]
                ring_mean = sum(l1_list[i] * torch.sum(mask[i]).item() for i in ring_indices) / sum(torch.sum(mask[i]).item() for i in ring_indices)
                return min(ring_mean, min(l1_list[i] for i in heter_indices))
            elif len(RING_WATERMARK_CHANNEL) == 1 and len(HETER_WATERMARK_CHANNEL) > 0:
                return min(l1_list)
            else:
                raise NotImplementedError

@torch.no_grad()
def get_distance_hsqr(qr_gt_bool, target_fft, channel=HSQR_WATERMARK_CHANNEL,
                    p=1, center=False):
    """
    qr_gt_bool : (c_wm,42,42) boolean
    target_fft : (1,4,64,64) complex64
    """
    center_row = target_fft.shape[-2] // 2 # 32
    qr_pix_len = qr_gt_bool.shape[-1]    # 42
    qr_pix_half = (qr_pix_len + 1) // 2 # 21
    qr_gt_f32 = torch.where(qr_gt_bool, torch.tensor(45.0), torch.tensor(-45.0)).to(torch.float32) # (c_wm,42,42) boolean -> float32
    qr_left = qr_gt_f32[0, :, :qr_pix_half]   # (42,21) float32
    qr_right = qr_gt_f32[0, :, qr_pix_half:]  # (42,21) float32
    qr_complex = torch.complex(qr_left, qr_right).to(target_fft.device) # (42,21) complex64
    if center:
        row_start = 10 + 1 # 11
        row_end = row_start + qr_pix_len # 53 = 11+42
        col_start = center_row + 1 # 33 = 32+1
        col_end = col_start + qr_pix_half # 54 = 33+21
    else:
        row_start = center_row - qr_pix_half + (1 if qr_pix_len % 2 else 0) # if odd length QR, plus 1
        row_end = center_row + qr_pix_half
        col_start = center_row + 1 # 33
        col_end = col_start + qr_pix_half # 33+21
        # [TBD] the odd case will be updated
    qr_slice = (0, channel, slice(row_start, row_end), slice(col_start, col_end)) # (42,21)
    diff = torch.abs(qr_complex - target_fft[qr_slice]) # (42,21)
    return torch.norm(diff, p=p).item() / diff.numel() if p != 1 else torch.mean(diff).item()

# ====================================================================================================
# [Metrics] Generation quality : CLIP score, FID
@torch.no_grad() 
def get_clip_score(image_batch, prompt_batch, model, clip_preprocess, tokenizer, device="cuda"):
    image_batch = [image_batch] if isinstance(image_batch, Image.Image) else image_batch    # N_i : image num_batch
    prompt_batch = [prompt_batch] if isinstance(prompt_batch, str) else prompt_batch        # N_p : prompt num_batch
    assert len(image_batch) == len(prompt_batch)
    # image features
    img_batch = [clip_preprocess(image).unsqueeze(0) for image in image_batch]
    img_batch = torch.concatenate(img_batch).to(device) # (N_i,3,224,224)
    image_features = model.encode_image(img_batch) # (N_i,1024)
    # text features
    text = tokenizer(prompt_batch).to(device) # (N_p,77)
    text_features = model.encode_text(text) # (N_p,1024)
    # normalize
    image_features /= image_features.norm(dim=-1, keepdim=True) # (N_i,1024)
    text_features /= text_features.norm(dim=-1, keepdim=True) # (N_p,1024)
    # return (image_features @ text_features.T).mean(-1) # (N_i,1024)@(1024,N_p)=(N_i,N_p) -> mean(-1) -> (N_i,)
    return (image_features * text_features).sum(-1) # (N_i,)=(N_p,)

# FID : Folder-based measurement (total image distribution)
def get_FID(gt_folder, target_folder, device="cuda"):
    return calculate_fid_given_paths([gt_folder, target_folder], batch_size=64, device=device, dims=2048, num_workers=16)

# [Metrics] Reference-based image quality : PSNR, SSIM, LPIPS
def path_to_pil(img_path):
    if isinstance(img_path, str) and os.path.isfile(img_path):
        return Image.open(img_path)
    elif isinstance(img_path, Image.Image):   
        return img_path
    else:
        raise ValueError("유효한 파일 경로나 Pillow 이미지 객체를 입력하세요.")

def get_psnr(img1, img2, eps=1e-10):
    # caluclate psnr
    img1 = np.array(path_to_pil(img1).convert('RGB'))
    img2 = np.array(path_to_pil(img2).convert('RGB'))
    mse = np.mean((img1 - img2) ** 2)
    mse = max(mse, eps)
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def get_ssim(img1, img2):
    # caluclate ssim
    img1 = np.array(path_to_pil(img1).convert('L')) # 흑백으로 변환
    img2 = np.array(path_to_pil(img2).convert('L')) # 흑백으로 변환
    ssim_value, _ = ssim(img1, img2, full=True)
    return ssim_value

loss_fn = lpips.LPIPS(net="vgg").to(device)
@torch.no_grad()
def get_lpips(img1, img2, device="cuda"):
    # caluclate LPIPS(VGG): image should be RGB, normalized to [-1,1]
    img1 = transform_img(path_to_pil(img1).convert('RGB'), resolution=224).unsqueeze(0).to(torch.float32).to(device)
    img2 = transform_img(path_to_pil(img2).convert('RGB'), resolution=224).unsqueeze(0).to(torch.float32).to(device)
    lpips_value = loss_fn(img1, img2)
    return lpips_value.item()
