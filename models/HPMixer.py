import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from models.Rev_in import RevIN
import pywt

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted

class LearnableSWT(nn.Module):

    def __init__(self, in_channels, wavelet_name='db2', level=2, trainable=True):
        super().__init__()
        self.in_channels = in_channels
        self.level = level

        wavelet = pywt.Wavelet(wavelet_name)
        h0 = torch.tensor(wavelet.dec_lo[::-1], dtype=torch.float32)
        h1 = torch.tensor(wavelet.dec_hi[::-1], dtype=torch.float32) 
        self.kernel_size = h0.shape[-1]

        h0 = h0.view(1, 1, -1).repeat(self.in_channels, 1, 1)
        h1 = h1.view(1, 1, -1).repeat(self.in_channels, 1, 1)
        self.h0 = nn.Parameter(h0, requires_grad=trainable)
        self.h1 = nn.Parameter(h1, requires_grad=trainable)

    def forward(self, x):

        coeffs = []
        approx_coeffs = x
        dilation = 1

        for _ in range(self.level):
            padding = dilation * (self.kernel_size - 1)
            padding_right = padding // 2
            padding_left = padding - padding_right
            approx_coeffs_pad = F.pad(approx_coeffs, (padding_left, padding_right), "circular")

            detail_coeff = F.conv1d(approx_coeffs_pad, self.h1, dilation=dilation, groups=self.in_channels)
            approx_coeffs = F.conv1d(approx_coeffs_pad, self.h0, dilation=dilation, groups=self.in_channels)
            
            coeffs.append(detail_coeff)
            dilation *= 2
            
        coeffs.append(approx_coeffs)
        return torch.stack(list(reversed(coeffs)), dim=2)

class LearnableISWT(nn.Module):

    def __init__(self, in_channels, wavelet_name='db2', level=2, trainable=True):
        super().__init__()
        self.in_channels = in_channels
        self.level = level

        wavelet = pywt.Wavelet(wavelet_name)
        g0 = torch.tensor(wavelet.rec_lo[::-1], dtype=torch.float32) 
        g1 = torch.tensor(wavelet.rec_hi[::-1], dtype=torch.float32)
        self.kernel_size = g0.shape[-1]

        g0 = g0.view(1, 1, -1).repeat(self.in_channels, 1, 1)
        g1 = g1.view(1, 1, -1).repeat(self.in_channels, 1, 1)
        self.g0 = nn.Parameter(g0, requires_grad=trainable)
        self.g1 = nn.Parameter(g1, requires_grad=trainable)

    def forward(self, coeffs):
        approx_coeff = coeffs[:,:,0,:]
        detail_coeffs = coeffs[:,:,1:,:]
        
        dilation = 2 ** (self.level - 1)

        for i in range(self.level):
            detail_coeff = detail_coeffs[:,:,self.level-1-i,:]

            padding = dilation * (self.kernel_size - 1)
            padding_left = (dilation * self.kernel_size) // 2
            pad = (padding_left, padding - padding_left)
            
            approx_coeff_pad = F.pad(approx_coeff, pad, "circular")
            detail_coeff_pad = F.pad(detail_coeff, pad, "circular")

            y_approx = F.conv1d(approx_coeff_pad, self.g0, groups=self.in_channels, dilation=dilation)
            y_detail = F.conv1d(detail_coeff_pad, self.g1, groups=self.in_channels, dilation=dilation)
            approx_coeff = (y_approx + y_detail) / 2
            
            dilation //= 2
            
        return approx_coeff
    
class Patching(nn.Module):
    def __init__(self, patch_len):
        super().__init__()
        self.patch_len = patch_len

    def forward(self, x):
        seq_len = x.shape[-1]
        pad_len = (self.patch_len - (seq_len % self.patch_len)) % self.patch_len
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))
        num_patches = x.shape[-1] // self.patch_len
        x_patched = x.view(*x.shape[:-1], num_patches, self.patch_len)
        return x_patched, pad_len


class UnPatching(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_patched, pad_len):
        x = x_patched.flatten(start_dim=-2)
        if pad_len > 0:
            x = x[..., :-pad_len]
        return x


class MixingCycle(nn.Module):
    def __init__(self, cycle_len, channel_size, d_model=512, dropout=0.1):
        super().__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.d_model = d_model
        
        self.data = nn.Parameter(torch.zeros(cycle_len, channel_size))
        nn.init.xavier_uniform_(self.data)

        self.mixer = nn.Sequential(
            nn.Linear(self.channel_size, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.channel_size)
        )
        
        self.norm_final = nn.LayerNorm([channel_size, cycle_len])

    def forward(self, start_index, length):
        B = start_index.shape[0]
        device = start_index.device

        gather_index = (start_index.view(-1, 1) + torch.arange(self.cycle_len, device=device).view(1, -1)) % self.cycle_len
        
        base_cycle_for_temporal = self.data[gather_index]
        base_shifted_cycle = base_cycle_for_temporal.permute(0, 2, 1)

        x_for_mlp = base_shifted_cycle.permute(0, 2, 1)
        
        mlp_out = self.mixer(x_for_mlp)
        
        refined_channel = mlp_out.permute(0, 2, 1)
        
        refined_cycle = refined_channel
        
        refined_cycle = self.norm_final(refined_cycle + base_shifted_cycle)

        if length > self.cycle_len:
            num_repeats = (length + self.cycle_len - 1) // self.cycle_len
            repeated_cycle = refined_cycle.repeat(1, 1, num_repeats)
            final_window = repeated_cycle[..., :length]
        else:
            final_window = refined_cycle[..., :length]

        return final_window
    
    
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.use_revin = configs.revin
        self.dropout_val = configs.dropout
        self.patch_len = configs.patch_size
        self.fine_patch_len = configs.fine_patch_size
        self.num_levels = configs.wavelet_j
        self.num_scales = self.num_levels + 1

        self.rev_norm = RevIN(self.enc_in, affine=configs.affine)
  
        self.mlp_cycle_module = MixingCycle(
            cycle_len=configs.cycle, 
            channel_size=self.enc_in,
            d_model=self.d_model 
        )

        self.swt_decomp = LearnableSWT(
            in_channels=self.enc_in,
            wavelet_name=configs.wavelet,
            level=self.num_levels,
            trainable=True
        )
        self.swt_recon = LearnableISWT(
            in_channels=self.enc_in,
            wavelet_name=configs.wavelet,
            level=self.num_levels,
            trainable=True
        )

        if self.patch_len >= self.seq_len or self.seq_len % self.patch_len != 0:
            self.patch_len = self.seq_len // 2
        self.patching_coarse = Patching(self.patch_len)
        self.patching_sub = Patching(self.fine_patch_len)
        self.unpatching_sub = UnPatching()
        self.unpatching_coarse = UnPatching()

        self.enc_embedding = DataEmbedding_inverted(
            self.patch_len,  
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.fc_dropout
        )
        
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.encoder_out_proj = nn.Linear(configs.d_model, self.patch_len)

        num_fine_patches = (self.patch_len + self.fine_patch_len - 1) // self.fine_patch_len
        num_coarse_patches = (self.seq_len + self.patch_len - 1) // self.patch_len
        coarse_patch_mlp_dim = num_coarse_patches * self.patch_len
        fine_patch_mlp_dim = num_fine_patches * self.fine_patch_len

        self.inter_fine_patch_mlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.inter_coarse_patch_mlps = nn.ModuleList()
        self.fine_res_mlps = nn.ModuleList()
        self.coarse_res_mlps = nn.ModuleList()
        
        self.coarse_patch_len_mlps = nn.ModuleList()
        self.fine_patch_len_mlps = nn.ModuleList()

        for _ in range(self.num_scales):
            self.coarse_patch_len_mlps.append(nn.Sequential(nn.Linear(self.patch_len, self.d_model), nn.GELU(),
                                                         nn.Dropout(p=self.dropout_val),
                                                         nn.Linear(self.d_model, self.patch_len)))
            
            self.fine_patch_len_mlps.append(nn.Sequential(nn.Linear(self.fine_patch_len, self.d_model), nn.GELU(),
                                                         nn.Dropout(p=self.dropout_val),
                                                         nn.Linear(self.d_model, self.fine_patch_len)))
            
            
            self.inter_fine_patch_mlps.append(nn.Sequential(
                nn.Linear(fine_patch_mlp_dim, self.d_model), nn.GELU(),
                nn.Dropout(p=self.dropout_val),
                nn.Linear(self.d_model, fine_patch_mlp_dim)))
            self.batch_norms.append(nn.BatchNorm1d(self.enc_in))
            self.inter_coarse_patch_mlps.append(nn.Sequential(
                nn.Linear(coarse_patch_mlp_dim, self.d_model), nn.GELU(),
                nn.Dropout(p=self.dropout_val), nn.Linear(self.d_model, coarse_patch_mlp_dim)))
  
        self.mlp_residual = nn.Sequential(
            nn.Linear(self.seq_len, self.d_model), nn.GELU(),
            nn.Dropout(p=self.dropout_val), nn.Linear(self.d_model, self.seq_len))
        
        self.pred_layer_residual = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x, cycle_index, x_mark_enc=None):
        B, L, C = x.shape
        x0 = self.rev_norm(x, 'norm') if self.use_revin else x
        x0_permuted = x0.permute(0, 2, 1)

        periodicity_in_sample = self.mlp_cycle_module(
            start_index=cycle_index, 
            length=self.seq_len
        )
        
        residual = x0_permuted - periodicity_in_sample
        coeffs = self.swt_decomp(residual)
        processed_coeffs_list = []

        for m in range(self.num_scales):
            scale_coeffs = coeffs[:, :, m, :]
            
            coarse_patches, pad_len_coarse = self.patching_coarse(scale_coeffs)
            
            residual_for_encoder = coarse_patches
            
            B_enc, C_enc, N_coarse, P_coarse = coarse_patches.shape
            encoder_input = coarse_patches.permute(0, 2, 1, 3).reshape(B_enc * N_coarse, C_enc, P_coarse).permute(0,2,1)
            enc_embedded_input = self.enc_embedding(encoder_input, None)
            enc_output, _ = self.encoder(enc_embedded_input, attn_mask=None)

            projected_output = self.encoder_out_proj(enc_output)

            processed_patches = projected_output.view(B_enc, N_coarse, C_enc, P_coarse).permute(0, 2, 1, 3)

            coarse_patches = residual_for_encoder + processed_patches

            N_coarse, P_coarse = coarse_patches.shape[-2:]
            coarse_patches_flat = coarse_patches.reshape(-1, P_coarse)
            fine_patches, pad_len_sub = self.patching_sub(coarse_patches_flat)
            
            fine_patches_out = self.fine_patch_len_mlps[m](fine_patches)
            fine_patche_out = fine_patches_out + fine_patches
            
            coarse_patches_processed_flat = self.unpatching_sub(fine_patche_out, pad_len_sub)
            coarse_patches_processed_flat = self.batch_norms[m](coarse_patches_processed_flat.reshape(B,C,-1))
            coarse_patches_processed = coarse_patches_processed_flat.view(B, C, N_coarse, P_coarse)

            res_identity_coarse = coarse_patches_processed
            blended_res_coarse = res_identity_coarse

            coarse_patches_flat = coarse_patches_processed.flatten(start_dim=2)
            mlp_output_flat = self.inter_coarse_patch_mlps[m](coarse_patches_flat)
            mlp_output_reshaped = mlp_output_flat.view(B, C, N_coarse, P_coarse)
            coarse_patches_processed = mlp_output_reshaped + blended_res_coarse
            coarse_patches_processed_out = self.coarse_patch_len_mlps[m](coarse_patches_processed)
            coarse_patches_processed_out = coarse_patches_processed_out + coarse_patches_processed

            processed_scale = self.unpatching_coarse(coarse_patches_processed, pad_len_coarse)
            processed_coeffs_list.append(processed_scale)
        
        processed_coeffs = torch.stack(processed_coeffs_list, dim=2)
        reconstructed_residual = self.swt_recon(processed_coeffs)
        
        mlp_output = self.mlp_residual(reconstructed_residual)
        out_residual = mlp_output + reconstructed_residual
        residual_forecast = self.pred_layer_residual(out_residual)
        
        periodicity_forecast = self.mlp_cycle_module(
            start_index=cycle_index + self.seq_len, 
            length=self.pred_len
        )

        final_forecast = periodicity_forecast + residual_forecast
        out = final_forecast.permute(0, 2, 1)
        out = self.rev_norm(out, 'denorm') if self.use_revin else out
        return out