import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import random

class RadialBasisFilter(nn.Module):
    def __init__(self, n_coeff, lamda):
        super().__init__()
        self.n_coeff = n_coeff
        self.n_ang_freq = 1
        # learnable coeffs (real-valued)
        self.coeff_mag   = nn.Parameter(torch.zeros(n_coeff))
        self.coeff_phase = nn.Parameter(torch.zeros(n_coeff))
        self.lamda = lamda

        self.raw_gate_mag = nn.Parameter(torch.ones(n_coeff))    
        self.raw_gate_phase = nn.Parameter(torch.ones(n_coeff))  

        mu = torch.linspace(0.0, 1.0, steps=n_coeff)
        self.register_buffer('mu', mu)
        self.log_bwh = nn.Parameter(torch.tensor(0.0))

    def forward(self, H: int, W: int, device, dtype):
        fy = torch.fft.fftfreq(H, dtype=dtype, device=device)[:, None]
        fx = torch.fft.rfftfreq(W, dtype=dtype, device=device)[None, :]
        r_hat = torch.sqrt(fx ** 2 + fy ** 2)
        r_hat = r_hat / r_hat.max()  # normalize to 0..1

        bwh = torch.exp(self.log_bwh) + 1e-6
        basis = torch.exp(-((r_hat.unsqueeze(0) - self.mu[:, None, None]) ** 2) / (2 * bwh ** 2))

        gate_mag = torch.sigmoid(self.raw_gate_mag)[:, None, None]
        gate_phase = torch.sigmoid(self.raw_gate_phase)[:, None, None]

        angular_mod = 0
        theta = torch.atan2(fy, fx + 1e-8)
        for n in range(1, self.n_ang_freq + 1):
            angular_mod += torch.cos(n * theta) + torch.sin(n * theta)
        angular_mod = angular_mod / (2 * self.n_ang_freq)
        angular_mod = 1 + 0.1 * angular_mod

        basis = basis * angular_mod.unsqueeze(0)

        diff_mag = (gate_mag * self.coeff_mag[:, None, None] * basis).sum(0, keepdim=True)
        diff_phase = (gate_phase * self.coeff_phase[:, None, None] * basis).sum(0, keepdim=True)
        return diff_mag, diff_phase

class LearnableFreFilter(nn.Module):
    def __init__(self, number_K = 10, lamda = 0.1):
        super().__init__()
        def generate_random_number():
            number = round(random.uniform(0.95, 1.05), 2)
            return number
        
        self.init_sigma_ratio = 0.2
        # self.alpha_rg  = nn.Parameter(torch.tensor(generate_random_number()))
        # self.alpha_gb  = nn.Parameter(torch.tensor(generate_random_number()))
        # self.alpha_rb  = nn.Parameter(torch.tensor(generate_random_number()))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))

        self.rad_filter  = RadialBasisFilter(number_K, lamda)

        self._sigma_init = False
    # ------------------------------------------------------------------
    def forward(self, img):
        B, C, H, W = img.shape
        dtype, device = img.dtype, img.device
        assert C == 3

        if not self._sigma_init:
            sigma_px = self.init_sigma_ratio * min(H, W)
            with torch.no_grad():
                self.log_sigma.copy_(torch.tensor(np.log(sigma_px), dtype=dtype, device=device))
            self._sigma_init = True

        diff_mag, diff_phase = self.rad_filter(H, W, device, dtype)  # (1,H,W/2+1)
        D = diff_mag.to(dtype) * torch.exp(1j * diff_phase.to(dtype))

        x = torch.log(img.clamp(min=1e-6))
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        fft_r, fft_g, fft_b = (torch.fft.rfft2(ch, norm='ortho') for ch in (r, g, b))
        diff_rg = fft_r - fft_g
        diff_gb = fft_g - fft_b
        diff_rb = fft_r - fft_b

        fy = torch.fft.fftfreq(H, dtype=dtype, device=device)[:, None]
        fx = torch.fft.rfftfreq(W, dtype=dtype, device=device)[None, :]
        r_grid = torch.sqrt(fx ** 2 + fy ** 2)
        sigma = torch.exp(self.log_sigma)
        Wg = torch.exp(- (r_grid / sigma) ** 2)  
        Wg = Wg.clone(); Wg[0, 0] = 0.0          


        def _filt(diff_fft):
            diff_fft = diff_fft.squeeze(1)
            out = torch.fft.irfft2((D * Wg) * diff_fft, s=(H, W), norm='ortho')
            return out.unsqueeze(1)

        fccr_rg = _filt(diff_rg)
        fccr_gb = _filt(diff_gb)
        fccr_rb = _filt(diff_rb)
        fccr_feat = torch.cat([fccr_rg, fccr_gb, fccr_rb], dim=1)  # (B,3,H,W)

        return fccr_feat

class FIINet(nn.Module):
    def __init__(self, number_K, lamda): 
        super(FIINet, self).__init__()

        self.spatial_net = nn.Sequential(*[nn.Conv2d(3, 24, 3, 1, 1, groups=1), 
                                        nn.BatchNorm2d(24),
                                        nn.LeakyReLU(),
                                        ])
        self.spectral_net = nn.Sequential(*[nn.Conv2d(3, 24, 3, 1, 1, groups=1), 
                                        nn.BatchNorm2d(24),
                                        nn.LeakyReLU(),
                                        ])
        self.fuse_net = nn.Sequential(*[nn.Conv2d(48, 32, 3, 1, 1, groups=2),
                                    nn.BatchNorm2d(32),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(32, 3, 3, 1, 1, groups=1)
                                    ])
        self.fim = LearnableFreFilter(number_K, lamda)

    def forward(self, x):
        feat_f = self.fim(x)
        feat_spatial = self.spatial_net(x)
        feat_spectral = self.spectral_net(feat_f)
        feat_agg = torch.concat((feat_spatial, feat_spectral), dim=1)
        x_out = self.fuse_net(feat_agg)
        return x_out

