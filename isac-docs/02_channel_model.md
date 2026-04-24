# 02 — Sub-THz Channel Model

> **Paper sections:** Sec. 2.3 (Sub-THz path loss), Sec. 2.4 (ISAC radar), Sec. 2.5 (CU activity), Sec. 4.1 (Received signal), Sec. 12.2 (MIMO channel)  
> **Code:** `src/channel.py`

---

## 2.1 Sub-THz Path Loss (Sec. 2.3)

The total path loss at 142 GHz combines three components:

$$\text{PL}(d, f) = \underbrace{20\log_{10}\!\left(\frac{4\pi d f}{c}\right)}_{\text{Free-space (FSPL)}} + \underbrace{\alpha_{\text{abs}}(f) \cdot d}_{\text{Molecular absorption}} + \underbrace{\chi_\sigma}_{\text{Shadowing}}$$

| Term | Formula | Parameter |
|------|---------|-----------|
| FSPL | $20\log_{10}(4\pi d f/c)$ | $f = 142\,\text{GHz}$, $c = 3\times10^8\,\text{m/s}$ |
| Molecular absorption | $\alpha_{\text{abs}} \cdot d$ | $\alpha_{\text{abs}} = 2.1\,\text{dB/km}$ (H₂O at 142 GHz) |
| Log-normal shadowing | $\chi_\sigma \sim \mathcal{N}(0, \sigma_\chi^2)$ | $\sigma_\chi = 3\,\text{dB}$ |

**Linear path-loss attenuation factor:**

$$L(d,f) = 10^{\text{PL}(d,f)/10}$$

> **Implementation:** `channel.py :: free_space_path_loss_db()`, `molecular_absorption_loss_db()`, `total_path_loss_db()`, `path_loss_linear()`

---

## 2.2 Nakagami-m Small-Scale Fading (Sec. 2.3)

Sub-THz channels at short range exhibit mild fading. We model the small-scale amplitude using the **Nakagami-m distribution**:

$$|h| \sim \text{Nakagami}(m, \Omega)$$

The squared amplitude (power) follows a Gamma distribution:

$$|h|^2 \sim \text{Gamma}\!\left(m,\, \frac{\Omega}{m}\right)$$

A complex channel coefficient with uniformly distributed phase:

$$h = |h| \cdot e^{j\phi}, \quad \phi \sim \mathcal{U}[0, 2\pi)$$

**Parameters:** $m = 2$ (mild fading, sub-THz short-range), $\Omega = 1/L(d,f)$ (mean power set by path loss).

> **Implementation:** `channel.py :: nakagami_amplitude()`, `complex_nakagami_channel()`  
> **Parameter:** `cfg.NAKAGAMI_M = 2.0`

---

## 2.3 Received Signal Model (Sec. 4.1)

For NCU $k$ at time slot $t$, the received complex baseband signal is:

$$y_k(i) = \begin{cases} \sqrt{P_c |h_{ck}|^2}\, s_c(i) + w_k(i) & H_t = 1\text{ (CU active)} \\ w_k(i) & H_t = 0\text{ (CU idle)} \end{cases}$$

where:
- $h_{ck} \in \mathbb{C}$ — CU-to-NCU fading coefficient (Nakagami-m)
- $P_c = 1\,\text{W}$ — CU transmit power ($30\,\text{dBm}$)
- $s_c(i)$ — unit-power CU signal sample
- $w_k(i) \sim \mathcal{CN}(0, \sigma_k^2)$ — AWGN at NCU receiver

**Thermal noise power:**

$$\sigma_k^2 = k_B \cdot T \cdot B \cdot \text{NF} = 1.38\times10^{-23} \cdot 290 \cdot 10^8 \cdot 5.01 \approx 2.01\times10^{-12}\,\text{W}$$

where NF $= 7\,\text{dB}$ (linear: 5.01).

> **Implementation:** `channel.py :: ChannelModel.sample_slot()`  
> **Parameters:** `cfg.NOISE_POWER_W ≈ 2.01e-12 W`, `cfg.P_CU_W ≈ 1 W`

---

## 2.4 Interference Channel (NCU → CU)

The interference power delivered by NCU $k$ to the CU receiver when NCU $k$ transmits at power $P_k$:

$$I_k = P_k |g_{kb}|^2$$

where $g_{kb}$ is the NCU-to-CU fading channel (Rayleigh, $d = 600\,\text{m}$).

**Aggregate harmful interference condition:**

$$I_{\text{harm}} = \sum_{k \in \mathcal{A}_t} P_k |g_{kb}|^2 > I_{\max} \quad (H_t = 1)$$

with $I_{\max} = -80\,\text{dBm}$ (maximum tolerable interference at the CU receiver).

---

## 2.5 ISAC Radar Signal Model (Sec. 2.4, 4.2)

The ISAC BS transmits a dual-function waveform: half the power for communication, half for radar sensing.

### Radar Range Equation

The radar SNR (per pulse) is:

$$\text{SNR}_r = \frac{P_r \cdot \sigma_{\text{RCS}} \cdot G_t \cdot G_r \cdot \lambda^2}{(4\pi)^3 \cdot R^4 \cdot k_B \cdot T \cdot B}$$

| Parameter | Value |
|-----------|-------|
| $P_r = P_{\max}/2$ | 0.5 W |
| $\sigma_{\text{RCS}}$ | 1.0 m² |
| $G_t = G_r$ | 30 (linear, ≈ 15 dBi) |
| $\lambda = c/f_c$ | ≈ 2.11 mm |
| $R$ | 1000 m (BS to CU) |

**Operating point:** The experiment uses $\text{SNR}_r = 10\,\text{dB}$ as the nominal radar operating point (Theorem 8.1 regime).

### CFAR Test Statistic

Over $L = 128$ integrated radar pulses, the matched-filter output is:

$$\Lambda_{\text{ISAC}} = \frac{Z_{\text{ISAC}}}{\hat{\sigma}_{\text{ref}}^2}$$

where:
- Under $H_0$: $Z_{\text{ISAC}} \sim \chi^2(2L)/(2L)$ (normalised, mean ≈ 1)
- Under $H_1$: $Z_{\text{ISAC}} \sim \chi^2_{\text{nc}}(2L,\, 2L\cdot\text{SNR}_r)/(2L)$
- $\hat{\sigma}_{\text{ref}}^2$ — mean power of $N_{\text{ref}} = 32$ reference cells (CFAR normalization)

> **Implementation:** `channel.py :: ISACRadarModel.compute_isac_statistic()`  
> **Parameters:** `cfg.N_RADAR_PULSES = 128`, `cfg.N_REF_CELLS = 32`

---

## 2.6 Saleh-Valenzuela Sub-THz MIMO Channel (Sec. 12.2)

For the BS MIMO beamforming, the channel matrix $\mathbf{H} \in \mathbb{C}^{N_r \times N_t}$ follows a clustered Saleh-Valenzuela model:

$$\mathbf{H} = \sqrt{\frac{N_t N_r}{N_{\text{cl}} N_{\text{ray}}}} \sum_{\ell=1}^{N_{\text{cl}}} \sum_{p=1}^{N_{\text{ray}}} \alpha_{\ell p}\, \mathbf{a}_r(\varphi_{\ell p}) \mathbf{a}_t^H(\varphi_{\ell p})$$

**ULA steering vector** ($d = \lambda/2$ spacing):

$$\mathbf{a}(\varphi) = \frac{1}{\sqrt{N}} \begin{bmatrix} 1,\, e^{j\pi\sin\varphi},\, \ldots,\, e^{j\pi(N-1)\sin\varphi} \end{bmatrix}^T$$

The channel is normalised so $\|\mathbf{H}\|_F^2 = 1/L(d,f)$.

| Parameter | Value |
|-----------|-------|
| $N_t$ (BS antennas) | 64 (ULA) |
| Clusters $N_{\text{cl}}$ | 3 |
| Rays per cluster $N_{\text{ray}}$ | 5 |
| AoA cluster spread | $\sigma_\varphi = 5°$ |

> **Implementation:** `channel.py :: generate_subthz_channel()`, `steering_vector()`

---

## 2.7 Mathematical Validation Checklist

| Formula | Code | Status |
|---------|------|--------|
| FSPL $= 20\log_{10}(4\pi df/c)$ | `free_space_path_loss_db()` | ✅ |
| Molecular absorption $= \alpha_{\text{abs}} \cdot d/1000$ | `molecular_absorption_loss_db()` | ✅ |
| Nakagami power $\sim \text{Gamma}(m, \Omega/m)$ | `nakagami_amplitude()` | ✅ |
| CU semi-Markov $\pi_1 = \lambda_{\text{on}}/(\lambda_{\text{on}}+\lambda_{\text{off}})$ | `CUActivityModel` | ✅ ($\pi_1 = 0.300$) |
| Radar range equation ($\sigma_{\text{RCS}}^1$, not $\sigma^2$) | `ISACRadarModel.radar_snr_db()` | ✅ (fixed) |
| CFAR statistic under $H_1$: non-central $\chi^2$ | `compute_isac_statistic()` | ✅ |
| Noise power $= k_B T B \cdot \text{NF}$ ≈ $2.01\times10^{-12}$ W | `cfg.NOISE_POWER_W` | ✅ (comment fixed) |
