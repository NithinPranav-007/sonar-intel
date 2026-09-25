# Research Analysis: SSS Normalization & Domain Methodology

**Document:** `docs/preprocessing/03_normalization.md`  
**Project:** SONAR-INTEL  
**Dataset:** AI4Shipwrecks (`data/raw/AI4Shipwrecks/`)  
**Domain Reference:** [SidescanTools](https://github.com/sonoware/sidescantools) (`SidescanPreprocessor`)  

---

## 1. Executive Summary

This document establishes the theoretical and physical baseline for normalizing side-scan sonar (SSS) imagery. 

Side-scan sonar backscatter differs fundamentally from optical imagery. In true hydroacoustic processing (e.g. `SidescanTools`), normalization involves multi-stage physical corrections: bottom tracking (First Bottom Return / FBR), geometric slant-range projection, empirical gain normalization (EGN) across acoustic incidence angles, and ping-wise energy balancing.

In the **AI4Shipwrecks** dataset, however, we are provided **only 8-bit single-channel PNG waterfall swaths without raw sensor telemetry (no altitude, slant range, or GPS)**. This document analyzes what `SidescanTools` does, compares its assumptions against the AI4Shipwrecks data reality, and determines what simplified, image-domain normalization is scientifically defensible.

---

## 2. Analysis of the Reference: SidescanTools

### 2.1 What Normalization Means in the SidescanTools Context
In `SidescanTools`, "normalization" is not a simple image contrast adjustment. It refers to a suite of physical corrections:
1. **Per-Ping Peak Normalization**: Dividing each ping vector by its maximum value (`portside / indv_max_portside[:, None]`) to facilitate threshold-based bottom-line detection.
2. **Beam Pattern & Empirical Gain Normalization (EGN)**: Correcting for acoustic attenuation as grazing angles decrease away from nadir. An angular response table (`egn_table[r_idx, alpha_idx]`) models backscatter as a function of slant range $r$ and grazing angle $\alpha$.
3. **Slant-Range Correction**: Projecting slant-range acoustic travel times onto horizontal ground range assuming a flat seabed:
   $$x_{ground} = \sqrt{r_{slant}^2 - h_{altitude}^2}$$
   This removes the hyperbolic distortion and interpolates the central nadir water-column blind zone.
4. **Energy Normalization**: Running a moving-window RMS power filter across pings (`np.sum(son_dat**2)`) to balance along-track acoustic power variations.

### 2.2 Input Representation Expected by SidescanTools
- **File Types**: Raw vendor formats such as eXtended Triton Format (`.xtf`) or Klein Sonar Format (`.jsf`).
- **Data Array**: A 3D floating-point NumPy array of shape `[num_channels, num_pings, ping_length]` (typically `num_channels = 2`: Port and Starboard).
- **Values**: Raw acoustic transducer voltage amplitudes, digitized analog-to-digital converter (ADC) counts, or logarithmic decibel ($\text{dB}$) conversions.

### 2.3 Operating Domain: Amplitudes vs. dB vs. Quantized Images
`SidescanTools` operates primarily on:
- **Raw floating-point acoustic amplitudes** ($A \in \mathbb{R}^+$)
- **Decibel ($\text{dB}$) transformed arrays**: $20 \log_{10}(A)$
- It does **not** operate on pre-rendered, gamma-corrected, 8-bit PNG images.

### 2.4 Physical Assumptions Made by SidescanTools
To perform its corrections, `SidescanTools` relies on:
1. **Sensor Primary Altitude ($h_{alt}$)**: Depth of towfish/AUV above seabed in meters, recorded by an altimeter or DVL.
2. **Slant-Range Vector ($r_{slant}$)**: Known physical distance in meters corresponding to the sampling window.
3. **Channel Separation**: Separate, unmixed Port and Starboard acoustic channels.
4. **Continuous Heading and Geodesic Position**: Longitude and Latitude GPS logs to calculate metric along-track resolution ($m/\text{ping}$).
5. **Flat-Bottom Assumption**: Seabed is assumed planar across the swath for geometric ground-range projection.

---

## 3. Gap Analysis: SidescanTools Assumptions vs. AI4Shipwrecks

| Assumption / Requirement in SidescanTools | Available in AI4Shipwrecks? | Impact / Limitation |
| :--- | :--- | :--- |
| **Raw float / dB acoustic amplitude** | **NO** | Data is pre-rendered and quantized into 8-bit integers (`uint8` $\in [0, 255]$). Non-linear gamma and clipping have already occurred. |
| **Towfish altitude ($h_{alt}$)** | **NO** | Physical slant-range Pythagorean projection cannot be computed. |
| **Physical slant range in meters** | **NO** | Exact metric across-track pixel scale is uncalibrated. |
| **Vessel / AUV GPS coordinates** | **NO** | Metric along-track scaling and geodetic interpolation are impossible. |
| **Separate Port & Starboard channels** | **PARTIALLY** | Left (Port) and Right (Starboard) are fused into a single 1,728px swath with nadir in the center, but channel boundaries are not separated in metadata. |
| **Angular Beam Pattern Metadata** | **NO** | Physical EGN angular correction table cannot be constructed. |

---

## 4. Scientifically Defensible Normalization Concept for AI4Shipwrecks

Because the physical acoustic metadata is absent, **we must not claim or attempt physical radiometric calibration, slant-range ground projection, or EGN correction**. Doing so would require inventing arbitrary sensor altitudes and slant ranges, violating scientific domain honesty.

### Defensible Concept: Swath-Level Robust Percentile Contrast Normalization
Instead, a defensible **image-domain normalization** must satisfy:
1. **Radiometric Honesty**: Treat the data purely as uncalibrated 8-bit acoustic intensity swaths.
2. **Preservation of Acoustic Highlight vs. Shadow**: The core acoustic signature of an anthropogenic target (shipwreck, debris, cargo) is an intense acoustic backscatter return (highlight) paired with a down-range deficit (acoustic shadow). Normalization must **never** invert, compress, or destroy this highlight-to-shadow contrast.
3. **No Ping-Wise Line Streaking**: In raw sonar, per-ping normalization (`ping / max(ping)`) works on ADC voltage, but applying per-ping division to 8-bit images amplifies noise in dark shadow pings and creates artificial along-track banding. Therefore, normalization must be applied at the **swath level**.
4. **Robustness to Extreme Speckle Spikes**: Acoustic speckle can generate isolated single-pixel values at 0 or 255. Using a robust percentile range (e.g. 1.0% to 99.0% or 0.5% to 99.5%) prevents single-pixel noise from dominating the dynamic range.

### Mathematical Formulation
For swath $I$:
$$p_{low} = \text{Percentile}(I, 1.0), \quad p_{high} = \text{Percentile}(I, 99.0)$$
$$I_{norm} = \text{clip}\left(\frac{I - p_{low}}{p_{high} - p_{low}} \times 255.0, \; 0, \; 255\right)$$

This linear contrast stretch expands the usable dynamic range of darker swaths while preserving the linear relationship between target highlight, ambient seabed, and acoustic shadow.

---

## 5. Experimental Protocol

To validate whether this defensible normalization should be kept, modified, or removed, an isolated experiment (`ml/preprocessing/03_normalize.py`) is conducted across **20 representative swaths**:
- 12 positive swaths with ground-truth shipwrecks
- 8 negative swaths with ambient seabed textures
- Spanning multiple sites, swath lengths (773px to 18,745px), and baseline mean intensities (24.2 to 90.9)
- Target preservation test: Measuring target-to-ambient contrast ratio and shadow deficit before and after normalization.
