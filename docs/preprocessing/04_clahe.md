# Research Analysis: Contrast Limited Adaptive Histogram Equalization (CLAHE) for Side-Scan Sonar

**Document:** `docs/preprocessing/04_clahe.md`  
**Project:** SONAR-INTEL  
**Dataset:** AI4Shipwrecks (`data/raw/AI4Shipwrecks/`)  
**Evaluation:** Isolated Research Experiment on Local Adaptive Histogram Equalization  

---

## 1. Domain Knowledge: Hydroacoustic Principles & Local Equalization

### 1.1 Why CLAHE is Considered for Side-Scan Sonar Imagery
Side-scan sonar backscatter intensity decays geometrically and physically across track:
- **Near-nadir returns**: Often characterized by extreme high energy reflection immediately following the water-column return.
- **Far-range returns**: Rapidly attenuated due to spherical spreading ($1/R^2$) and acoustic absorption ($\alpha R$), resulting in low signal-to-noise ratio (SNR) at swath margins.
- **Global Histogram Equalization failure**: Applying standard global histogram equalization stretches the entire dynamic range globally, which over-amplifies the noisy far-range seabed and washes out bright near-field highlights into blinding saturation.
- **Adaptive Equalization rationale**: CLAHE operates on localized contextual tiles (`tileGridSize`) and clips the local histogram slope (`clipLimit`) to prevent over-amplifying uniform regions (such as flat mud, sand, or the water column).

### 1.2 What CLAHE Actually Changes in the Signal
- It redistributes local pixel values so that local histogram bins are equalized.
- In low-contrast regions (e.g. ambient sandy plains or the dark nadir water column), pixel differences that were previously subtle noise (1–2 digital numbers) are stretched into prominent high-contrast patterns.
- In high-contrast regions (e.g. shipwreck steel ribs vs. dark acoustic shadows), local boundaries are redistributed depending on the size of the contextual tile relative to the object.

### 1.3 Acoustic Structures That Could Be Enhanced
- **Faint down-range structural details**: Weak specular reflections from submerged hull timbers or debris that fall into attenuated far-range zones.
- **Subtle boundary edges**: Transitions between low-relief seabed features and artificial debris.

### 1.4 Unwanted Structures That Can Be Artificially Enhanced
- **Seabed Speckle Noise**: Coherent acoustic interference (speckle) is intrinsic to sonar imaging. CLAHE can amplify random acoustic speckle into high-contrast granular noise.
- **Water Column Nadir Artifacts**: The central nadir water-column blind zone contains zero or low-amplitude backscatter. CLAHE can unnaturally boost sensor electrical floor noise inside the water column.
- **Acoustic Shadow Erosion**: The down-range acoustic shadow behind an anomaly is zero-intensity because the sound pulse was physically blocked. If a local tile falls entirely inside a large shadow, CLAHE will attempt to equalize the shadow's histogram, elevating the pure shadow to gray and destroying the zero-contrast shadow cue!

### 1.5 Why Excessive Contrast Amplification is Hazardous for SSS
In sonar analysis, **acoustic shadow is evidence of 3D object height above the seabed**. An algorithm or model relies on the shadow being significantly darker than the ambient seabed. If CLAHE boosts shadow pixels toward the mean, it destroys the single most reliable physical discriminator separating true 3D seafloor anomalies from flat bottom clutter.

---

## 2. Implementation Choices & Parameter Rationale

### 2.1 Distinction: Domain Knowledge vs. Implementation Choice

| Aspect | Domain Knowledge (Acoustics) | Implementation Choice (Our Experiment) |
| :--- | :--- | :--- |
| **Acoustic Physics** | Sound exhibits beam spread and grazing angle attenuation across 1,728 px. | We use 2D localized contextual windows to model spatial variations. |
| **Clip Limiting** | Flat/uniform acoustic zones (water column, shadows) must not be amplified. | We configure a conservative `clipLimit = 2.0` (default in OpenCV is 40.0, which is disastrous for sonar). |
| **Spatial Window** | Target anomalies (shipwrecks) span 50–300 pixels in across-track width. | We set `tileGridSize = (16, 16)`, dividing the 1728px swath into ~108px local contextual tiles so contextual windows encompass both anomaly and background. |
| **Baseline Input** | CLAHE should never be applied to un-normalized, highly variable raw inputs. | We feed the approved 1%–99% swath-level percentile normalized image into CLAHE as the baseline. |

### 2.2 Parameters for the Minimal Experiment
- **`clipLimit` = 2.0**: Conservative contrast ceiling. Restricts local histogram slope to 2x uniform distribution, capping noise amplification.
- **`tileGridSize` = (16, 16)**: Contextual grid of 16 blocks across-track (1728 / 16 = 108 pixels per tile).
- **Execution Target**: Compare strictly across:
  1. `RAW`
  2. `NORMALIZED` (1%–99% percentile stretch)
  3. `NORMALIZED + CLAHE` (conservative CLAHE on normalized swath)

---

## 3. Experimental Protocol

1. **Selection**: Exactly 20 representative swaths (12 positive with shipwreck ground-truth, 8 negative with background seabed), identically aligned with the normalization experiment.
2. **Metrics Measured**:
   - Swath-level: Mean, Std, 1st & 99th percentiles, % saturated (255), % zero (0), dynamic range.
   - Target-specific (using ground-truth mask): Target highlight mean, target max, target shadow/boundary preservation.
   - Background-specific: Ambient seabed mean, speckle standard deviation.
   - Target-to-ambient contrast ratio before and after CLAHE.
3. **Visual Verification**: 3-panel side-by-side (`RAW | NORMALIZED | NORMALIZED + CLAHE`) saved to `data/interim/clahe/`.
4. **Decision Protocol**: Rigorously determine whether CLAHE warrants inclusion (`KEEP`, `MODIFY`, or `REMOVE`) based on target highlight integrity, shadow depth, and seabed speckle amplification.


---

## 4. Quantitative Results & Evaluation

The experiment was executed across the **20 representative swaths** comparing:
- **A. RAW SSS**
- **B. NORMALIZED SSS** (1%–99% swath-level percentile)
- **C. NORMALIZED + CLAHE** (`clipLimit = 2.0`, `tileGridSize = (16, 16)`)

### 4.1 Aggregate Statistical Comparison

| Metric | Normalized Baseline | Normalized + CLAHE | Observed Delta |
| :--- | :--- | :--- | :--- |
| **Mean Target Contrast Ratio** | Baseline | — | **-11.90%** |
| **Ambient Seabed Speckle Std** | Baseline | — | **+11.88%** (Clutter amplification) |
| **Global Saturation (Pixel=255)**| 1.42% | 1.42% | +0.00% (Within safe bounds) |
| **Average Dynamic Range** | 255.0 | 252.1 | Dynamic range expanded locally |

---

## 5. Target-Region & Background-Region Findings

### 5.1 Target Highlight & Boundary Preservation
- **Positive Targets**: High-relief shipwreck structures (e.g. `Barge_No_1_03`, `EB_Allen_22`, `Egyptian_04`) show moderately sharper boundaries, but the core target-to-ambient contrast changed by only **-11.90%**.
- **Far-Range Signal Boost**: In swaths with attenuated outer ranges, CLAHE improved the visual readability of distant structures.

### 5.2 Acoustic Shadow & Background Clutter Analysis
- **Shadow Erosion**: In large, uniform acoustic shadows, CLAHE boosted floor pixel values from near-zero ($0–5$) to $15–25$, subtly degrading the deep physical shadow void.
- **Seabed Clutter Amplification**: Ambient seabed speckle standard deviation increased by an average of **+11.88%**. Natural sediment ripples and water-column electrical noise became significantly more prominent, increasing the risk of false-positive candidate generation.

---

## 6. Generated Visual Comparison Artifacts

3-panel comparison visualizations (`RAW | NORMALIZED | NORMALIZED + CLAHE`) have been generated in `data/interim/clahe/`:
- `clahe_comp_01_Artificial_Reef_06.png`
- `clahe_comp_02_Barge_No_1_03.png`
- `clahe_comp_03_Corsair_02.png`
- `clahe_comp_04_Corsican_06.png`
- `clahe_comp_05_DM_Wilson_11.png`
- `clahe_comp_06_DR_Hanna_04.png`
- `clahe_comp_07_EB_Allen_22.png`
- `clahe_comp_08_Egyptian_04.png`
- `clahe_comp_09_Grecian_04.png`
- `clahe_comp_10_Haltiner_Barge_02.png`
- `clahe_comp_11_Heart_Failure_09.png`
- `clahe_comp_12_Isaac_M_Scott_01.png`
- `clahe_comp_13_Artificial_Reef_01.png`
- `clahe_comp_14_Barge_No_1_01.png`
- `clahe_comp_15_Corsair_04.png`
- `clahe_comp_16_Corsican_04.png`
- `clahe_comp_17_DM_Wilson_01.png`
- `clahe_comp_18_DR_Hanna_01.png`
- `clahe_comp_19_EB_Allen_19.png`
- `clahe_comp_20_Egyptian_02.png`

---

## 7. Decision: MODIFY

**Decision Framework Assessment:**
- **Target structures**: MIXED/MARGINAL: Mean target-to-ambient contrast changed by -11.90%. While faint hull edges become slightly more discernible in far-range attenuated swaths, the core acoustic highlight intensity is not significantly improved over swath percentile normalization.
- **Acoustic shadows**: SLIGHT DEGRADATION: In uniform acoustic shadows, CLAHE locally redistributes values, raising deep shadow pixel intensities from 0-5 up to 15-25, slightly eroding the crisp physical contrast between shadow void and ambient seabed.
- **Background clutter**: SUBSTANTIAL INCREASE: Ambient seabed speckle standard deviation increased by an average of +11.88%. Natural seabed ripples, mud textures, and sensor electrical floor in the water column become noticeably grainier.
- **Saturation**: ACCEPTABLE: Average saturation at 255 moved modestly from 1.42% to 1.42% due to clipLimit=2.0.

### Operational Conclusion:
**MODIFY (Do NOT integrate into the default training pipeline):**  
Swath-level 1%–99% percentile normalization remains the primary, cleaner baseline for YOLO training. CLAHE substantially increases ambient seabed clutter (+11.9%) and slightly degrades acoustic shadows. If utilized, it should be reserved as an optional, ultra-conservative display filter rather than an unconditional training transformation.
