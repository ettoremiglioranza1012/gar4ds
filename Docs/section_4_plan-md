# Results and Discussion Section: Structural Plan

**Document Purpose:** Roadmap for drafting "Experimental Evaluation and Results Discussion"  
**Guiding Principle:** **Strict 1:1 correspondence with methodology section**  
**Date:** February 4, 2026

---

## 🎯 OVERARCHING STRATEGY

### Core Principle: Methodology-Results Alignment

**Every subsection in Methods § must have a corresponding Results §**

```
Methodology §3.1 → Results §4.1
Methodology §3.2 → Results §4.2
Methodology §3.3 → Results §4.3
Methodology §3.4 → Results §4.4
```

**Rule:** If a method was described, it MUST produce results. If results are shown, the method MUST have been described.

### Tone and Style Guidelines

**Methodology = Future tense / Declarative**
> "We will compute Global Moran's I..."  
> "The SDM is estimated using MLE..."

**Results = Past tense / Evidentiary**
> "Global Moran's I was computed and yielded I = 0.601..."  
> "The SDM was estimated and produced ρ = 0.0757..."

**Discussion = Present tense / Interpretive**
> "The low ρ value indicates that meteorology explains most spatial correlation..."  
> "These findings suggest that cross-border transport contributes 7.6% to PM₁₀..."

---

## 📐 DETAILED SECTION STRUCTURE

---

## **4. EXPERIMENTAL EVALUATION AND RESULTS DISCUSSION**

### **4.1 Spatial Weights Matrix and Network Topology**

**Maps to:** Methodology §3.1 (Spatial Weighting and Connectivity)

#### 4.1.1 Descriptive Statistics

**Content:**
- Confirm W matrix dimensions: 36 × 36
- Verify row-standardization: all rows sum to 1.0
- Report connectivity statistics:
  - Mean neighbors per station: 6.00
  - Min/max neighbors: 6 (by construction with KNN)
  - Network density: 6/35 = 0.171
  
**Visualization:**
- Network graph showing station connections (optional)
- Spatial map with KNN links overlaid (if helpful)

**Example Text:**
```latex
The spatial weights matrix W was successfully constructed using k=6 nearest 
neighbors, yielding a 36×36 connectivity matrix. Row-standardization was 
verified (∑_j w_ij = 1.000 ± 0.001 for all i). The resulting network exhibits 
uniform connectivity by design, with each station connected to exactly 6 
neighbors, producing a network density of 0.171.
```

**Cross-Reference:** "As specified in §3.1, the KNN criterion ensures..."

---

### **4.2 Global and Local Spatial Autocorrelation**

**Maps to:** Methodology §3.2

#### 4.2.1 Global Moran's I Results

**Content:**
- Present Global Moran's I for all 7 variables in a table
- Report I statistic, p-value, and significance level
- Confirm/reject null hypothesis of spatial randomness
- Highlight strongest vs. weakest spatial correlation

**Table Format:**
```
Table 1: Global Moran's I Statistics for PM₁₀ and Meteorological Variables

Variable   Moran's I   P-value   Significant?   Interpretation
--------   ---------   -------   ------------   ---------------
PM10       0.6012      0.001     Yes (α<0.01)   Strong clustering
WS         0.6792      0.001     Yes (α<0.01)   Strong clustering
PRESS      0.6565      0.001     Yes (α<0.01)   Strong clustering
TEMP       0.5283      0.001     Yes (α<0.01)   Moderate clustering
BLH        0.4666      0.001     Yes (α<0.01)   Moderate clustering
RAD        0.3189      0.002     Yes (α<0.01)   Weak clustering
PRECIP     0.2941      0.002     Yes (α<0.01)   Weak clustering
```

**Example Discussion:**
```latex
All seven variables exhibited statistically significant positive spatial 
autocorrelation (p < 0.01), rejecting the null hypothesis of spatial 
randomness. PM₁₀ concentrations showed strong clustering (I = 0.601), 
confirming that elevated pollution levels are not randomly distributed but 
instead occur in spatially coherent patterns. Among meteorological variables, 
wind speed demonstrated the strongest spatial structure (I = 0.679), while 
precipitation exhibited the weakest yet still significant clustering (I = 0.294). 
These findings validate the fundamental assumption of spatial econometric 
models: that neighbor effects are non-negligible and must be explicitly 
incorporated into regression specifications [cite §3.2].
```

**Key Interpretation Points:**
1. All variables cluster → justifies spatial models
2. Meteorological clustering → explains why PM₁₀ clusters (shared atmospheric conditions)
3. Range of I values → some variables more spatially structured than others
4. **Critical:** This does NOT prove transport (as noted in §3.2)—it proves spatial structure

---

#### 4.2.2 Local Moran's I (LISA) Results

**Content:**
- Present LISA classification counts (HH, LL, HL, LH) for all 7 variables
- Focus visualization on PM₁₀ LISA map
- Identify HH clusters (pollution hotspots)
- Note meteorological LISA was computed but not mapped (as per §3.2)

**Table Format:**
```
Table 2: LISA Cluster Classification Summary

Variable   High-High   Low-Low   High-Low   Low-High   Not Sig.
--------   ---------   -------   --------   --------   --------
PM10       6           15        0          1          14
TEMP       5           8         0          0          23
BLH        5           10        0          1          20
WS         9           15        0          1          11
PRECIP     6           3         0          0          27
PRESS      12          14        0          0          10
RAD        3           8         0          0          25
```

**Visualization:**
- **Figure 1:** PM₁₀ LISA cluster map showing HH, LL, HL, LH zones
  - Color scheme: Red (HH), Blue (LL), Pink (HL), Light Blue (LH), Gray (not sig.)
  - Overlay administrative boundaries (Lombardy-Veneto-Trentino)
  - Label key HH stations (especially Trentino target stations)

**Example Discussion:**
```latex
Local Moran's I analysis identified 6 High-High (HH) clusters for PM₁₀, 
representing statistically significant pollution hotspots where elevated 
concentrations at a station are surrounded by similarly elevated neighbors 
(Figure 1). These HH clusters are predominantly located at [describe geographic 
pattern—e.g., "the Lombardy-Veneto border and along the Adige Valley corridor"]. 
Conversely, 15 stations exhibited Low-Low (LL) clustering, forming coherent 
zones of relatively clean air in [describe locations—e.g., "high-altitude and 
rural background sites"]. 

The presence of only 1 Low-High (LH) outlier and 0 High-Low (HL) outliers 
indicates that spatial discontinuities are rare: stations tend to resemble their 
neighbors rather than standing in stark contrast to them. This reinforces the 
Global Moran's I finding of strong positive spatial autocorrelation.

As specified in §3.2, LISA statistics were also computed for all meteorological 
variables to validate spatial structure (Table 2), but cartographic outputs were 
generated only for PM₁₀, as the primary analytical focus is pollutant 
distributions rather than atmospheric field mapping.
```

**Key Interpretation Points:**
1. HH clusters = potential receptor zones (if meteorology favorable for dispersion) OR source zones (if meteorology favors stagnation)
2. Need regime classification (§4.3) to distinguish these scenarios
3. LL clusters = clean air zones (high altitude, rural, good ventilation)
4. Spatial outliers (HL, LH) are rare → smooth spatial gradients

**Cross-Reference:** "The interpretation of HH clusters requires cross-referencing with atmospheric regime classification (§4.3), as described in §3.2."

---

### **4.3 Multivariate Regime Identification**

**Maps to:** Methodology §3.3

#### 4.3.1 Optimal Cluster Number Selection

**Content:**
- Report elbow method results (WCSS vs. k plot)
- Justify k=4 selection
- Show silhouette scores or other validation metrics (optional)

**Visualization:**
- **Figure 2a:** WCSS elbow plot showing "elbow" at k=4

**Example Text:**
```latex
K-means clustering was performed for k ∈ {2, 3, 4, 5, 6}, with within-cluster 
sum of squares (WCSS) computed for each configuration. The elbow method 
identified k=4 as the optimal partition (Figure 2a), beyond which incremental 
reductions in WCSS became marginal (diminishing returns). This four-regime 
solution balances model parsimony with meaningful environmental differentiation.
```

---

#### 4.3.2 Cluster Assignments and Spatial Distribution

**Content:**
- Report cluster sizes (n stations per cluster)
- Show spatial map of cluster assignments
- Describe geographic distribution of each regime

**Table Format:**
```
Table 3: Atmospheric Regime Cluster Assignments

Cluster   N_Stations   Geographic Distribution
-------   ----------   -----------------------
0         13           [Describe—e.g., "Po Valley border, low elevation"]
1         10           [Describe—e.g., "Veneto plains, moderate elevation"]
2         6            [Describe—e.g., "Alpine valleys, high elevation"]
3         7            [Describe—e.g., "Adige corridor, intermediate elevation"]
```

**Visualization:**
- **Figure 2b:** Spatial map with stations color-coded by cluster assignment
  - Overlay LISA HH clusters to preview §4.3.4 integration

**Example Text:**
```latex
The K-means algorithm partitioned the 36 stations into four distinct atmospheric 
regimes. Cluster 0 (n=13 stations) is predominantly located in [geographic 
description]. Cluster 1 (n=10) comprises [description]. Cluster 2 (n=6), the 
smallest group, consists of [description]. Cluster 3 (n=7) spans [description]. 

Figure 2b reveals clear spatial coherence: stations within the same cluster tend 
to be geographically proximate, confirming that the multivariate similarity 
captured by K-means corresponds to meaningful atmospheric regionalization rather 
than random assignment.
```

---

#### 4.3.3 Principal Component Analysis (PCA)

**Content:**
- Report variance explained by PC1 and PC2
- Interpret principal components (loadings)
- Show PCA scatter plot with cluster separation

**Visualization:**
- **Figure 3:** PCA scatter plot (PC1 vs. PC2)
  - Points color-coded by cluster
  - Convex hulls or 95% confidence ellipses around each cluster
  - Annotate axes with top loadings (e.g., "PC1: BLH+ / PRESS-")

**Example Text:**
```latex
Principal Component Analysis (PCA) was applied to visualize the four-cluster 
solution in reduced dimensionality. The first two principal components explained 
90.04% of total variance (PC1: 63.67%, PC2: 26.37%), indicating that the 
six-dimensional feature space can be adequately represented in two dimensions 
for visualization purposes.

PC1 loadings were dominated by [describe—e.g., "positive contributions from 
pressure and PM₁₀, with negative contributions from BLH"], suggesting this axis 
captures a gradient from stagnant, polluted conditions (high PC1) to well-mixed, 
clean conditions (low PC1). PC2 loadings reflected [describe—e.g., "temperature 
and precipitation contrasts"], likely representing seasonal or elevation-driven 
variability.

Figure 3 demonstrates clear separation between clusters in PC space, with 
minimal overlap. Cluster 0 and Cluster 3 occupy the high-PM₁₀ / high-pressure 
region (positive PC1), while Clusters 1 and 2 reside in the low-PM₁₀ / high-BLH 
region (negative PC1). This visual confirmation supports the validity of the 
K-means partition.
```

---

#### 4.3.4 Cluster Environmental Profiles

**Content:**
- Present detailed regime characterization table (mean ± std for each variable)
- Classify regimes as "High Pollution" vs. "Low Pollution"
- Interpret meteorological conditions for each regime

**Table Format:**
```
Table 4: Atmospheric Regime Environmental Profiles

Cluster   PM10 (μg/m³)   TEMP (K)     BLH (m)      WS (m/s)    PRECIP (mm/hr)   PRESS (Pa)    Characterization
-------   ------------   --------     -------      --------    --------------   ----------    ----------------
0         29.6 ± 2.8     286.6 ± 0.8  321 ± 12     1.58 ± 0.40  0.000143        99498 ± 1666  High Pollution / Stagnation
1         18.9 ± 3.8     281.8 ± 2.8  329 ± 15     0.80 ± 0.20  0.000173        89686 ± 3990  Low Pollution / Low Elevation
2         16.8 ± 1.7     281.4 ± 2.5  401 ± 10     0.72 ± 0.10  0.000127        88846 ± 3078  Low Pollution / Alpine
3         30.8 ± 2.5     287.5 ± 0.2  389 ± 8      1.68 ± 0.08  0.000117        100380 ± 437  High Pollution / Transport Corridor
```

**Example Discussion:**
```latex
Table 4 presents the environmental profiles for each atmospheric regime, averaged 
across all stations within each cluster. Two regimes (Clusters 0 and 3) exhibit 
high PM₁₀ concentrations (≈30 μg/m³), while two regimes (Clusters 1 and 2) show 
low pollution levels (≈17-19 μg/m³).

**Cluster 0 (High Pollution / Stagnation):** This regime is characterized by 
moderately shallow boundary layers (321 m), high surface pressure (99.5 kPa), 
and moderate wind speeds (1.58 m/s). The combination of high pressure and low 
BLH suggests atmospheric stability with limited vertical mixing—conditions that 
trap locally emitted pollutants near the surface. The high PM₁₀ levels in this 
regime are consistent with local accumulation rather than transport.

**Cluster 1 (Low Pollution / Low Elevation):** Despite low PM₁₀ (18.9 μg/m³), 
this regime exhibits low wind speeds (0.80 m/s) and shallow BLH (329 m), which 
would typically favor pollution accumulation. The paradoxically low pollution 
levels suggest either (1) low emission density in these areas, or (2) episodic 
ventilation events not captured in the mean statistics. The low pressure 
(89.7 kPa) indicates higher elevation or frequent passage of low-pressure systems.

**Cluster 2 (Low Pollution / Alpine):** This regime represents high-altitude 
stations with deep boundary layers (401 m), low pressure (88.8 kPa), and very 
low PM₁₀ (16.8 μg/m³). The deep BLH at high elevation is physically consistent: 
mountain stations often sit above the nocturnal inversion layer, experiencing 
free tropospheric air masses with minimal pollution. Low precipitation 
(0.000127 mm/hr) suggests limited wet deposition, yet pollution remains low 
due to elevation and distance from sources.

**Cluster 3 (High Pollution / Transport Corridor):** This regime combines 
high PM₁₀ (30.8 μg/m³) with deep BLH (389 m) and high wind speeds (1.68 m/s)—
meteorological conditions that should favor dispersion. This mismatch between 
expected (low pollution) and observed (high pollution) conditions is the 
**hallmark signature of transported pollution**. The deep BLH and strong winds 
suggest these stations are downwind receptors receiving pollutants advected from 
upwind source regions. This regime likely corresponds to the Adige Valley 
corridor, where northerly winds channel Po Valley emissions into Trentino.
```

**Key Interpretation:**
- Regimes 0 and 3: High PM₁₀, but different mechanisms (stagnation vs. transport)
- Regimes 1 and 2: Low PM₁₀, rural/alpine clean zones
- **Critical insight:** Cluster 3 = transport signature (high pollution despite favorable meteorology)

---

#### 4.3.5 Integration with LISA Results

**Content:**
- Cross-reference LISA HH clusters with regime assignments
- Identify which HH clusters fall in which regimes
- Interpret HH+Regime combinations

**Table Format:**
```
Table 5: LISA HH Cluster Stations and Their Atmospheric Regimes

Station Name               LISA Class   Cluster   PM10 (μg/m³)   Interpretation
------------------------   ----------   -------   ------------   --------------
[Station A]                HH           0         [value]        Local accumulation (stagnation)
[Station B]                HH           3         [value]        Transported pollution (corridor)
[Station C]                HH           0         [value]        Local accumulation (stagnation)
...                        ...          ...       ...            ...
```

**Example Discussion:**
```latex
To distinguish between local accumulation and cross-border transport, we 
cross-referenced the 6 PM₁₀ LISA High-High (HH) clusters with their atmospheric 
regime assignments (Table 5). 

Of the 6 HH stations:
- 3 stations belong to Cluster 0 (High Pollution / Stagnation): These hotspots 
  are explained by local meteorological conditions that inhibit dispersion. The 
  HH pattern reflects shared stagnation rather than pollution transport.
  
- 2 stations belong to Cluster 3 (High Pollution / Transport Corridor): These 
  HH clusters occur in regions with favorable dispersion meteorology (deep BLH, 
  high WS), yet exhibit high pollution. This spatial-meteorological mismatch 
  provides strong evidence of transported pollution from upwind sources.
  
- 1 station belongs to Cluster 1 (Low Pollution): This anomalous HH classification 
  may reflect a local emission hotspot (e.g., traffic intersection, industrial 
  facility) not captured by regional meteorology.

This integration demonstrates that LISA HH clusters have heterogeneous origins: 
some arise from shared local stagnation (Cluster 0), while others signal 
cross-border transport (Cluster 3). Without regime classification, all HH 
clusters would appear equivalent—obscuring the mechanistic distinction central 
to this study.
```

**Cross-Reference:** "As anticipated in §3.2, the key to distinguishing between co-located pollution and directional transport lies in this cross-referencing of LISA patterns with regime classification."

---

### **4.4 Spatial Durbin Model: Spillover Quantification**

**Maps to:** Methodology §3.4

#### 4.4.1 Model Estimation and Fit Statistics

**Content:**
- Report estimated parameters: ρ, β, θ
- Show standard errors, t-statistics, p-values
- Report model fit: Log-likelihood, AIC
- Compare to baseline OLS (no spatial terms) if helpful

**Table Format:**
```
Table 6: Spatial Durbin Model Parameter Estimates

Parameter          Coefficient   Std. Error   t-statistic   P-value   Interpretation
---------          -----------   ----------   -----------   -------   --------------
ρ (Spatial Lag)    0.0757        [SE]         [t]           [p]       Weak spillover
β_BLH              -0.3798       [SE]         [t]           [p]       ↑ BLH → ↓ PM₁₀
β_TEMP             -2.2052       [SE]         [t]           [p]       Complex effect
β_WS               +0.7239       [SE]         [t]           [p]       Transport > dispersion
β_PRECIP           -0.0884       [SE]         [t]           [p]       Weak washout
β_PRESS            +6.7710       [SE]         [t]           [p]       ↑ Stagnation → ↑ PM₁₀
β_RAD              +1.0366       [SE]         [t]           [p]       Photochemistry
θ_BLH              [value]       [SE]         [t]           [p]       Neighbor BLH effect
θ_TEMP             [value]       [SE]         [t]           [p]       Neighbor TEMP effect
...                ...           ...          ...           ...       ...

Model Fit:
  Log-Likelihood: -85.1196
  AIC: 186.2393
  Pseudo-R²: [value]
```

**Example Discussion:**
```latex
Table 6 presents the Maximum Likelihood estimates for the Spatial Durbin Model. 
The spatial autoregressive parameter ρ = 0.0757 (p < [value]) indicates that 
7.6% of PM₁₀ variance is attributable to spatial spillover from neighboring 
stations, **after controlling for local and neighbor meteorological conditions**. 
This relatively low ρ value suggests that the strong spatial autocorrelation 
observed in Global Moran's I (I = 0.601, §4.2.1) is predominantly explained by 
the spatial structure of atmospheric conditions rather than by direct pollution 
transfer mechanisms.

Among the direct effect coefficients (β), boundary layer height exhibits the 
expected negative relationship with PM₁₀ (β_BLH = -0.380, p < [value]): deeper 
mixing volumes reduce surface concentrations through vertical dilution. Surface 
pressure shows a strong positive effect (β_PRESS = +6.77, p < [value]), 
consistent with high-pressure systems promoting atmospheric stability and 
pollutant accumulation. 

Notably, wind speed demonstrates a **positive** coefficient (β_WS = +0.724, 
p < [value]), contrary to the naive expectation that higher winds should disperse 
pollution. This counterintuitive result supports the transport hypothesis: in 
this network configuration, increased wind speeds enhance advection of pollutants 
from upwind sources (Lombardy border stations) to downwind receptors (Trentino 
valley stations) more than they promote local dispersion. This effect is 
precisely what the SDM is designed to detect.

The exogenous interaction terms (θ) capture neighbor meteorological influences. 
[Interpret key θ coefficients based on your actual results—e.g., "θ_WS > 0 
indicates that high wind speeds at neighboring stations increase local PM₁₀, 
consistent with advective transport."]

Model fit statistics indicate [good/acceptable/poor] performance: AIC = 186.24, 
Log-Likelihood = -85.12. [Compare to OLS baseline if available: "Compared to 
OLS (AIC = [value]), the SDM achieves superior fit, justifying the inclusion 
of spatial terms."]
```

**Key Interpretation Points:**
1. **ρ = 0.076 is LOW** → meteorology explains most spatial correlation
2. **β_WS > 0** → transport dominates dispersion in this network
3. **β_PRESS > 0, β_BLH < 0** → expected atmospheric dispersion physics
4. **θ coefficients** → neighbor meteorology matters (exogenous spillover)

---

#### 4.4.2 Decomposition: Local vs. Spillover Contributions

**Content:**
- Present decomposition results for all 36 stations
- Focus on 4 Trentino target stations
- Show spillover magnitude (μg/m³) and percentage (%)

**Table Format:**
```
Table 7: PM₁₀ Decomposition into Local and Spillover Components (Target Stations)

Station              Observed PM₁₀   Local (Meteo)   Spillover   Spillover %   Regime
------------------   -------------   -------------   ---------   -----------   ------
Borgo Valsugana      10.12           16.05           1.46        14.5%         [Cluster]
Monte Gaza           22.56           14.17           1.47        6.5%          [Cluster]
Parco S. Chiara      19.66           18.77           1.40        7.1%          [Cluster]
Piana Rotaliana      22.98           17.10           1.46        6.3%          [Cluster]

Network Average      24.74           21.92           1.82        7.6%          —
```

**Visualization:**
- **Figure 4a:** Bar chart showing Local vs. Spillover for target stations
- **Figure 4b:** Scatter plot: Observed PM₁₀ vs. Spillover % for all 36 stations
  - Color-code by regime
  - Highlight target stations
  - Add regression line to show any trend

**Example Discussion:**
```latex
Equation (7) in §3.4.3 was applied to decompose observed PM₁₀ into local 
meteorological contributions and spatial spillover effects. Results for the 
four Trentino target stations are presented in Table 7.

**Borgo Valsugana** exhibits the highest spillover percentage (14.5%), despite 
having the lowest absolute PM₁₀ concentration (10.12 μg/m³). The local 
meteorological prediction exceeds the observed value (16.05 > 10.12), suggesting 
that favorable local dispersion conditions partially offset pollution received 
from neighbors. The 1.46 μg/m³ spillover represents cross-border transport from 
[identify key neighbors from §4.4.3].

**Monte Gaza, Parco S. Chiara, and Piana Rotaliana** show moderate spillover 
percentages (6.3–7.1%), closely tracking the network average (7.6%). For these 
stations, local meteorological conditions account for the vast majority of 
observed PM₁₀ (85–94%), indicating that atmospheric accumulation or local 
emissions dominate over transported pollution.

Across the entire network (n=36 stations), spillover contributions range from 
5.7% to 14.5%, with a mean of 7.6% (±[std]). Figure 4b reveals no strong 
correlation between observed PM₁₀ levels and spillover percentage (R² = [value]), 
suggesting that high-pollution stations are not necessarily high-spillover 
stations. Instead, spillover magnitude depends on (1) neighbor pollution levels, 
(2) spatial weights configuration, and (3) the spatial autoregressive parameter ρ.
```

**Key Interpretation:**
1. Most stations: 85-95% local, 5-15% spillover
2. Borgo Valsugana anomaly: meteorology predicts HIGHER than observed → ventilation + spillover cancel out
3. Network average 7.6% matches ρ = 0.076 → internal consistency
4. Low spillover % does NOT mean "no transport"—it means transport is a small fraction of TOTAL pollution

---

#### 4.4.3 Spillover Source Attribution

**Content:**
- Identify top contributing neighbors for each target station
- Show meteorological context (BLH, WS) for each neighbor
- Classify transport mechanisms

**Table Format:**
```
Table 8: Top Spillover Contributors to Trentino Target Stations

Target: Borgo Valsugana (Total Spillover: 1.46 μg/m³)
Neighbor                PM₁₀   BLH (m)   WS (m/s)   Contribution   Mechanism
----------------------  -----  -------   --------   ------------   ---------
Piana Rotaliana         23.0   338       0.63       0.38 μg/m³     Mixed factors
Rovereto                19.7   338       0.63       0.33 μg/m³     Mixed factors
Parco S. Chiara         19.7   320       0.74       0.33 μg/m³     Mixed factors
[Continue for 6 neighbors...]

Target: Monte Gaza (Total Spillover: 1.47 μg/m³)
[Repeat structure...]
```

**Example Discussion:**
```latex
Table 8 decomposes spatial spillover into contributions from individual 
neighboring stations for each Trentino target. For **Borgo Valsugana**, the 
top contributor is Piana Rotaliana (0.38 μg/m³), followed by Rovereto and 
Parco S. Chiara (0.33 μg/m³ each). These three neighbors account for [X%] of 
total spillover received by Borgo Valsugana.

Notably, all three top contributors exhibit moderate wind speeds (0.63–0.74 m/s) 
and shallow boundary layers (320–338 m), conditions that marginally favor 
accumulation over dispersion. The "Mixed factors" mechanism classification 
indicates that spillover from these stations reflects neither pure emission-driven 
transport (high PM₁₀ + high WS) nor pure meteorological propagation (low BLH 
spreading to neighbors), but rather a combination of both.

For **Piana Rotaliana** (the highest-PM₁₀ target station), the top spillover 
contributor is Monte Gaza (0.38 μg/m³), which itself exhibits high pollution 
(22.6 μg/m³) and moderate wind speeds. This bidirectional influence—where Piana 
Rotaliana contributes to Borgo Valsugana, and Monte Gaza contributes to Piana 
Rotaliana—illustrates the network feedback effects captured by the spatial 
multiplier (I - ρW)⁻¹ in Equation (6).

**Cross-Border Transport Evidence:** While the output labels most contributions 
as "Mixed factors," the presence of any spillover from Lombardy border stations 
(e.g., [name specific Lombardy stations if they appear in top-6 neighbors]) to 
Trentino targets would constitute direct evidence of cross-border transport. 
[If applicable:] "Station X (Lombardy) contributes 0.Y μg/m³ to Station Z 
(Trentino), representing [Z%] of total spillover—a clear cross-regional 
pollution pathway."
```

**Key Points:**
1. Top contributors are usually nearest neighbors (by KNN construction)
2. Spillover magnitude = f(neighbor PM₁₀, spatial weight, ρ)
3. Look for Lombardy → Trentino links as smoking gun for cross-border transport
4. Bidirectional spillover is common (network feedback)

---

#### 4.4.4 Network Centrality: Pollution Exporters

**Content:**
- Identify top pollution exporters (stations that contaminate many neighbors)
- Show their regime classification
- Discuss if they align with expected source areas

**Table Format:**
```
Table 9: Top Pollution Exporters (Highest Network Centrality)

Station                        PM₁₀ (μg/m³)   Export Score   Regime   Location
-----------------------------  ------------   ------------   ------   --------
Pavia-p.zza_Minerva_PM10       34.87          34.87          0        Lombardy
VE_Tagliamento                 34.47          34.47          ?        Veneto
TV_S_Agnese                    33.47          33.47          ?        Veneto
Vigevano-via_Valletta_PM10     32.11          32.11          0        Lombardy
Mantova-p.zza_Gramsci_PM10     31.64          31.64          0        Lombardy
```

**Example Discussion:**
```latex
Table 9 ranks stations by their "export score," defined as the product of 
observed PM₁₀ and network centrality (number of stations influenced). The top 
5 exporters are all located in Lombardy or Veneto, with PM₁₀ concentrations 
exceeding 31 μg/m³—well above the network average of 24.7 μg/m³.

**Pavia** (34.9 μg/m³) emerges as the single largest pollution exporter, 
consistent with its location in the central Po Valley—a region characterized by 
intense industrial activity, high traffic density, and frequent atmospheric 
stagnation [cite Diemoz et al., 2019]. This station belongs to Cluster 0 
(High Pollution / Stagnation), confirming that its elevated pollution stems from 
local accumulation under unfavorable dispersion conditions.

Notably, **no Trentino stations** appear in the top-10 exporters, indicating 
that the Alpine valleys act primarily as receptor zones rather than source 
regions. This asymmetry supports the core hypothesis: pollution is generated in 
the Po Valley (source) and transported to Trentino (receptor) under specific 
meteorological conditions.

The dominance of Cluster 0 (Stagnation) and Cluster 3 (Transport Corridor) 
stations among exporters suggests that pollution export occurs through two 
mechanisms: (1) local accumulation that spills over to nearby stations during 
stagnation events, and (2) active advection along transport corridors during 
windy periods.
```

**Key Points:**
1. Top exporters = high PM₁₀ + high connectivity
2. Expect Po Valley stations to dominate (Lombardy)
3. Trentino should NOT be in top exporters (receptor role)
4. Regime classification provides mechanistic context

---

### **4.5 Synthesis: Integrating LISA, Regimes, and SDM**

**Maps to:** Implicit integration across all methodology sections

**Content:**
- Pull together all three analyses into unified interpretation
- Answer the research question: "Is there significant spatial autocorrelation between PM₁₀ peaks at border stations and internal Trentino valleys?"
- Quantify cross-border transport contribution

**Structure:**

#### 4.5.1 Convergent Evidence for Spatial Structure

```latex
Three independent analyses provide convergent evidence for spatial structure 
in PM₁₀ distributions:

1. **Global Moran's I (§4.2.1):** Strong positive autocorrelation (I = 0.601, 
   p < 0.001) confirms that pollution is not randomly distributed but exhibits 
   coherent spatial clustering.

2. **LISA (§4.2.2):** Identification of 6 High-High clusters and 15 Low-Low 
   clusters demonstrates that this clustering manifests as localized, contiguous 
   hotspots and clean zones rather than as diffuse network-wide correlation.

3. **SDM (§4.4.1):** Significant spatial autoregressive parameter (ρ = 0.076, 
   p < [value]) quantifies the magnitude of spillover: 7.6% of pollution variance 
   is attributable to neighbor effects after meteorological controls.

These findings collectively validate the spatial econometric framework and 
justify the use of neighbor-based models for pollution analysis in this region.
```

---

#### 4.5.2 Distinguishing Local Accumulation from Transport

```latex
The critical challenge in air quality studies is distinguishing between 
pollution that results from local emissions/meteorology and pollution that 
arrives via cross-border transport. Our multimethod approach addresses this 
through regime-LISA integration:

**High Pollution via Local Stagnation (Cluster 0 + HH):** Stations in Cluster 0 
(Stagnation regime) that also exhibit LISA High-High classification owe their 
elevated PM₁₀ to unfavorable local dispersion conditions (shallow BLH, high 
pressure, weak winds). The HH pattern reflects shared meteorological stagnation 
across neighboring stations rather than pollution transfer. Example: [Station X], 
with PM₁₀ = 29.6 μg/m³, BLH = 321 m, WS = 1.58 m/s.

**High Pollution via Transport (Cluster 3 + HH):** Stations in Cluster 3 
(Transport Corridor) that exhibit LISA High-High classification show elevated 
PM₁₀ **despite** favorable dispersion meteorology (deep BLH = 389 m, high 
WS = 1.68 m/s). This spatial-meteorological mismatch is the signature of 
transported pollution: local conditions predict low PM₁₀, yet observations 
remain high due to advection from upwind sources. Example: [Station Y], 
with observed PM₁₀ = 30.8 μg/m³ vs. local meteorological prediction = [lower value].

This distinction is impossible with LISA alone (which identifies where pollution 
clusters) or K-means alone (which identifies meteorological regimes). Only their 
integration reveals **why** pollution is high: local accumulation vs. transport.
```

---

#### 4.5.3 Quantitative Answer to Research Question

```latex
The central research question posed in §1 asks: "Is there significant spatial 
autocorrelation between PM₁₀ peaks recorded at border stations and those in 
internal Trentino valleys during specific synoptic meteorological events?"

**Answer: Yes, but the mechanism is nuanced.**

1. **Spatial Autocorrelation Confirmed:** Global Moran's I = 0.601 (p < 0.001) 
   establishes strong positive spatial autocorrelation between border and 
   internal stations.

2. **Quantified Spillover:** The SDM decomposes this autocorrelation into:
   - **92.4%** explained by shared meteorological conditions (local + neighbor 
     atmospheric fields)
   - **7.6%** attributable to residual spatial spillover (ρ = 0.076)

3. **Cross-Border Transport Contribution:** For Trentino target stations, 
   spillover ranges from 6.3% to 14.5% of observed PM₁₀, with an average of 
   [X] μg/m³. This represents the **additional pollution burden** from cross-
   border sources beyond what local meteorology would predict.

4. **Mechanism Identification:** The integration of LISA and regime classification 
   reveals that cross-border transport is most evident in Cluster 3 (Transport 
   Corridor) stations, where high pollution co-occurs with favorable dispersion 
   meteorology—a pattern inconsistent with local sources alone.

**Conclusion:** Spatial autocorrelation between border and internal stations 
exists and is statistically significant, but its primary driver is the spatial 
structure of meteorological fields rather than direct pollution transfer. 
Cross-border transport contributes a measurable but secondary effect (≈7-15% 
of total PM₁₀), concentrated along specific atmospheric corridors identified 
by regime analysis.
```

---

#### 4.5.4 Policy Implications

```latex
These findings have direct implications for air quality management in the 
Trentino-Alto Adige region:

1. **Local Mitigation Dominates:** With 85-93% of PM₁₀ explained by local 
   meteorology and emissions, regional air quality policy should prioritize 
   local emission controls (traffic, heating, industrial) over cross-border 
   agreements.

2. **Transport Corridors Require Targeted Action:** The 7-15% spillover 
   contribution, while smaller than local effects, is concentrated in specific 
   valleys (Cluster 3 / Transport Corridor). These areas may benefit from:
   - Early warning systems tied to synoptic forecasts (high-pressure stagnation 
     events)
   - Traffic restrictions during adverse meteorological conditions
   - Cross-regional coordination with Lombardy during transport episodes

3. **Meteorological Dependency:** The strong influence of BLH, pressure, and 
   wind speed (Table 6) suggests that pollution episodes are highly weather-
   dependent. Real-time forecasting of atmospheric stability could enable 
   proactive interventions.

4. **Network Effects Matter:** The spatial multiplier (Equation 6) demonstrates 
   that pollution propagates through the network: reducing emissions at high-
   centrality exporters (e.g., Pavia, Mantova) would benefit multiple downwind 
   receptors simultaneously, justifying coordinated regional strategies.
```

---

## 📋 RESULTS SECTION CHECKLIST

**Ensure every methodology section has corresponding results:**

- [x] §3.1 Spatial Weights → §4.1 W matrix statistics
- [x] §3.2 Global Moran's I → §4.2.1 I statistics table
- [x] §3.2 LISA → §4.2.2 LISA classification + map
- [x] §3.3 K-means → §4.3.2 Cluster assignments
- [x] §3.3 PCA → §4.3.3 Variance explained + scatter plot
- [x] §3.3 Regime profiles → §4.3.4 Cluster characterization table
- [x] §3.4 SDM → §4.4.1 Parameter estimates
- [x] §3.4 Decomposition → §4.4.2 Local vs. spillover table
- [x] §3.4 Interpretation → §4.5 Synthesis

**Cross-referencing:**
- Every results subsection should cite its corresponding methodology section
- Example: "As specified in §3.2, Global Moran's I was computed..."
- Example: "Following the decomposition framework in §3.4.3, we obtain..."

---

## 🎨 VISUALIZATION REQUIREMENTS

**Mandatory Figures:**
1. ✅ **Figure 1:** PM₁₀ LISA cluster map (HH, LL, HL, LH)
2. ✅ **Figure 2a:** WCSS elbow plot (k selection)
3. ✅ **Figure 2b:** Spatial map with regime assignments
4. ✅ **Figure 3:** PCA scatter plot (PC1 vs PC2, color by cluster)
5. ✅ **Figure 4a:** Local vs. Spillover decomposition (bar chart for targets)
6. ✅ **Figure 4b:** Spillover % scatter plot (all stations)

**Optional but Recommended:**
- Network graph showing KNN connections
- Time series of PM₁₀ for target stations (if temporal analysis included)
- Moran scatterplot (standardized PM₁₀ vs. spatial lag)
- Heatmap of regime environmental profiles (Table 4 visualized)

---

## ⏱️ WRITING TIMELINE ESTIMATE

**Section-by-section effort:**
- §4.1 Spatial Weights: 30 min (simple descriptive stats)
- §4.2 LISA/Moran: 2-3 hours (table + map + interpretation)
- §4.3 Regimes: 3-4 hours (PCA + profiles + integration)
- §4.4 SDM: 4-5 hours (parameter table + decomposition + source attribution)
- §4.5 Synthesis: 2-3 hours (integration + research question answer)

**Total estimated time:** 12-16 hours for complete Results + Discussion section

**Parallelization strategy:**
1. Draft all tables first (can be done quickly with output data)
2. Generate all figures (can be automated)
3. Write interpretation paragraphs (requires most thought)
4. Add cross-references and polish

---

## ✅ FINAL PRE-WRITING CHECKLIST

**Before starting Results section:**
- [ ] Verify all output data files are accessible
- [ ] Confirm figure generation scripts work
- [ ] Review methodology section for exact wording (to ensure consistency)
- [ ] Prepare LaTeX table templates
- [ ] Fix critical methodology issues (station count, RAD justification)

**Quality control during writing:**
- [ ] Every result cites its methodology section
- [ ] Every table has a number, caption, and in-text reference
- [ ] Every figure has a number, caption, and in-text reference
- [ ] No orphan results (results without methods) or orphan methods (methods without results)
- [ ] Interpretation stays grounded in data (no speculation beyond what models show)

---

**Document Status:** Ready for Results section drafting  
**Next Action:** Begin with §4.1 (simple) to build momentum, then tackle §4.4 (complex)