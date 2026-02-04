## 3. impact_matrix_analysis.py

### Current Purpose
Computes the **Impact Matrix (I - ρW)^(-1)** which captures the total network-wide effects of pollution shocks, including:
- **Direct effects:** Station A → Station B
- **Indirect effects:** Station A → C → B (and all longer paths)
- **Network centrality:** Most influential and most vulnerable stations
- **Feedback loops:** How pollution circulates through the network

The matrix represents the spatial multiplier that transforms local shocks into system-wide impacts.

### Current Implementation
- **Input:** PM10 means, spatial lag parameter ρ from a basic spatial lag model
- **Model:** Uses geographic coordinates (Lat/Lon) to estimate ρ
- **Analysis:** Pure network topology analysis of pollution propagation
- **Output:** Full impact matrix, target sub-matrix, centrality rankings, influence profiles

### Assessment: Benefits from Multi-Variable Dataset
**Score: ⭐⭐⭐ MODERATELY BENEFICIAL**

The impact matrix analysis is primarily a **network topology** exercise. However, multi-variable data provides value in two ways:

1. **Better ρ Estimation:** More accurate spatial lag parameter when model includes proper covariates
2. **Conditional Impact Matrices:** Calculate different impact structures under different meteorological regimes

**Key Insight:** The impact matrix itself is a **geometric object** (depends on W and ρ), so adding variables doesn't change its fundamental calculation. However, having a **better-estimated ρ** improves accuracy.

**Implementation Strategy:** Start with Enhancement 1 (better ρ estimation) as it's straightforward and improves all downstream analyses. Enhancement 2 is more advanced but reveals important regime-dependent behavior.

---

## 4. internal_external_spillover.py

### Current Purpose
Decomposes spatial spillover for target stations into two components:
1. **Internal Spillover:** Pollution from OTHER target stations (self-reinforcing cluster effect)
2. **External Spillover:** Pollution from stations OUTSIDE the target network

**Key Question Answered:** Are target stations a self-reinforcing pollution cluster, or are they victims of external pollution pressure?

### Current Implementation
- **Input:** PM10 means, spatial weights, spatial lag parameter
- **Analysis:** Partitions spillover by source type (internal vs external neighbors)
- **Output:** 
  - Spillover percentage breakdown per target station
  - Top external polluters affecting target network
  - Network-wide verdict (cluster vs victims)

### Assessment: Benefits from Multi-Variable Dataset
**Score: ⭐⭐⭐⭐ HIGHLY BENEFICIAL**

This analysis significantly benefits from meteorological data for two reasons:

1. **Identifying Why External Sources Matter:** 
   - Are external polluters high-emitters (high PM10) or well-positioned (favorable winds, low BLH)?
   - Can we predict when external spillover is strongest?

2. **Understanding Internal Cluster Dynamics:**
   - Do target stations share similar meteorological profiles (same weather trap)?
   - Is internal spillover due to proximity or shared atmospheric conditions?

3. **Policy Relevance:**
   - If external spillover dominates due to wind patterns → need regional coordination
   - If internal cluster due to local inversions → need local emission controls

**Implementation Priority:** Enhancement 1 is essential for understanding mechanism, Enhancement 2 provides temporal dynamics insight, Enhancement 3 explains target vulnerability.

---

## Summary Table: Multi-Variable Integration Priority

| Script | Current Focus | Benefit Level | Priority | Key Enhancement |
|--------|--------------|---------------|----------|-----------------|
| **durbin_spillover_analysis.py** | Spatial Durbin Model decomposition | ⭐⭐⭐⭐⭐ | **CRITICAL** | Add meteorological variables as model features - transforms from correlation to causation |
| **spatial_analysis.py** | Moran's I spatial clustering | ⭐⭐⭐⭐⭐ | **HIGH** | Run parallel LISA for each variable + multivariate clustering analysis |
| **internal_external_spillover.py** | Internal vs external spillover | ⭐⭐⭐⭐ | **HIGH** | Profile external sources by weather + temporal regime analysis |
| **impact_matrix_analysis.py** | Network impact propagation | ⭐⭐⭐ | **MEDIUM** | Use better ρ from weather-controlled model + conditional impact matrices |

---

## Implementation Roadmap

### Phase 1: Quick Wins (Week 1)
1. **durbin_spillover_analysis.py:** Add meteorological variables to X matrix
2. **spatial_analysis.py:** Run separate LISA for PM10, BLH, TEMP, WS

### Phase 2: Enhanced Analysis (Week 2-3)
3. **internal_external_spillover.py:** Add meteorological profiles to external sources
4. **impact_matrix_analysis.py:** Estimate conditional ρ under different weather regimes

### Phase 3: Advanced Integration (Week 4+)
5. **All scripts:** Implement temporal regime-based analysis
6. **New script:** Create `multivariate_spatial_model.py` for integrated analysis
7. **Validation:** Compare results to original PM10-only analysis

---

### Standardization is Critical
When mixing meteorological variables with different units:
- PM10: μg/m³
- BLH: meters
- TEMP: Kelvin or Celsius
- WS: m/s
- PRECIP: mm or kg/m²
- PRESS: Pascals
- RAD: W/m²

**Always standardize before regression:**
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Model Interpretation After Standardization
When features are standardized, coefficients represent **effect of 1 standard deviation change**:
- β_BLH = -0.5 means: "1 SD increase in BLH → 0.5 μg/m³ decrease in PM10"
- This makes coefficients **directly comparable** across variables

---

## Expected Research Insights from Multi-Variable Analysis

1. **Pollution Transport Mechanisms:**
   - Quantify: "X% of spillover occurs during low BLH + high wind events"
   - Identify: "Station Y is a source because of favorable wind patterns, not high emissions"

2. **Atmospheric Trapping:**
   - Detect: "Target stations share low BLH → form a meteorological trap"
   - Temporal: "Internal clustering peaks during winter inversions"

3. **Policy Targets:**
   - Prioritize: "External station Z has high export potential due to location + weather → priority for intervention"
   - Conditions: "Regional coordination needed most during [specific weather regime]"

4. **Prediction:**
   - Forecast: "Next week's low BLH forecast → expect spillover to increase by X%"
   - Early warning: "Current wind patterns favor pollution transport to targets"

---

## Conclusion

All four scripts will benefit from the multi-variable dataset, with **durbin_spillover_analysis.py** showing the most dramatic improvement. The current PM10-only analysis captures spatial patterns but cannot explain **why** they occur. Adding meteorological variables transforms these scripts from **descriptive tools** into **mechanistic models** that reveal the physical processes driving pollution dispersion and spillover.

The recommended implementation strategy is to:
1. Start with durbin_spillover_analysis.py (highest impact)
2. Extend spatial_analysis.py for multivariate patterns
3. Enhance internal_external_spillover.py with source profiling
4. Refine impact_matrix_analysis.py with better parameter estimates

This progressive approach allows validation at each step and builds toward a comprehensive understanding of how meteorological conditions modulate spatial pollution dynamics.
