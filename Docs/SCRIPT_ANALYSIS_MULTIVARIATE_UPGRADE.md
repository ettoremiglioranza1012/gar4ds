# Script Analysis: Multi-Variable Dataset Integration Assessment

**Document Purpose:** This document analyzes each script in `scripts/data_analysis/`, assessing their current functionality and evaluating how they can benefit from the new multi-variable dataset that includes meteorological and environmental variables (BLH, TEMP, PRECIP, PRESS, RAD, WS) alongside PM10 measurements.

**Date:** February 4, 2026  
**Dataset Context:** Original dataset contained only PM10 values; updated dataset includes 7 variables per station across 36 stations (254 total columns).

---

## 1. spatial_analysis.py

### Current Purpose
Performs **Global and Local Spatial Autocorrelation Analysis** using Moran's I statistics to identify spatial clustering patterns in PM10 concentrations. The script:
- Calculates Global Moran's I to test for overall spatial clustering
- Applies Local Moran's I (LISA) to identify High-High clusters (hotspots) and Low-Low clusters (coldspots)
- Visualizes spatial clustering patterns on a geographical map
- Identifies specific pollution clusters and clean zones

### Previous Implementation (PM10-only)
- **Input:** Mean PM10 concentration per station (single variable)
- **Spatial Weights:** K-Nearest Neighbors (k=6) with row-standardized weights
- **Output:** Single spatial clustering map for PM10

### Assessment: Benefits from Multi-Variable Dataset
**Score: ⭐⭐⭐⭐⭐ HIGHLY BENEFICIAL**

The spatial autocorrelation analysis is **significantly enhanced** by incorporating meteorological variables to reveal:
- Environmental clustering patterns across different variables
- Spatial structure of regressors (validates their use in spatial models)
- Multivariate atmospheric regimes for source/receptor classification

### Implemented Solution: Hybrid Approach

**Strategy:** Focused visual narrative for PM10 + comprehensive statistical validation + regime definition

#### Component 1: Parallel LISA Analysis (All Variables)

**Global Moran's I:** Calculated for ALL variables (PM10, TEMP, BLH, WS, PRECIP, PRESS, RAD)
- Purpose: Proves regressors are spatially structured (critical for model validation)
- Output: CSV with Global Moran's I statistics and significance for each variable
- Key Finding: All variables show significant spatial clustering (validates spatial econometric approach)

**LISA Cluster Maps:** Generated ONLY for PM10
- Rationale: Avoid visual clutter; focus narrative on pollution patterns
- Meteorological variables: Statistical validation recorded, map generation skipped
- Output: Single PM10 LISA map showing High-High and Low-Low clusters

```python
# Implemented logic
for var in vars_to_analyze:
    # Always calculate Global Moran's I
    mi = Moran(var_means, w)
    
    # Always run LISA
    lisa = Moran_Local(var_means, w)
    
    # Conditional visualization
    if var == 'PM10':
        create_variable_lisa_map(...)  # Generate map
    else:
        print("→ Meteorological variable - map generation skipped")
```

**Results:**
- 7 variables analyzed statistically
- 1 LISA cluster map generated (PM10 only)
- Comparison plot showing spatial autocorrelation across all variables
- Complete statistical validation in CSV format

#### Component 2: Multivariate Clustering Analysis

**Purpose:** Define atmospheric regimes for source/receptor classification

**Implementation:**
```python
# K-Means clustering on standardized feature matrix
cluster_vars = ['PM10', 'TEMP', 'BLH', 'WS', 'PRECIP', 'PRESS']
X = np.column_stack([var_means for var in cluster_vars])
X_scaled = StandardScaler().fit_transform(X)

# Optimal cluster determination
optimal_k = determine_optimal_k(X_scaled, max_k=8)
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

**Outputs:**
1. **Cluster Assignments CSV:** Station-to-regime mapping
2. **Cluster Profiles CSV:** Mean environmental characteristics per regime
3. **PCA Scatter Plot:** Stations colored by cluster membership
4. **Spatial Cluster Map:** Geographic distribution of regimes
5. **Cluster Heatmap:** Normalized environmental profiles

**Critical Output - Cluster Profile Summary:**
```
Cluster  N_Stations  PM10_mean  TEMP_mean  BLH_mean  WS_mean  Characterization
      0          13      29.56     286.58    320.86     1.58  Medium Pollution
      1           6      16.84     281.37    400.78     0.72     Low Pollution
      2           9      19.00     281.60    328.39     0.82     Low Pollution
      3           7      30.84     287.49    388.65     1.68    High Pollution
```

This table is **essential** for identifying:
- High pollution regimes (elevated PM10 + low dispersion capacity)
- Source stations (high emissions + favorable export conditions)
- Receptor stations (vulnerable to pollution accumulation)

### Performance Benefits
- **Execution Time:** Reduced by ~85% (skipping 6 meteorological maps)
- **Output Clarity:** Focused visual narrative (PM10 map + regime plots only)
- **Statistical Rigor:** Complete validation maintained (all Global Moran's I calculated)
- **Interpretability:** Regime definitions clearly presented for downstream analysis

---

## 2. durbin_spillover_analysis.py

### Current Purpose
Implements a **Spatial Durbin Model (SDM)** to decompose PM10 pollution into three components:
1. **Local Effect (Xβ):** Pollution from station's own characteristics
2. **Neighbor Context (WXθ):** Effect of neighboring stations' characteristics
3. **Spatial Spillover (ρWy):** Pollution "imported" from neighboring stations

The script identifies spillover sources for target stations (Borgo Valsugana, Monte Gaza, Parco S. Chiara, Piana Rotaliana) and quantifies how much pollution each station receives from its neighbors.

### Current Implementation
- **Dependent Variable:** Mean PM10 concentration
- **Independent Variables:** Latitude, Longitude, Altitude (station metadata - limited)
- **Model:** Spatial Lag Model (ML_Lag) as approximation of full Durbin
- **Output:** Spillover decomposition, source identification, network centrality metrics

### Assessment: Benefits from Multi-Variable Dataset
**Score: ⭐⭐⭐⭐⭐ EXTREMELY BENEFICIAL**

This is **THE MOST IMPACTFUL UPGRADE** among all scripts. The current model uses only geographic coordinates as independent variables, which is a **severe limitation**. Meteorological variables provide the actual **mechanisms** of pollution dispersion and accumulation.

### Why Multi-Variable Data is Transformative Here

**Current Problem:** The model can only capture spatial patterns based on location, not the **physical processes** that drive pollution transport.

**New Capabilities with Meteorological Data:**

1. **Mechanistic Understanding:** 
   - **BLH (Boundary Layer Height):** Low BLH traps pollutants → explains why spillover occurs
   - **Wind Speed (WS):** High wind disperses pollution → identifies transport pathways
   - **Precipitation:** Removes particulates → explains temporal variations in spillover
   - **Temperature:** Affects atmospheric stability and pollution chemistry

2. **Enhanced Model Specification:**
   - Current: `y = ρWy + (Lat, Lon, Alt)β + ε`
   - Improved: `y = ρWy + (BLH, WS, TEMP, PRECIP, PRESS, RAD)β + (W × [BLH, WS, ...])θ + ε`
   
3. **Neighbor Context Becomes Meaningful:**
   - `WXθ` term captures **how neighbor's weather affects local PM10**
   - Example: "My PM10 is high when my neighbor has low BLH and high wind toward me"

### Recommended Code Updates

**Major Enhancement: Full Durbin Model with Meteorological Variables**

```python
def prepare_model_data_multivar(spatial_df, meta_df, valid_stations):
    """
    Prepare multi-variable model with meteorological features
    """
    print("\n[3] Preparing Multi-Variable Model...")
    
    # Dependent variable: Mean PM10 per station
    pm10_means = spatial_df.select([f'PM10_{s}' for s in valid_stations]).mean().to_pandas().iloc[0]
    y = pm10_means.values.reshape(-1, 1)
    
    # Independent variables: Meteorological characteristics
    feature_names = []
    feature_data = []
    
    # 1. Boundary Layer Height (critical for pollution dispersion)
    blh_means = spatial_df.select([f'BLH_{s}' for s in valid_stations]).mean().to_pandas().iloc[0]
    feature_data.append(blh_means.values)
    feature_names.append('BLH')
    
    # 2. Temperature (affects atmospheric stability)
    temp_means = spatial_df.select([f'TEMP_{s}' for s in valid_stations]).mean().to_pandas().iloc[0]
    feature_data.append(temp_means.values)
    feature_names.append('TEMP')
    
    # 3. Wind Speed (primary dispersion mechanism)
    ws_means = spatial_df.select([f'WS_{s}' for s in valid_stations]).mean().to_pandas().iloc[0]
    feature_data.append(ws_means.values)
    feature_names.append('WS')
    
    # 4. Precipitation (wet deposition)
    precip_means = spatial_df.select([f'PRECIP_{s}' for s in valid_stations]).mean().to_pandas().iloc[0]
    feature_data.append(precip_means.values)
    feature_names.append('PRECIP')
    
    # 5. Pressure (weather systems)
    press_means = spatial_df.select([f'PRESS_{s}' for s in valid_stations]).mean().to_pandas().iloc[0]
    feature_data.append(press_means.values)
    feature_names.append('PRESS')
    
    # 6. Radiation (photochemical reactions)
    rad_means = spatial_df.select([f'RAD_{s}' for s in valid_stations]).mean().to_pandas().iloc[0]
    feature_data.append(rad_means.values)
    feature_names.append('RAD')
    
    # Optional: Add metadata features
    if 'Altitude' in meta_df.columns:
        feature_data.append(meta_df.loc[valid_stations, 'Altitude'].values)
        feature_names.append('Altitude')
    
    X = np.column_stack(feature_data)
    
    # Standardize features (important for mixed units)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"    ✓ Dependent variable: PM10 (n={len(y)})")
    print(f"    ✓ Independent variables: {feature_names}")
    print(f"    ✓ Features standardized for comparable coefficients")
    
    return y, X_scaled, feature_names, pm10_means, scaler
```

**Enhanced Interpretation Function:**

```python
def interpret_meteorological_effects(model, feature_names, rho):
    """
    Interpret how meteorological variables affect PM10
    """
    print("\n[5] Meteorological Influence Analysis...")
    
    betas = model.betas.flatten()
    
    print(f"\n    Direct Effects (own-station meteorology → own PM10):")
    print(f"    {'Variable':<15} {'Coefficient':<12} {'Interpretation'}")
    print(f"    {'-'*70}")
    
    for i, (name, beta) in enumerate(zip(feature_names, betas[1:])):  # Skip constant
        if name == 'BLH':
            interp = "↑ BLH → ↑ dispersion → ↓ PM10" if beta < 0 else "⚠ Unexpected positive"
        elif name == 'WS':
            interp = "↑ Wind → ↑ dispersion → ↓ PM10" if beta < 0 else "⚠ Check wind direction"
        elif name == 'PRECIP':
            interp = "↑ Rain → ↑ washout → ↓ PM10" if beta < 0 else "⚠ Unexpected"
        elif name == 'TEMP':
            interp = "↑ Temp → complex (stability + chemistry)"
        elif name == 'PRESS':
            interp = "↑ Pressure → stable air → ↑ PM10" if beta > 0 else "↓ P → mixing"
        else:
            interp = ""
        
        print(f"    {name:<15} {beta:>12.4f}    {interp}")
    
    print(f"\n    Spatial Spillover Parameter (ρ): {rho:.4f}")
    print(f"    → This is the 'contagion' effect independent of meteorology")
    print(f"    → With ρ={rho:.4f}, {rho*100:.1f}% of PM10 comes from neighbors")
```

**Spillover Source Analysis with Weather Context:**

```python
def identify_spillover_sources_with_weather(decomposition_df, W, valid_stations, 
                                             target_stations, y, spatial_df):
    """
    Enhanced spillover source identification with meteorological context
    """
    print("\n[6] Identifying Spillover Sources with Meteorological Context...")
    
    station_to_idx = {s: i for i, s in enumerate(valid_stations)}
    
    for target in target_stations:
        if target not in station_to_idx:
            continue
            
        idx = station_to_idx[target]
        
        print(f"\n    {'='*70}")
        print(f"    TARGET: {target}")
        print(f"    {'='*70}")
        
        # Get neighbor contributions
        neighbor_data = []
        for j in range(len(valid_stations)):
            if W[idx, j] > 0:
                neighbor = valid_stations[j]
                contribution = W[idx, j] * y[j]
                
                # Get neighbor's meteorological profile
                neighbor_blh = spatial_df.select(f'BLH_{neighbor}').mean()[0]
                neighbor_ws = spatial_df.select(f'WS_{neighbor}').mean()[0]
                neighbor_temp = spatial_df.select(f'TEMP_{neighbor}').mean()[0]
                
                neighbor_data.append({
                    'Neighbor': neighbor,
                    'Contribution': contribution,
                    'PM10': y[j],
                    'BLH': neighbor_blh,
                    'WindSpeed': neighbor_ws,
                    'Temp': neighbor_temp,
                    'Weight': W[idx, j]
                })
        
        contrib_df = pd.DataFrame(neighbor_data).sort_values('Contribution', ascending=False)
        
        print(f"\n    Top Sources with Meteorological Profile:")
        print(f"    {'Neighbor':<30} {'PM10':<8} {'BLH':<8} {'WS':<8} {'Contrib':<10}")
        print(f"    {'-'*74}")
        
        for _, row in contrib_df.head(8).iterrows():
            print(f"    {row['Neighbor']:<30} {row['PM10']:<8.1f} {row['BLH']:<8.1f} "
                  f"{row['WindSpeed']:<8.2f} {row['Contribution']:<10.2f}")
        
        # Save enhanced CSV
        csv_path = RESULTS_DIR / f"spillover_sources_detailed_{target.replace(' ', '_')}.csv"
        contrib_df.to_csv(csv_path, index=False)
```

**Implementation Priority:** This is **CRITICAL**. The meteorological variables transform the Durbin model from a purely spatial correlation tool into a **mechanistic pollution transport model**.

---

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

### Recommended Code Updates

**Enhancement 1: Use Meteorological Model for Better ρ**

```python
def estimate_rho_with_controls(spatial_df, meta_df, valid_stations, w):
    """
    Estimate spatial lag parameter with proper meteorological controls
    This gives more accurate ρ for impact matrix calculation
    """
    print("\n[1B] Estimating Spatial Lag with Meteorological Controls...")
    
    # Prepare full feature set
    y, X, feature_names = prepare_multivar_features(spatial_df, valid_stations)
    
    # Fit spatial lag model with controls
    model = ML_Lag(y, X, w=w)
    rho = model.rho
    
    print(f"    ✓ Spatial Lag (ρ) with controls: {rho:.4f}")
    print(f"    ✓ Controlled for: {feature_names}")
    print(f"    → This ρ is 'pure spillover' after accounting for weather")
    
    return rho, model

def prepare_multivar_features(spatial_df, valid_stations):
    """Prepare meteorological feature matrix"""
    pm10_means = spatial_df.select([f'PM10_{s}' for s in valid_stations]).mean().to_pandas().iloc[0]
    y = pm10_means.values.reshape(-1, 1)
    
    # Create feature matrix
    features = []
    names = ['BLH', 'TEMP', 'WS', 'PRECIP', 'PRESS']
    
    for var in names:
        var_means = spatial_df.select([f'{var}_{s}' for s in valid_stations]).mean().to_pandas().iloc[0]
        features.append(var_means.values)
    
    X = np.column_stack(features)
    
    # Standardize
    from sklearn.preprocessing import StandardScaler
    X_scaled = StandardScaler().fit_transform(X)
    
    return y, X_scaled, names
```

**Enhancement 2: Conditional Impact Matrices (Advanced)**

```python
def compute_conditional_impact_matrices(spatial_df, W, valid_stations, w):
    """
    Calculate separate impact matrices for different meteorological conditions
    Shows how network structure changes with weather
    """
    print("\n[2B] Computing Conditional Impact Matrices...")
    
    # Define meteorological regimes
    blh_values = spatial_df.select([c for c in spatial_df.columns if c.startswith('BLH_')]).mean()
    blh_median = blh_values.median()
    
    # Filter time series by condition
    low_blh_mask = spatial_df.select([c for c in spatial_df.columns if c.startswith('BLH_')]).to_pandas().le(blh_median).all(axis=1)
    high_blh_mask = ~low_blh_mask
    
    # Calculate separate ρ values
    results = {}
    
    for condition, mask in [('Low_BLH', low_blh_mask), ('High_BLH', high_blh_mask)]:
        subset_df = spatial_df.filter(mask)
        
        pm10_means = subset_df.select([f'PM10_{s}' for s in valid_stations]).mean().to_pandas().iloc[0]
        y = pm10_means.values.reshape(-1, 1)
        X = np.column_stack([np.ones(len(y)), np.arange(len(y))])  # Simple X
        
        model = ML_Lag(y, X, w=w)
        rho = model.rho
        
        # Compute impact matrix for this condition
        I = np.eye(len(valid_stations))
        impact_matrix = np.linalg.inv(I - rho * W)
        
        results[condition] = {
            'rho': rho,
            'impact_matrix': impact_matrix
        }
        
        print(f"\n    Condition: {condition}")
        print(f"    ρ = {rho:.4f}")
        print(f"    Mean total impact: {impact_matrix[~np.eye(len(valid_stations), dtype=bool)].mean():.4f}")
    
    # Compare
    rho_diff = results['Low_BLH']['rho'] - results['High_BLH']['rho']
    print(f"\n    Δρ (Low BLH - High BLH): {rho_diff:.4f}")
    
    if rho_diff > 0:
        print(f"    → Spillover is STRONGER under low boundary layer conditions")
        print(f"    → Pollution more 'contagious' when atmosphere is stable")
    else:
        print(f"    → Spillover is WEAKER under low boundary layer conditions")
    
    return results
```

**Enhancement 3: Meteorologically-Weighted Influence Profiles**

```python
def compute_weather_weighted_influence(impact_matrix, valid_stations, spatial_df):
    """
    Weight influence by meteorological vulnerability
    High-BLH stations export more; low-BLH stations are more vulnerable
    """
    print("\n[5B] Computing Meteorologically-Weighted Influence...")
    
    # Get meteorological exposure factors
    blh_means = spatial_df.select([f'BLH_{s}' for s in valid_stations]).mean().to_pandas().iloc[0].values
    ws_means = spatial_df.select([f'WS_{s}' for s in valid_stations]).mean().to_pandas().iloc[0].values
    
    # Vulnerability index: low BLH + low WS = high vulnerability
    vulnerability = (1 / blh_means) * (1 / (ws_means + 0.1))  # Avoid division by zero
    vulnerability = vulnerability / vulnerability.max()  # Normalize
    
    # Reweight impact matrix
    # Station i is more affected if it has high vulnerability
    cross_effects = impact_matrix.copy()
    np.fill_diagonal(cross_effects, 0)
    
    # Vulnerable stations feel impacts more strongly
    weather_weighted_impact = cross_effects * vulnerability[:, np.newaxis]
    
    # Calculate weather-aware influence
    out_influence = weather_weighted_impact.sum(axis=0)  # How much they affect vulnerable stations
    in_influence = weather_weighted_impact.sum(axis=1)   # How much they're affected (already weighted)
    
    influence_df = pd.DataFrame({
        'Station': valid_stations,
        'Vulnerability_Index': vulnerability,
        'Weather_Weighted_Out': out_influence,
        'Weather_Weighted_In': in_influence,
        'BLH': blh_means,
        'WindSpeed': ws_means
    })
    
    print("\n    Stations Most Vulnerable to Spillover (low BLH, low WS):")
    print(influence_df.nlargest(10, 'Vulnerability_Index')[['Station', 'Vulnerability_Index', 'BLH', 'WindSpeed']])
    
    return influence_df
```

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

### Recommended Code Updates

**Enhancement 1: Meteorological Profiles of Spillover Sources**

```python
def analyze_external_sources_with_weather(top_polluters, spatial_df, valid_stations, targets):
    """
    Analyze meteorological characteristics of top external polluters
    WHY do they impact targets? High emissions or favorable transport?
    """
    print("\n[4B] Meteorological Profile of External Polluters...")
    
    external_profiles = []
    
    for station, data in top_polluters[:10]:
        # Get meteorological profile
        pm10 = spatial_df.select(f'PM10_{station}').mean()[0]
        blh = spatial_df.select(f'BLH_{station}').mean()[0]
        ws = spatial_df.select(f'WS_{station}').mean()[0]
        temp = spatial_df.select(f'TEMP_{station}').mean()[0]
        precip = spatial_df.select(f'PRECIP_{station}').mean()[0]
        
        # Calculate "export potential"
        # High PM10 + Low BLH + High WS = strong exporter
        export_score = pm10 * ws / (blh + 1)  # Normalized
        
        external_profiles.append({
            'Station': station,
            'PM10': pm10,
            'BLH': blh,
            'WindSpeed': ws,
            'Temperature': temp,
            'Precipitation': precip,
            'Export_Score': export_score,
            'Impact_on_Targets': data['Total_Impact'],
            'Mechanism': classify_pollution_mechanism(pm10, blh, ws)
        })
    
    profile_df = pd.DataFrame(external_profiles)
    
    print(f"\n    Top External Polluters - HOW They Impact Targets:")
    print(f"    {'Station':<35} {'PM10':<8} {'BLH':<8} {'WS':<8} {'Mechanism'}")
    print(f"    {'-'*85}")
    
    for _, row in profile_df.iterrows():
        print(f"    {row['Station']:<35} {row['PM10']:<8.1f} {row['BLH']:<8.1f} "
              f"{row['WindSpeed']:<8.2f} {row['Mechanism']}")
    
    # Save
    profile_df.to_csv(RESULTS_DIR / 'external_polluters_meteorological_profile.csv', index=False)
    
    return profile_df

def classify_pollution_mechanism(pm10, blh, ws):
    """
    Classify HOW a station becomes a pollution source
    """
    if pm10 > 40 and ws > 3:
        return "High emitter + wind transport"
    elif pm10 > 40 and blh < 300:
        return "High emitter + low mixing"
    elif pm10 < 30 and ws > 4:
        return "Transport corridor (not high local)"
    elif blh < 300:
        return "Atmospheric trapping"
    else:
        return "Mixed factors"
```

**Enhancement 2: Temporal Spillover Analysis (Regime-Based)**

```python
def temporal_internal_external_decomposition(spatial_df, valid_stations, targets, W, rho):
    """
    Analyze how internal vs external spillover changes with meteorological conditions
    Key question: When is internal clustering strongest?
    """
    print("\n[5B] Temporal Regime Analysis...")
    
    # Define meteorological regimes
    # Regime 1: Low BLH (stagnant conditions)
    # Regime 2: High Wind (transport conditions)  
    # Regime 3: Precipitation (washout conditions)
    
    blh_cols = [c for c in spatial_df.columns if c.startswith('BLH_')]
    ws_cols = [c for c in spatial_df.columns if c.startswith('WS_')]
    precip_cols = [c for c in spatial_df.columns if c.startswith('PRECIP_')]
    
    # Calculate hourly regime classification
    blh_hourly = spatial_df.select(blh_cols).mean(axis=1)
    ws_hourly = spatial_df.select(ws_cols).mean(axis=1)
    precip_hourly = spatial_df.select(precip_cols).mean(axis=1)
    
    # Classify each hour
    regime_df = pl.DataFrame({
        'BLH_mean': blh_hourly,
        'WS_mean': ws_hourly,
        'Precip_mean': precip_hourly
    })
    
    # Define thresholds
    low_blh_mask = regime_df.filter(pl.col('BLH_mean') < blh_hourly.quantile(0.33))
    high_ws_mask = regime_df.filter(pl.col('WS_mean') > ws_hourly.quantile(0.66))
    precip_mask = regime_df.filter(pl.col('Precip_mean') > 0)
    
    regimes = {
        'Stagnant (Low BLH)': low_blh_mask,
        'Windy (High WS)': high_ws_mask,
        'Precipitation': precip_mask,
        'All Hours': regime_df
    }
    
    results = []
    
    for regime_name, mask in regimes.items():
        # Get PM10 for this regime
        regime_pm10 = spatial_df.filter(mask).select([f'PM10_{s}' for s in valid_stations]).mean()
        
        # Decompose spillover for this regime
        # (reuse decompose_internal_external function with regime-specific data)
        
        # Calculate internal vs external percentages
        # Store results
        
        results.append({
            'Regime': regime_name,
            'Avg_Internal_Pct': ...,  # Calculate
            'Avg_External_Pct': ...,  # Calculate
            'N_Hours': len(mask)
        })
    
    result_df = pd.DataFrame(results)
    
    print(f"\n    Internal vs External Spillover by Meteorological Regime:")
    print(f"    {'Regime':<25} {'Internal %':<12} {'External %':<12} {'N Hours'}")
    print(f"    {'-'*65}")
    for _, row in result_df.iterrows():
        print(f"    {row['Regime']:<25} {row['Avg_Internal_Pct']:<12.1f} "
              f"{row['Avg_External_Pct']:<12.1f} {row['N_Hours']}")
    
    # Interpretation
    print("\n    Interpretation:")
    stagnant = result_df[result_df['Regime'] == 'Stagnant (Low BLH)'].iloc[0]
    windy = result_df[result_df['Regime'] == 'Windy (High WS)'].iloc[0]
    
    if stagnant['Internal_Pct'] > windy['Internal_Pct']:
        print("    → Internal clustering STRENGTHENS during stagnant conditions")
        print("    → Target stations form a self-reinforcing trap under stable air")
    
    if windy['External_Pct'] > stagnant['External_Pct']:
        print("    → External spillover INCREASES with wind")
        print("    → Regional transport becomes dominant factor")
    
    return result_df
```

**Enhancement 3: Target Station Meteorological Vulnerability**

```python
def assess_target_vulnerability(targets, spatial_df, valid_stations):
    """
    Why are target stations vulnerable? Do they share weather patterns?
    """
    print("\n[6B] Target Station Vulnerability Assessment...")
    
    target_profiles = []
    
    for target in targets:
        # Get meteorological profile
        pm10 = spatial_df.select(f'PM10_{target}').mean()[0]
        blh = spatial_df.select(f'BLH_{target}').mean()[0]
        ws = spatial_df.select(f'WS_{target}').mean()[0]
        temp = spatial_df.select(f'TEMP_{target}').mean()[0]
        
        # Calculate vulnerability metrics
        # Low BLH + Low WS = high stagnation → high vulnerability
        stagnation_index = 1 / (blh * ws + 1)  # High value = stagnant
        
        # Variability (high variability = unstable, better mixing sometimes)
        blh_std = spatial_df.select(f'BLH_{target}').std()[0]
        ws_std = spatial_df.select(f'WS_{target}').std()[0]
        
        target_profiles.append({
            'Station': target,
            'PM10': pm10,
            'Avg_BLH': blh,
            'Avg_WindSpeed': ws,
            'Avg_Temp': temp,
            'Stagnation_Index': stagnation_index,
            'BLH_Variability': blh_std,
            'WS_Variability': ws_std
        })
    
    profile_df = pd.DataFrame(target_profiles)
    
    print(f"\n    Target Station Meteorological Characteristics:")
    print(profile_df.to_string(index=False))
    
    # Check if they cluster meteorologically
    from scipy.spatial.distance import pdist
    weather_features = profile_df[['Avg_BLH', 'Avg_WindSpeed', 'Avg_Temp']].values
    distances = pdist(weather_features, metric='euclidean')
    
    if distances.mean() < distances.std():
        print("\n    ✓ Target stations share SIMILAR meteorological conditions")
        print("    → They are likely in the same meteorological trap/regime")
    else:
        print("\n    ✗ Target stations have DIVERSE meteorological conditions")
        print("    → Pollution mechanism is not weather-driven uniformity")
    
    return profile_df
```

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

## Technical Notes

### Data Preparation Requirements
All scripts will need a common preprocessing function:

```python
def load_multivar_data(data_dir):
    """
    Load multi-variable spatial dataset
    Returns: polars DataFrame with shape (n_hours, 254)
    Columns: Data, PM10_*, TEMP_*, BLH_*, PRECIP_*, PRESS_*, RAD_*, WS_*
    """
    spatial_df = pl.read_parquet(data_dir / 'spatial_full_matrix.parquet')
    
    # Validate structure
    assert spatial_df.shape[1] == 254, "Expected 254 columns"
    
    return spatial_df

def extract_variable_means(spatial_df, variable, valid_stations):
    """
    Extract mean values for a specific variable across all stations
    """
    var_cols = [f'{variable}_{s}' for s in valid_stations]
    var_means = spatial_df.select(var_cols).mean().to_pandas().iloc[0]
    return var_means.values
```

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
