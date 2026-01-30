# Project Purpose: Pollution Corridors Analysis (Po Valley - Alpine Region)

**Last Updated:** 30 January 2026

---

## 🎯 Core Research Question

**"To what extent does the spatial connectivity of the monitoring network (modeled as a graph) explain the diffusion of PM₁₀ clusters during Po Valley transport events?"**

### Sub-questions:
- Is there significant spatial autocorrelation between border and internal monitoring stations?
- How do synoptic meteorological events (wind patterns, boundary layer height) influence the formation of pollution corridors?
- Can network analysis identify the primary pathways of pollutant transport from the Po Valley into the Alpine valleys?

---

## 📋 Project Objectives

1. **Detect spatial patterns** of PM₁₀ concentrations using Global and Local Moran's I to identify "High-High" pollution clusters
2. **Model spatial spillover effects** using Spatial Autoregressive Models (SAR/SDM) to quantify how pollution from the Po Valley affects Alpine regions
3. **Map pollution corridors** using network analysis where monitoring stations are nodes and wind-flow/distance define edges
4. **Visualize synoptic conditions** through interactive web maps that show how spatial clusters shift under different meteorological scenarios

---

## 🔧 Technical Approach (Aligned with Course Requirements)

### Data Wrangling
- **Tools:** GeoPandas (vector data), GDAL (ERA5 raster integration)
- **Data Sources:** 
  - PM₁₀ monitoring stations
  - ERA5 meteorological data (wind vectors, boundary layer height)

### Exploratory Spatial Analysis
- **Global Moran's I:** Assess overall spatial autocorrelation of PM₁₀
- **Local Moran's I (LISA):** Identify local clusters and outliers
- **Hot Spot Analysis:** Detect statistically significant pollution hotspots

### Spatial Modeling
- **Spatial Autoregressive Model (SAR)** or **Spatial Durbin Model (SDM)**
- **Purpose:** Quantify spatial spillover of PM₁₀ from Po Valley to Alpine regions
- **Key Innovation:** Use network-based adjacency weighted by ERA5 wind vectors to define the Spatial Weight Matrix (W)

### Network Analysis
- **Nodes:** Monitoring stations
- **Edges:** Weighted by:
  - Geographic distance
  - Wind flow direction and magnitude (ERA5 data)
  - Correlation of PM₁₀ time series between stations
- **Objective:** Identify primary pollution corridors and assess network connectivity

---

## 🗺️ Interactive Map Requirements

**Mandatory Feature:** Time/scenario selector for synoptic meteorological events

### Functionality:
- Toggle between different wind conditions (ERA5 data)
- Visualize how PM₁₀ spatial clusters shift across Alpine valleys
- Display network edges (corridors) with varying opacity/color based on transport intensity
- Layer controls for:
  - Monitoring stations (with PM₁₀ levels)
  - Wind vectors
  - Spatial clusters (LISA results)
  - Pollution corridors (network edges)

**Tools:** Folium, Leaflet, or similar libraries

---

## 📊 Required Course Techniques

| Component | Technique | Implementation |
|-----------|-----------|----------------|
| **Data Wrangling** | GeoPandas, GDAL | Vector/raster integration |
| **Exploratory Analysis** | Global & Local Moran's I | Detect High-High clusters |
| **Spatial Modeling** | SAR/SDM | Quantify spatial spillover |
| **Network Analysis** | Graph-based corridor mapping | Identify transport pathways |
| **Visualization** | Interactive web map | Time-based scenario comparison |

---

## ✅ Reproducibility Checklist

- [ ] GitHub repository with complete code
- [ ] `requirements.txt` with pinned library versions (GeoPandas, PySAL, etc.)
- [ ] Clear documentation of all data sources and preprocessing steps
- [ ] If QGIS is used, document the basic sequence of commands
- [ ] Include sample data or clear instructions for data acquisition
- [ ] Jupyter notebooks with markdown explanations for each analysis step
- [ ] README with step-by-step instructions to reproduce results

---

## 🔑 Key Methodological Innovation

**Wind-Weighted Spatial Weight Matrix:**
Instead of using simple contiguity or distance-based weights, define the spatial weight matrix (W) using ERA5 wind vectors. This creates a directed, weighted network where:
- Edge weights reflect the likelihood of pollutant transport based on prevailing wind patterns
- The adjacency structure captures the physical mechanisms of pollution diffusion
- The model explicitly accounts for meteorological drivers of spatial correlation

---

## 📂 Expected Deliverables

1. **Analytical Report** with research question, methods, results, and interpretation
2. **Interactive Web Map** with time/scenario controls
3. **GitHub Repository** with all code, data, and documentation
4. **Spatial Statistics Results:**
   - Moran's I statistics (global and local)
   - SAR/SDM model coefficients and diagnostics
   - Network metrics (centrality, clustering coefficient, etc.)
5. **Visualizations:**
   - Static maps of pollution clusters
   - Network diagram of pollution corridors
   - Time series plots of PM₁₀ at key stations
   - Wind rose diagrams for different synoptic conditions

---

## 🎓 Course Alignment Notes

This project demonstrates:
- **Specificity of geospatial data model:** Network-based representation of pollution transport
- **Spatial statistics:** Autocorrelation analysis and spatial regression
- **Network analysis:** Graph-based corridor identification
- **Practical application:** Environmental policy relevance (transboundary pollution)
- **Technical rigor:** Integration of multiple data sources and advanced spatial methods

---

## 💡 When in Doubt, Remember:

**This project is about understanding HOW pollution moves from the Po Valley into the Alps through specific geographic corridors, using the spatial structure of the monitoring network and meteorological conditions to explain these patterns.**
