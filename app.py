import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import scipy.spatial.distance as sp_dist
import matplotlib.pyplot as plt
import json

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="CMA-SIH: Outbreak GIS & Trajectory Prediction",
    page_icon="🦠",
    layout="wide"
)

# ---------------------------------------------------------
# SIDEBAR CONTROLS & MAPPLS KEY
# ---------------------------------------------------------
st.sidebar.header("⚙️ Control Panel")

# Target Disease Selection
disease_options = ["Dengue", "Malaria", "Japanese Encephalitis", "Chikungunya"]
selected_disease = st.sidebar.selectbox("Select Target Disease:", disease_options)

st.sidebar.markdown("---")
st.sidebar.subheader("🗺️ Mappls (MapmyIndia) Configuration")
mappls_api_key = st.sidebar.text_input("Enter Mappls API Key:", value="", type="password")

st.sidebar.markdown("---")

st.title("🦠 CMA-SIH: Causal Multimodal AI Framework")
st.subheader(f"Disease Forecasting & Mappls Spatial Trend Analytics ({selected_disease}) — West Bengal")
st.markdown("---")

# ---------------------------------------------------------
# DATA & GRAPH SETUP
# ---------------------------------------------------------
@st.cache_data
def load_data_and_graph(disease_name):
    torch.manual_seed(42)
    np.random.seed(42)

    districts_wb = [
        {"name": "Kolkata", "lat": 22.5726, "lng": 88.3639},
        {"name": "Howrah", "lat": 22.5958, "lng": 88.2636},
        {"name": "North 24 Parganas", "lat": 22.7220, "lng": 88.4800},
        {"name": "South 24 Parganas", "lat": 22.1400, "lng": 88.4200},
        {"name": "Hooghly", "lat": 22.9000, "lng": 88.3900},
        {"name": "Nadia", "lat": 23.4710, "lng": 88.5565},
        {"name": "Murshidabad", "lat": 24.1750, "lng": 88.2800},
        {"name": "Birbhum", "lat": 23.8400, "lng": 87.6100},
        {"name": "Bankura", "lat": 23.2324, "lng": 87.0784},
        {"name": "Purulia", "lat": 23.3320, "lng": 86.3650},
        {"name": "Paschim Medinipur", "lat": 22.4257, "lng": 87.3199},
        {"name": "Purba Medinipur", "lat": 21.9300, "lng": 87.7800},
        {"name": "Jhargram", "lat": 22.4500, "lng": 86.9800},
        {"name": "Paschim Bardhaman", "lat": 23.6833, "lng": 86.9833},
        {"name": "Purba Bardhaman", "lat": 23.2333, "lng": 87.8667},
        {"name": "Malda", "lat": 25.0000, "lng": 88.1400},
        {"name": "Uttar Dinajpur", "lat": 25.6200, "lng": 88.1200},
        {"name": "Dakshin Dinajpur", "lat": 25.2200, "lng": 88.7700},
        {"name": "Jalpaiguri", "lat": 26.5200, "lng": 88.7300},
        {"name": "Darjeeling", "lat": 27.0410, "lng": 88.2663},
        {"name": "Kalimpong", "lat": 27.0600, "lng": 88.4700},
        {"name": "Cooch Behar", "lat": 26.3200, "lng": 89.4500},
        {"name": "Alipurduar", "lat": 26.4900, "lng": 89.5200}
    ]

    disease_scale = {
        "Dengue": 1.2,
        "Malaria": 0.8,
        "Japanese Encephalitis": 0.4,
        "Chikungunya": 0.6
    }.get(disease_name, 1.0)

    N_DISTRICTS = len(districts_wb)
    N_WEEKS = 156
    LAT_LON = np.array([[d["lat"], d["lng"]] for d in districts_wb])

    weeks = np.arange(N_WEEKS)
    seasonal_temp = 26 + 6 * np.sin(2 * np.pi * weeks / 52)
    seasonal_preci = np.maximum(0, 180 * np.sin(2 * np.pi * (weeks - 12) / 52) + np.random.normal(0, 10, N_WEEKS))
    seasonal_lai = 1.4 + 0.7 * np.sin(2 * np.pi * (weeks - 8) / 52)

    data_list = []
    for i, dist in enumerate(districts_wb):
        dist_mult = 1.25 if dist["name"] in ["Kolkata", "North 24 Parganas", "Howrah", "Murshidabad"] else 0.85
        dist_temp = seasonal_temp + np.random.normal(0, 0.4, N_WEEKS)
        dist_preci = np.maximum(0, seasonal_preci + np.random.normal(0, 5, N_WEEKS))
        dist_lai = np.maximum(0.2, seasonal_lai + np.random.normal(0, 0.03, N_WEEKS))
        biobert_risk = np.clip(0.12 * dist_preci / 20.0 + np.random.normal(0.25, 0.015, N_WEEKS), 0, 1)
        
        lagged_preci = np.roll(dist_preci, 3)
        lagged_preci[:3] = dist_preci[:3]
        
        base_cases = disease_scale * dist_mult * (15.0 + 2.1 * (dist_temp - 20) + 0.55 * lagged_preci + 30.0 * dist_lai + 75.0 * biobert_risk)
        cases = np.maximum(5, base_cases + np.random.normal(0, 1.2, N_WEEKS))
        
        for w in range(N_WEEKS):
            data_list.append({
                "district": dist["name"],
                "district_idx": i,
                "week": w,
                "temp": dist_temp[w],
                "preci": dist_preci[w],
                "lai": dist_lai[w],
                "biobert_signal": biobert_risk[w],
                "cases": cases[w]
            })

    df = pd.DataFrame(data_list)

    dist_matrix = sp_dist.squareform(sp_dist.pdist(LAT_LON, metric='euclidean'))
    spatial_adj =
