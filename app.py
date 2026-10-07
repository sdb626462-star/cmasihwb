import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import scipy.spatial.distance as sp_dist
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="CMA-SIH: Disease Forecasting - West Bengal",
    page_icon="🦠",
    layout="wide"
)

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.header("⚙️ Control Panel")

# Disease Selector
disease_options = ["Dengue", "Malaria", "Japanese Encephalitis", "Chikungunya"]
selected_disease = st.sidebar.selectbox("Select Target Disease:", disease_options)

st.sidebar.markdown("---")

st.title("🦠 CMA-SIH: Causal Multimodal AI Framework")
st.subheader(f"Vector-Borne Disease Forecasting ({selected_disease}) — West Bengal")
st.markdown("---")

# ---------------------------------------------------------
# DATA & MODEL SETUP (CACHED FOR PERFORMANCE)
# ---------------------------------------------------------
@st.cache_data
def load_data_and_graph(disease_name):
    torch.manual_seed(42)
    np.random.seed(42)

    districts_wb = [
        {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639},
        {"name": "Howrah", "lat": 22.5958, "lon": 88.2636},
        {"name": "North 24 Parganas", "lat": 22.7220, "lon": 88.4800},
        {"name": "South 24 Parganas", "lat": 22.1400, "lon": 88.4200},
        {"name": "Hooghly", "lat": 22.9000, "lon": 88.3900},
        {"name": "Nadia", "lat": 23.4710, "lon": 88.5565},
        {"name": "Murshidabad", "lat": 24.1750, "lon": 88.2800},
        {"name": "Birbhum", "lat": 23.8400, "lon": 87.6100},
        {"name": "Bankura", "lat": 23.2324, "lon": 87.0784},
        {"name": "Purulia", "lat": 23.3320, "lon": 86.3650},
        {"name": "Paschim Medinipur", "lat": 22.4257, "lon": 87.3199},
        {"name": "Purba Medinipur", "lat": 21.9300, "lon": 87.7800},
        {"name": "Jhargram", "lat": 22.4500, "lon": 86.9800},
        {"name": "Paschim Bardhaman", "lat": 23.6833, "lon": 86.9833},
        {"name": "Purba Bardhaman", "lat": 23.2333, "lon": 87.8667},
        {"name": "Malda", "lat": 25.0000, "lon": 88.1400},
        {"name": "Uttar Dinajpur", "lat": 25.6200, "lon": 88.1200},
        {"name": "Dakshin Dinajpur", "lat": 25.2200, "lon": 88.7700},
        {"name": "Jalpaiguri", "lat": 26.5200, "lon": 88.7300},
        {"name": "Darjeeling", "lat": 27.0410, "lon": 88.2663},
        {"name": "Kalimpong", "lat": 27.0600, "lon": 88.4700},
        {"name": "Cooch Behar", "lat": 26.3200, "lon": 89.4500},
        {"name": "Alipurduar", "lat": 26.4900, "lon": 89.5200}
    ]

    # Scaling factor for different disease baselines
    disease_scale = {
        "Dengue": 1.2,
        "Malaria": 0.8,
        "Japanese Encephalitis": 0.4,
        "Chikungunya": 0.6
    }.get(disease_name, 1.0)

    N_DISTRICTS = len(districts_wb)
    N_WEEKS = 156
    LAT_LON = np.array([[d["lat"], d["lon"]] for d in districts_wb])

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

    # Spatial Adjacency & Correlation Matrix (DCMG)
    dist_matrix = sp_dist.squareform(sp_dist.pdist(LAT_LON, metric='euclidean'))
    spatial_adj = np.exp(-dist_matrix / (0.5 * np.std(dist_matrix)))
    case_matrix = df.pivot(index='week', columns='district_idx', values='cases').values
    corr_matrix = np.maximum(0, np.nan_to_num(np.corrcoef(case_matrix.T), 0))

    A_dcmg = 0.6 * spatial_adj + 0.4 * corr_matrix
    deg = np.diag(np.sum(A_dcmg, axis=1)**(-0.5))
    A_norm = np.dot(np.dot(deg, A_dcmg), deg)

    return districts_wb, df, A_norm

# Model Definition
class STGNNCell(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(STGNNCell, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.gate = nn.Linear(in_dim, out_dim)
        
    def forward(self, x, adj):
        ax = torch.einsum('ij,bjk->bik', adj, x)
        h = self.fc(ax)
        g = torch.sigmoid(self.gate(ax))
        return torch.relu(h) * g

class PhysicsSTGNN(nn.Module):
