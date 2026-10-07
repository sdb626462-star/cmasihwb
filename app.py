import os
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import scipy.spatial.distance as sp_dist
import matplotlib.pyplot as plt
import pydeck as pdk

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Explainable ST-GNN Outbreak Surveillance",
    page_icon="🔎",
    layout="wide"
)

# ---------------------------------------------------------
# MODEL ARCHITECTURE
# ---------------------------------------------------------
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
    def __init__(self, in_features, hidden_dim):
        super(PhysicsSTGNN, self).__init__()
        self.gcn1 = STGNNCell(in_features, hidden_dim)
        self.gcn2 = STGNNCell(hidden_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=1)
        self.out_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x_seq, adj):
        B, T, N, F = x_seq.shape
        gcn_list = []
        for t in range(T):
            h1 = self.gcn1(x_seq[:, t], adj)
            h2 = self.gcn2(h1, adj) + h1
            gcn_list.append(h2)
            
        gcn_seq = torch.stack(gcn_list, dim=1)
        gcn_seq = gcn_seq.permute(0, 2, 1, 3).contiguous().view(B * N, T, -1)
        _, (hn, _) = self.lstm(gcn_seq)
        return self.out_head(hn[-1]).view(B, N)

# ---------------------------------------------------------
# DATA & SURVEILLANCE GRAPH LOADERS
# ---------------------------------------------------------
@st.cache_data
def load_data_and_graph(disease_name):
    seed_map = {"Dengue": 42, "Malaria": 101, "Japanese Encephalitis": 202, "Chikungunya": 303}
    seed = seed_map.get(disease_name, 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

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
        "Malaria": 0.85,
        "Japanese Encephalitis": 0.35,
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
        dist_mult = 1.25 if dist["name"] in ["Kolkata", "North 24 Parganas", "Howrah", "Murshidabad", "Jalpaiguri"] else 0.85
        dist_temp = seasonal_temp + np.random.normal(0, 0.4, N_WEEKS)
        dist_preci = np.maximum(0, seasonal_preci + np.random.normal(0, 5, N_WEEKS))
        dist_lai = np.maximum(0.2, seasonal_lai + np.random.normal(0, 0.03, N_WEEKS))
        biobert_risk = np.clip(0.12 * dist_preci / 20.0 + np.random.normal(0.25, 0.015, N_WEEKS), 0, 1)
        
        lagged_preci = np.roll(dist_preci, 3)
        lagged_preci[:3] = dist_preci[:3]
        
        base_cases = disease_scale * dist_mult * (15.0 + 2.1 * (dist_temp - 20) + 0.55 * lagged_preci + 30.0 * dist_lai + 75.0 * biobert_risk)
        cases = np.maximum(0, base_cases + np.random.normal(0, 1.2, N_WEEKS))
        
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
    spatial_adj = np.exp(-dist_matrix / (0.5 * np.std(dist_matrix)))
    case_matrix = df.pivot(index='week', columns='district_idx', values='cases').values
    corr_matrix = np.maximum(0, np.nan_to_num(np.corrcoef(case_matrix.T), 0))

    A_dcmg = 0.6 * spatial_adj + 0.4 * corr_matrix
    deg = np.diag(np.sum(A_dcmg, axis=1)**(-0.5))
    A_norm = np.dot(np.dot(deg, A_dcmg), deg)

    # Weekly dates generation starting from 2022-01-02
    test_dates = pd.date_range(start="2022-01-02", periods=N_WEEKS, freq="W").strftime('%Y-%m-%d').tolist()

    return districts_wb, df, A_norm, test_dates

# ---------------------------------------------------------
# FAST INFERENCE LOADER (ZERO TRAINING OVERHEAD)
# ---------------------------------------------------------
def get_model_and_predictions(disease_name, df, A_norm):
    file_prefix = disease_name.lower().replace(" ", "_")
    model_path = f"saved_model_{file_prefix}.pt"
    
    N_WEEKS = 156
    SEQ_LEN = 6
    features_cols = ["temp", "preci", "lai", "biobert_signal", "cases"]
    
    feature_matrix = np.array([df[df["week"] == w][features_cols].values for w in range(N_WEEKS)])
    mean_f = np.mean(feature_matrix, axis=(0,1), keepdims=True)
    std_f = np.std(feature_matrix, axis=(0,1), keepdims=True)
    feature_matrix_norm = (feature_matrix - mean_f) / std_f

    mean_cases, std_cases = mean_f[0, 0, 4], std_f[0, 0, 4]

    X, Y = [], []
    for t in range(N_WEEKS - SEQ_LEN):
        X.append(feature_matrix_norm[t : t + SEQ_LEN])
        Y.append(feature_matrix_norm[t + SEQ_LEN, :, 4])

    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)

    train_size = int(len(X) * 0.8)
    X_test, Y_test = X[train_size:], Y[train_size:]

    A_tensor = torch.tensor(A_norm, dtype=torch.float32)
    model = PhysicsSTGNN(in_features=5, hidden_dim=32)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model.eval()
    with torch.no_grad():
        y_pred_actual = model(X_test, A_tensor).numpy() * std_cases + mean_cases
        y_true_actual = Y_test.numpy() * std_cases + mean_cases

    return y_pred_actual, y_true_actual, train_size, SEQ_LEN

# ---------------------------------------------------------
# DASHBOARD CONTROLS & SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("⚙️ Surveillance Settings")
disease_options = ["Dengue", "Malaria", "Japanese Encephalitis", "Chikungunya"]
selected_disease = st.sidebar.selectbox("Select Disease:", disease_options)

districts_wb, df, A_norm, test_dates = load_data_and_graph(selected_disease)
y_pred_actual, y_true_actual, train_size, SEQ_LEN = get_model_and_predictions(selected_disease, df, A_norm)

# Test dates slice corresponding to test predictions
available_test_dates = test_dates[train_size + SEQ_LEN :]

# Main Application Tabs
tab_explain, tab_map_overview = st.tabs(["🔎 Explainable ST-GNN / DCMG Map", "🗺️ Geographical Overview"])

# ---------------------------------------------------------
# TAB 1: EXPLAINABLE ST-GNN / DCMG MAP (NEW FEATURE)
# ---------------------------------------------------------
with tab_explain:
    st.title("🔎 Explainable ST-GNN / DCMG map")
    st.caption(f"Interactive West Bengal map for the {selected_disease} DCMG ST-GNN experiment. Select a test date and target district to inspect prediction error and the strongest incoming graph relationships.")
    
    # Selection Controls
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        selected_date = st.selectbox("Forecast week", available_test_dates, index=len(available_test_dates)-1)
    with col_sel2:
        district_names = [d["name"] for d in districts_wb]
        selected_target = st.selectbox("Target district", district_names, index=1) # Default: Howrah
        
    date_idx = available_test_dates.index(selected_date)
    target_idx = district_names.index(selected_target)

    # Calculate actual, predicted, and absolute error metrics
    actual_val = max(0.0, float(y_true_actual[date_idx, target_idx]))
    predicted_val = max(0.0, float(y_pred_actual[date_idx, target_idx]))
    abs_error = abs(actual_val - predicted_val)

    # Compute Incoming DCMG Links and Weights
    incoming_weights = A_norm[:, target_idx].copy()
    
    # Filter non-zero incoming links (excluding self-loops)
    incoming_indices = [i for i in range(len(districts_wb)) if i != target_idx]
    incoming_indices = sorted(incoming_indices, key=lambda i: incoming_weights[i], reverse=True)[:5]
    
    total_top_weight = sum(incoming_weights[i] for i in incoming_indices)

    # Render Metric Row
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("Actual", f"{actual_val:.1f}" if actual_val > 0 else "0")
    col_m2.metric("Predicted", f"{predicted_val:.1f}")
    col_m3.metric("Absolute error", f"{abs_error:.1f}")
    col_m4.metric("Incoming DCMG links", len(incoming_indices))

    st.markdown("---")
    
    # Influencer Table Section
    st.subheader(f"🧠 What is influencing {selected_target}?")

    table_data = []
    for idx in incoming_indices:
        w = float(incoming_weights[idx])
        inf_pct = (w / total_top_weight * 100) if total_top_weight > 0 else 0.0
        table_data.append({
            "Source district": districts_wb[idx]["name"],
            "weight": f"{w:.3f}",
            "Influence (%)": f"{inf_pct:.1f}"
        })

    df_influencers = pd.DataFrame(table_data)
    st.dataframe(
        df_influencers,
        use_container_width=True,
        hide_index=True
    )

    st.caption("These are normalized DCMG graph weights. They describe how strongly the learned graph routes information from source districts into the target; they are not causal effects.")

# ---------------------------------------------------------
# TAB 2: GEOGRAPHICAL OVERVIEW MAP
# ---------------------------------------------------------
with tab_map_overview:
    st.subheader(f"3D Spatial Risk Map ({selected_disease})")
    nodes_data = []
    for i, d in enumerate(districts_wb):
        p_val = float(y_pred_actual[-1, i])
        nodes_data.append({
            "name": d["name"],
            "lat": d["lat"],
            "lng": d["lng"],
            "cases": round(p_val, 1),
            "radius": int(p_val * 250 + 4000)
        })

    view_state = pdk.ViewState(latitude=23.8000, longitude=87.8000, zoom=6.5, pitch=30)
    layer = pdk.Layer(
        "ScatterplotLayer",
        pd.DataFrame(nodes_data),
        get_position=["lng", "lat"],
        get_fill_color="[235, 52, 52, 180]",
        get_radius="radius",
        pickable=True
    )
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "{name}: {cases} predicted cases"}))
