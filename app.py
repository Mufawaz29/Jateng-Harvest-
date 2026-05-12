import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import math

# ==========================================
# PAGE CONFIGURATION & THEME STYLE
# ==========================================
st.set_page_config(
    page_title="Jateng Harvest Dashboard | Sistem Perencanaan Logistik & Estimasi Panen",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling with Glassmorphism, tailored animations, and high-end colors
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        color: #ffffff;
    }

    /* Background and slate gradient container */
    .stApp {
        background: radial-gradient(circle at 50% 10%, #1e293b 0%, #0f172a 70%);
        color: #f1f5f9;
    }

    /* Sidebar Glassmorphism styling */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.85) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Header card */
    .header-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(245, 158, 11, 0.1) 100%);
        border: 1px solid rgba(16, 185, 129, 0.25);
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
        backdrop-filter: blur(12px);
    }

    /* General Glassmorphism container */
    .glass-card {
        background: rgba(30, 41, 59, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }

    /* Recommendation card micro-animations */
    .rec-card {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.03);
        border-left: 5px solid #10b981;
        padding: 16px 20px;
        border-radius: 12px;
        margin-bottom: 12px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .rec-card:hover {
        transform: translateY(-4px) scale(1.01);
        background: rgba(30, 41, 59, 0.7);
        border-color: rgba(16, 185, 129, 0.2);
        box-shadow: 0 10px 20px -5px rgba(0, 0, 0, 0.4);
    }
    
    .rec-card-gold {
        border-left-color: #f59e0b !important;
    }
    .rec-card-blue {
        border-left-color: #3b82f6 !important;
    }

    /* Stat values */
    .metric-val {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #10b981, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .metric-val-gold {
        background: linear-gradient(90deg, #f59e0b, #fbbf24) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }
    
    /* Buttons custom styles */
    .stButton>button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 15px -3px rgba(16, 185, 129, 0.4) !important;
        transition: all 0.2s ease-in-out !important;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px -5px rgba(16, 185, 129, 0.6) !important;
    }
    .stButton>button:active {
        transform: translateY(1px) !important;
    }

    /* Subtext formatting */
    .sub-text {
        font-size: 0.85rem;
        color: #94a3b8;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Month list and mapping constants
MONTHS_ID = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']
MONTH_MAP_ID_TO_NUM = {name: i+1 for i, name in enumerate(MONTHS_ID)}
MONTH_MAP_NUM_TO_ID = {i+1: name for i, name in enumerate(MONTHS_ID)}

# ==========================================
# DATA & MODEL LOADING (WITH CACHING)
# ==========================================
@st.cache_resource
def load_models_and_encoders():
    """Loads Machine Learning Model and Label Encoders."""
    model_path = "model_padi_final.pkl"
    le_kab_path = "le_kab.pkl"
    le_kec_path = "le_kec.pkl"
    
    if os.path.exists(model_path) and os.path.exists(le_kab_path) and os.path.exists(le_kec_path):
        model = joblib.load(model_path)
        le_kab = joblib.load(le_kab_path)
        le_kec = joblib.load(le_kec_path)
        return model, le_kab, le_kec
    else:
        st.error("Gagal memuat berkas model (`.pkl`). Pastikan berkas-berkas pkl berada dalam direktori aplikasi.")
        return None, None, None

@st.cache_data
def load_historical_data():
    """Loads the core central Java 2025 agricultural dataset."""
    csv_path = "data_siap_ml.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    else:
        st.error("Gagal memuat dataset `data_siap_ml.csv`.")
        return pd.DataFrame()

# Initialize resources
model, le_kab, le_kec = load_models_and_encoders()
df_hist = load_historical_data()

# MAE derived from model validation evaluation
MODEL_MAE = 63.4

# ==========================================
# LOGICAL HELPER FUNCTIONS (AUTOMATION CORE)
# ==========================================
def get_historical_luas_tanam(df, kabupaten, kecamatan, month_num):
    """
    Extracts the average historical Luas Tanam for a given area and month.
    Includes smart fallback to Kabupaten average or Central Java average.
    """
    if df.empty:
        return 0.0
    
    # Try exact match (Kabupaten and Kecamatan)
    sub_df = df[(df['Kabupaten'].str.lower() == kabupaten.lower()) & (df['Kecamatan'].str.lower() == kecamatan.lower())]
    
    # Fallback 1: Match by Kabupaten only
    if sub_df.empty:
        sub_df = df[df['Kabupaten'].str.lower() == kabupaten.lower()]
        
    # Fallback 2: No area match, return regional average
    if sub_df.empty:
        return float(df['Luas_Tanam'].mean())
        
    # Filter by specific month_num
    month_df = sub_df[sub_df['Bulan_Angka'] == month_num]
    if not month_df.empty:
        return float(month_df['Luas_Tanam'].mean())
    else:
        # Month specific fallback: general average for this sub_df
        return float(sub_df['Luas_Tanam'].mean())

def construct_prediction_features(df, kabupaten, kecamatan, current_month_num, current_luas_tanam, target_month_num):
    """
    Transformational Logic: Constructs the 7 required model features for predicting Luas_Panen 
    of a target month automatically behind the scenes.
    Features: ['Kab_Enc', 'Kec_Enc', 'Bulan_Angka', 'Luas_Tanam', 'Fase_Vegetatif_Est', 'Fase_Generatif_Est', 'Fase_Siap_Panen_Est']
    """
    # 1. Encode Kabupaten and Kecamatan
    try:
        kab_encoded = le_kab.transform([kabupaten])[0]
    except Exception:
        # Fallback to closest matching class or first class if brand new
        kab_encoded = le_kab.transform([le_kab.classes_[0]])[0]
        
    try:
        kec_encoded = le_kec.transform([kecamatan])[0]
    except Exception:
        # Fallback to closest matching class or first class if brand new
        kec_encoded = le_kec.transform([le_kec.classes_[0]])[0]
        
    # 2. Determine raw feature parameters using crop lag logic
    # Target month Luas Tanam
    target_luas_tanam = current_luas_tanam if target_month_num == current_month_num else get_historical_luas_tanam(df, kabupaten, kecamatan, target_month_num)
    
    # Growth phase 1: Vegetatif_Est (1 month prior)
    t_minus_1 = target_month_num - 1 if target_month_num > 1 else 12
    if t_minus_1 == current_month_num:
        fase_vegetatif = current_luas_tanam
    else:
        fase_vegetatif = get_historical_luas_tanam(df, kabupaten, kecamatan, t_minus_1)
        
    # Growth phase 2: Generatif_Est (2 months prior)
    t_minus_2 = target_month_num - 2
    if t_minus_2 <= 0:
        t_minus_2 += 12
    if t_minus_2 == current_month_num:
        fase_generatif = current_luas_tanam
    else:
        fase_generatif = get_historical_luas_tanam(df, kabupaten, kecamatan, t_minus_2)
        
    # Growth phase 3: Siap_Panen_Est (3 months prior)
    t_minus_3 = target_month_num - 3
    if t_minus_3 <= 0:
        t_minus_3 += 12
    if t_minus_3 == current_month_num:
        fase_siap_panen = current_luas_tanam
    else:
        fase_siap_panen = get_historical_luas_tanam(df, kabupaten, kecamatan, t_minus_3)
        
    return {
        'Kab_Enc': kab_encoded,
        'Kec_Enc': kec_encoded,
        'Bulan_Angka': target_month_num,
        'Luas_Tanam': target_luas_tanam,
        'Fase_Vegetatif_Est': fase_vegetatif,
        'Fase_Generatif_Est': fase_generatif,
        'Fase_Siap_Panen_Est': fase_siap_panen
    }

def get_3_month_predictions(df, kabupaten, kecamatan, current_month_name, current_luas_tanam):
    """Runs harvest area forecasting for the next 3 months using the RandomForest model."""
    predictions = []
    current_month_num = MONTH_MAP_ID_TO_NUM[current_month_name]
    
    for month_offset in range(1, 4):
        target_month_num = current_month_num + month_offset
        if target_month_num > 12:
            target_month_num -= 12
            
        target_month_name = MONTH_MAP_NUM_TO_ID[target_month_num]
        
        # Build features using growth cycle lag logic
        features_dict = construct_prediction_features(df, kabupaten, kecamatan, current_month_num, current_luas_tanam, target_month_num)
        
        # Prepare feature vector in exact training order
        feature_vector = [
            features_dict['Kab_Enc'],
            features_dict['Kec_Enc'],
            features_dict['Bulan_Angka'],
            features_dict['Luas_Tanam'],
            features_dict['Fase_Vegetatif_Est'],
            features_dict['Fase_Generatif_Est'],
            features_dict['Fase_Siap_Panen_Est']
        ]
        
        # Run ML model prediction
        pred_log = model.predict([feature_vector])[0]
        
        # Apply expm1 to inverse the log1p transform used during model training
        pred_ha = max(0.0, np.expm1(pred_log))
        
        predictions.append({
            'Bulan_Angka': target_month_num,
            'Bulan_Nama': target_month_name,
            'Features': features_dict,
            'Prediksi_Ha': round(pred_ha, 2),
            'Min_Ha': round(max(0.0, pred_ha - MODEL_MAE), 2),
            'Max_Ha': round(pred_ha + MODEL_MAE, 2)
        })
        
    return predictions

# ==========================================
# NAVIGATION & SIDEBAR MANAGEMENT
# ==========================================
st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 10px 0;">
        <h2 style="margin:0; font-size:1.8rem; background: linear-gradient(135deg, #10b981 0%, #f59e0b 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Jateng Harvest
        </h2>
        <p style="color:#64748b; font-size:0.85rem; font-weight:500; margin-top:5px;">Sistem Perencanaan & Perhitungan Panen</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("<hr style='border: 1px solid rgba(255,255,255,0.05); margin-top:0px;'>", unsafe_allow_html=True)

# Unique values for responsive location dropdowns
if not df_hist.empty:
    list_kabupaten = sorted(df_hist['Kabupaten'].unique())
else:
    list_kabupaten = sorted(le_kab.classes_)

selected_kab = st.sidebar.selectbox("Pilih Kabupaten", list_kabupaten)

# Responsive Kecamatan filtering
if not df_hist.empty:
    list_kecamatan = sorted(df_hist[df_hist['Kabupaten'] == selected_kab]['Kecamatan'].unique())
else:
    list_kecamatan = [k for k in sorted(le_kec.classes_) if k in df_hist['Kecamatan'].values] # safety check

selected_kec = st.sidebar.selectbox("Pilih Kecamatan", list_kecamatan)

st.sidebar.markdown("### 🚜 Kondisi Lahan Terkini")
selected_month = st.sidebar.selectbox("Pilih Bulan Tanam Terakhir", MONTHS_ID, index=4) # Default Mei

# Calculate historical baseline as an intuitive default value for Luas Tanam input
default_luas_tanam = 100.0
if not df_hist.empty:
    baseline = get_historical_luas_tanam(df_hist, selected_kab, selected_kec, MONTH_MAP_ID_TO_NUM[selected_month])
    if baseline > 0:
        default_luas_tanam = round(baseline, 2)

manual_luas_tanam = st.sidebar.number_input(
    "Luas Tanam Saat Ini (Ha)",
    min_value=0.0,
    max_value=15000.0,
    value=default_luas_tanam,
    step=10.0,
    help="Masukkan luas lahan padi yang ditanam pada bulan ini di wilayah terpilih."
)

st.sidebar.markdown("<hr style='border: 1px solid rgba(255,255,255,0.05);'>", unsafe_allow_html=True)

# File uploader for Simontadi
st.sidebar.markdown("### 📥 Integrasi Simontadi")
uploaded_file = st.sidebar.file_uploader(
    "Unggah Data Simontadi 2026",
    type=["csv", "xlsx"],
    help="Unggah laporan tanam bulanan versi Simontadi untuk memproses prediksi secara otomatis/batch."
)

# Custom metrics inputs for recommendations
st.sidebar.markdown("### ⚙️ Parameter Produktivitas")
productivity_rate = st.sidebar.slider(
    "Rata-rata Produktivitas (Ton/Ha)",
    min_value=3.0,
    max_value=10.0,
    value=5.8,
    step=0.1,
    help="Ubah jika ingin menyesuaikan estimasi berat hasil panen gabah per hektar."
)

st.sidebar.markdown(
    """
    <div style="background: rgba(30,41,59,0.5); padding:15px; border-radius:12px; border: 1px solid rgba(255,255,255,0.03); margin-top: 15px;">
        <span style="font-size:0.75rem; color:#64748b; font-weight:600; text-transform:uppercase; letter-spacing:1px;">Prinsip Aplikasi</span>
        <p style="font-size:0.8rem; color:#94a3b8; margin: 5px 0 0 0; line-height: 1.4;">
            <b>Simontadi:</b> "Apa yang terjadi"<br>
            <b>Jateng Harvest:</b> "Apa yang harus Anda siapkan"
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================
# MAIN PAGE ROUTING & DASHBOARD BODY
# ==========================================

# 1. Welcome Header and digital landing
st.markdown(
    """
    <div class="header-card">
        <span style="background-color: #10b981; color: #0f172a; padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">
            Sugeng Rawuh! Jateng Harvest
        </span>
        <h1 style="margin: 10px 0 5px 0; font-size: 2.5rem; letter-spacing: -1px; background: linear-gradient(90deg, #ffffff 0%, #cbd5e1 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Mari Siapkan Panen Anda Secara Terukur
        </h1>
        <p style="margin: 0; color: #94a3b8; font-size: 1.1rem; font-weight: 400; line-height:1.5;">
            Mengolah data tanam mentah <b>Simontadi</b> menjadi estimasi perencanaan logistik dan operasional pasca-panen secara otomatis dan terstruktur.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Handle file upload processing if present
simontadi_data = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            simontadi_data = pd.read_csv(uploaded_file)
        else:
            simontadi_data = pd.read_excel(uploaded_file)
        st.success(f"✔️ Berkas '{uploaded_file.name}' berhasil diunggah! Sistem siap memproses batch prediksi.")
    except Exception as e:
        st.error(f"Gagal membaca berkas: {e}")

# MAIN TABS:
# Tab 1: AI Predictor (The Core Engine)
# Tab 2: Regional 2025 Summary (Historical Backdrop/Novelty Proof)
# Tab 3: Simontadi 2026 Batch Predictor (Digital Integration)
# Tab 4: Price & Selling Education (Edukasi Harga)

tab_predict, tab_batch, tab_summary, tab_education = st.tabs([
    "🔮 Estimasi & Instruksi Pasca-Panen",
    "📊 Estimasi Massal Laporan Simontadi 2026",
    "📈 Ringkasan Data Tanam Jateng 2025",
    "💡 Edukasi Tren Harga & Waktu Jual"
])

# ------------------------------------------
# TAB 1: INDIVIDUAL AI PREDICTOR
# ------------------------------------------
with tab_predict:
    # Quick Summary Row of the Chosen Area
    st.markdown(f"### 📍 Analisis Lahan: Kecamatan **{selected_kec}**, {selected_kab}")
    
    col_input_summary, col_calc_btn = st.columns([3, 1])
    with col_input_summary:
        st.markdown(
            f"""
            <div style="background: rgba(255,255,255,0.02); padding: 12px 20px; border-radius:12px; border:1px solid rgba(255,255,255,0.03);">
                📅 <b>Bulan Input:</b> {selected_month} &nbsp;|&nbsp; 
                🌱 <b>Luas Tanam:</b> {manual_luas_tanam:,.1f} Ha &nbsp;|&nbsp;
                ⚙️ <b>Asumsi Produktivitas:</b> {productivity_rate:.1f} Ton/Ha
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_calc_btn:
        run_prediction = st.button("🚀 Hitung Persiapan Panen")

    # Run predictions automatically or when button is clicked
    if True: # Always show current selected as default, button adds highlight
        predictions = get_3_month_predictions(df_hist, selected_kab, selected_kec, selected_month, manual_luas_tanam)
        
        # Display Prediction Cards
        st.markdown("<h4 style='margin-top:25px; margin-bottom:15px;'>🔮 Estimasi Luas Panen 3 Bulan ke Depan</h4>", unsafe_allow_html=True)
        
        pred_cols = st.columns(3)
        for idx, pred in enumerate(predictions):
            with pred_cols[idx]:
                st.markdown(
                    f"""
                    <div class="glass-card" style="border-top: 4px solid #10b981 !important; text-align: center;">
                        <span style="font-size:0.8rem; color:#10b981; font-weight:700; text-transform:uppercase; letter-spacing:1px;">
                            Bulan Ke-{idx+1} ({pred['Bulan_Nama']})
                        </span>
                        <div class="metric-val" style="margin: 15px 0 5px 0;">{pred['Prediksi_Ha']:,.1f} Ha</div>
                        <div class="sub-text" style="color:#e2e8f0; font-size: 0.95rem; font-weight:500;">
                            Rentang Batas Estimasi Toleransi:
                        </div>
                        <div style="font-size:1.1rem; color:#fbbf24; font-weight:700; margin-top:2px;">
                            {pred['Min_Ha']:,.1f} - {pred['Max_Ha']:,.1f} Ha
                        </div>
                        <p class="sub-text" style="margin-top:12px; line-height:1.4;">
                            *Rentang di atas menunjukkan batas toleransi wajar berdasarkan perhitungan historis wilayah (63.4 Ha).
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Plotly Prediction Forecast with Error Bands
        st.markdown("<h4 style='margin-top:20px; margin-bottom:15px;'>📈 Grafik Tren Estimasi Luas Panen</h4>", unsafe_allow_html=True)
        
        months_plot = [p['Bulan_Nama'] for p in predictions]
        preds_plot = [p['Prediksi_Ha'] for p in predictions]
        mins_plot = [p['Min_Ha'] for p in predictions]
        maxs_plot = [p['Max_Ha'] for p in predictions]
        
        fig_forecast = go.Figure()
        
        # Error Area/Band
        fig_forecast.add_trace(go.Scatter(
            x=months_plot + months_plot[::-1],
            y=maxs_plot + mins_plot[::-1],
            fill='toself',
            fillcolor='rgba(16, 185, 129, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name="Batas Toleransi Perhitungan"
        ))
        
        # Main Line
        fig_forecast.add_trace(go.Scatter(
            x=months_plot,
            y=preds_plot,
            line=dict(color='#10b981', width=4),
            marker=dict(size=10, color='#f59e0b', symbol='diamond'),
            mode='lines+markers+text',
            text=[f"{p:,.1f} Ha" for p in preds_plot],
            textposition="top center",
            name="Estimasi Luas Panen Utama"
        ))
        
        fig_forecast.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#cbd5e1',
            margin=dict(l=40, r=40, t=20, b=40),
            height=300,
            xaxis=dict(showgrid=False),
            yaxis=dict(title="Luas Panen (Hektar)", showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_forecast, use_container_width=True)

        # Rencana Instruksi Persiapan Panen Otomatis
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%); border-left: 5px solid #f59e0b; padding: 20px; border-radius: 12px; margin-top:30px; margin-bottom:20px;">
                <h3 style="margin:0 0 5px 0; font-size:1.4rem; color: #f59e0b; display:flex; align-items:center; gap: 8px;">
                    🌾 Rencana Instruksi Persiapan Panen Otomatis
                </h3>
                <p style="margin:0; color:#cbd5e1; font-size:0.95rem;">
                    Berdasarkan estimasi luas panen di atas, sistem secara otomatis merancang instruksi kebutuhan karung, alat pemanen, dan lantai jemur untuk masing-masing bulan.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Tabs for instructions of Month 1, 2, and 3
        inst_tabs = st.tabs([f"📦 Persiapan Bulan {p['Bulan_Nama']}" for p in predictions])
        
        for idx, pred in enumerate(predictions):
            with inst_tabs[idx]:
                ha = pred['Prediksi_Ha']
                # Harvest statistics calculations
                total_production_tons = ha * productivity_rate
                sacks_needed = math.ceil(total_production_tons * 20) # 20 sacks of 50kg per ton
                harvesters_needed = math.ceil(ha / 15.0) # 1 harvester handles 1.5 Ha per day over 10 days
                drying_area_sqm = total_production_tons * 12 # 12 sqm per ton
                storage_vol_m3 = total_production_tons * 1.8 # 1.8 m3 per ton
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(
                        f"""
                        <div class="rec-card">
                            <span style="font-size:0.75rem; color:#10b981; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">LOGISTIK PENGEMASAN</span>
                            <h4 style="margin: 5px 0; font-size:1.15rem;">📦 Pembelian Karung Gabah</h4>
                            <div style="font-size:1.6rem; font-weight:800; color:#10b981; margin:10px 0 5px 0;">{sacks_needed:,} Lembar</div>
                            <p style="font-size:0.8rem; color:#94a3b8; margin:0; line-height:1.4;">
                                Berdasarkan estimasi produksi <b>{total_production_tons:,.1f} Ton</b> gabah basah. Dibutuhkan karung kapasitas 50 Kg.
                            </p>
                            <div style="background: rgba(16, 185, 129, 0.1); border-radius: 8px; padding: 6px 12px; margin-top:12px; font-size:0.8rem; color:#34d399; font-weight:500;">
                                💡 Pesan karung selambatnya 2 minggu sebelum panen.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                with col2:
                    st.markdown(
                        f"""
                        <div class="rec-card rec-card-gold">
                            <span style="font-size:0.75rem; color:#f59e0b; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">OPERASIONAL LAPANGAN</span>
                            <h4 style="margin: 5px 0; font-size:1.15rem;">🚜 Sewa Combine Harvester</h4>
                            <div style="font-size:1.6rem; font-weight:800; color:#f59e0b; margin:10px 0 5px 0;">{harvesters_needed} Unit Alat</div>
                            <p style="font-size:0.8rem; color:#94a3b8; margin:0; line-height:1.4;">
                                Diperlukan untuk menyelesaikan panen seluas <b>{ha:,.1f} Ha</b> dalam durasi jendela aman panen raya 10 hari.
                            </p>
                            <div style="background: rgba(245, 158, 11, 0.1); border-radius: 8px; padding: 6px 12px; margin-top:12px; font-size:0.8rem; color:#fbbf24; font-weight:500;">
                                💡 Hubungi kelompok tani (UPJA) terdekat untuk pemesanan.
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                with col3:
                    st.markdown(
                        f"""
                        <div class="rec-card rec-card-blue">
                            <span style="font-size:0.75rem; color:#3b82f6; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">PASCA PANEN & PENYIMPANAN</span>
                            <h4 style="margin: 5px 0; font-size:1.15rem;">🏢 Lantai Jemur & Gudang</h4>
                            <div style="font-size:1.35rem; font-weight:800; color:#3b82f6; margin:10px 0 5px 0;">📐 {drying_area_sqm:,.0f} m² Lantai Jemur</div>
                            <div style="font-size:1.35rem; font-weight:800; color:#60a5fa; margin:0 0 10px 0;">📦 {storage_vol_m3:,.0f} m³ Gudang</div>
                            <p style="font-size:0.8rem; color:#94a3b8; margin:0; line-height:1.4;">
                                Luasan lantai jemur gabah agar kadar air turun ke 14% serta kapasitas ruang penyimpanan terstandar.
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

# ------------------------------------------
# TAB 2: BATCH PREDICTION SIMONTADI 2026
# ------------------------------------------
with tab_batch:
    st.markdown("### 📊 Proses Batch Perencanaan Simontadi 2026")
    st.markdown(
        """
        Gunakan menu ini untuk memproses data dari sistem eksternal **Simontadi**. Anda cukup mengunggah data tanam bulanan, 
        dan sistem akan menghitung estimasi panen raya serta merancang logistik persiapan panen untuk seluruh wilayah sekaligus!
        """
    )
    
    # Showcase Template for users to download
    st.markdown("#### 📥 Contoh/Template Struktur CSV Simontadi")
    template_data = pd.DataFrame([
        {"Kabupaten": "Banjarnegara", "Kecamatan": "Banjarmangu", "Bulan": "Mei", "Luas_Tanam": 65.24},
        {"Kabupaten": "Banyumas", "Kecamatan": "Ajibarang", "Bulan": "Mei", "Luas_Tanam": 405.46},
        {"Kabupaten": "Batang", "Kecamatan": "Bandar", "Bulan": "Mei", "Luas_Tanam": 177.80}
    ])
    st.dataframe(template_data, use_container_width=True)
    
    # Process uploaded file
    if simontadi_data is not None:
        st.markdown("#### 🔍 Pratinjau Data Simontadi yang Diunggah")
        st.dataframe(simontadi_data.head(10), use_container_width=True)
        
        # Verify columns
        required_cols = ['kabupaten', 'kecamatan', 'bulan', 'luas_tanam']
        actual_cols = [c.lower() for c in simontadi_data.columns]
        
        missing_cols = [c for c in required_cols if c not in actual_cols]
        if missing_cols:
            st.error(f"Kolom berkas tidak sesuai. Kolom wajib yang hilang: {missing_cols}. Pastikan kolom berisi Kabupaten, Kecamatan, Bulan, dan Luas_Tanam.")
        else:
            # Map columns cleanly
            col_mapping = {c: simontadi_data.columns[i] for i, c in enumerate(actual_cols)}
            
            run_batch = st.button("🔮 Jalankan Proses Perhitungan Estimasi & Logistik Massal")
            
            if run_batch:
                with st.spinner("Menghitung estimasi luas panen dan rincian logistik untuk seluruh baris data..."):
                    results = []
                    
                    for index, row_data in simontadi_data.iterrows():
                        row_kab = row_data[col_mapping['kabupaten']]
                        row_kec = row_data[col_mapping['kecamatan']]
                        row_bulan = row_data[col_mapping['bulan']]
                        row_luas_tanam = float(row_data[col_mapping['luas_tanam']])
                        
                        # Identify starting month number
                        if row_bulan in MONTHS_ID:
                            month_num = MONTH_MAP_ID_TO_NUM[row_bulan]
                        else:
                            # default fallback
                            month_num = 5
                            
                        # Predict next month (Month + 1)
                        target_month_num = month_num + 1
                        if target_month_num > 12:
                            target_month_num -= 12
                        target_month_name = MONTH_MAP_NUM_TO_ID[target_month_num]
                        
                        # Build features
                        features_dict = construct_prediction_features(df_hist, row_kab, row_kec, month_num, row_luas_tanam, target_month_num)
                        
                        feature_vector = [
                            features_dict['Kab_Enc'],
                            features_dict['Kec_Enc'],
                            features_dict['Bulan_Angka'],
                            features_dict['Luas_Tanam'],
                            features_dict['Fase_Vegetatif_Est'],
                            features_dict['Fase_Generatif_Est'],
                            features_dict['Fase_Siap_Panen_Est']
                        ]
                        
                        try:
                            pred_log = model.predict([feature_vector])[0]
                            pred_ha = max(0.0, np.expm1(pred_log))
                        except Exception:
                            pred_ha = 0.0
                            
                        # Recommendation math
                        tons = pred_ha * productivity_rate
                        sacks = math.ceil(tons * 20)
                        harvesters = math.ceil(pred_ha / 15.0)
                        
                        results.append({
                            "Kabupaten": row_kab,
                            "Kecamatan": row_kec,
                            "Bulan Tanam": row_bulan,
                            "Luas Tanam (Ha)": row_luas_tanam,
                            "Bulan Estimasi Panen": target_month_name,
                            "Estimasi Luas Panen (Ha)": round(pred_ha, 1),
                            "Estimasi Produksi (Ton)": round(tons, 1),
                            "Kebutuhan Karung (Pcs)": sacks,
                            "Kebutuhan Harvester (Unit)": harvesters
                        })
                        
                    df_results = pd.DataFrame(results)
                    
                    st.success("🎉 Berhasil memproses perhitungan estimasi massal!")
                    st.markdown("#### 📋 Hasil Perhitungan Estimasi & Rekomendasi Logistik Wilayah")
                    st.dataframe(df_results, use_container_width=True)
                    
                    # File downloader
                    csv_bytes = df_results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Unduh Laporan Logistik & Estimasi Lengkap (CSV)",
                        data=csv_bytes,
                        file_name="Laporan_Logistik_Panen_Simontadi_2026.csv",
                        mime="text/csv"
                    )
    else:
        st.info("💡 Silakan unggah berkas Simontadi pada sidebar untuk mengaktifkan fitur Batch Processing.")

# ------------------------------------------
# TAB 3: CENTRAL JAVA 2025 SUMMARY (HISTORICAL BACKGROUND)
# ------------------------------------------
with tab_summary:
    st.markdown("### 📊 Ringkasan Panen Jawa Tengah 2025")
    st.markdown(
        """
        Dashboard ini divalidasi dan dibangun di atas data riil pertanian Jawa Tengah. Berikut adalah gambaran sebaran 
        dan pola tanam di Central Java untuk memperkaya keputusan Anda.
        """
    )
    
    if not df_hist.empty:
        # High level stats
        tot_luas_tanam = df_hist['Luas_Tanam'].sum()
        tot_luas_panen = df_hist['Luas_Panen'].sum()
        avg_fase_siap = df_hist['Fase_Siap_Panen_Est'].mean()
        
        c_stats_1, c_stats_2, c_stats_3 = st.columns(3)
        with c_stats_1:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align: center; border-left: 5px solid #10b981;">
                    <span style="font-size:0.8rem; color:#94a3b8; font-weight:700; text-transform:uppercase;">TOTAL HISTORIS LUAS TANAM</span>
                    <div class="metric-val" style="margin:10px 0 0 0;">{tot_luas_tanam:,.1f} Ha</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c_stats_2:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align: center; border-left: 5px solid #f59e0b;">
                    <span style="font-size:0.8rem; color:#94a3b8; font-weight:700; text-transform:uppercase;">TOTAL HISTORIS LUAS PANEN</span>
                    <div class="metric-val metric-val-gold" style="margin:10px 0 0 0;">{tot_luas_panen:,.1f} Ha</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c_stats_3:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align: center; border-left: 5px solid #3b82f6;">
                    <span style="font-size:0.8rem; color:#94a3b8; font-weight:700; text-transform:uppercase;">RATA-RATA LAHAN SIAP PANEN</span>
                    <div class="metric-val" style="color:#3b82f6; margin:10px 0 0 0;">{avg_fase_siap:,.1f} Ha</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        # Top Kabupaten Visualizer
        st.markdown("#### 🏆 10 Kabupaten dengan Luas Panen Terbesar")
        df_kab_sum = df_hist.groupby('Kabupaten')['Luas_Panen'].sum().reset_index().sort_values('Luas_Panen', ascending=False).head(10)
        
        fig_kab = px.bar(
            df_kab_sum,
            y='Kabupaten',
            x='Luas_Panen',
            orientation='h',
            color='Luas_Panen',
            color_continuous_scale=['#34d399', '#10b981', '#f59e0b'],
            labels={'Luas_Panen': 'Total Luas Panen (Ha)', 'Kabupaten': 'Kabupaten'}
        )
        fig_kab.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#cbd5e1',
            margin=dict(l=20, r=20, t=10, b=20),
            height=300,
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(showgrid=False, categoryorder='total ascending'),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_kab, use_container_width=True)
        
        # Monthly Seasonal Cycle of Planting vs Harvesting
        st.markdown("#### 📅 Pola Siklus Bulanan: Masa Tanam vs Masa Panen")
        df_monthly = df_hist.groupby('Bulan_Angka')[['Luas_Tanam', 'Luas_Panen']].mean().reset_index()
        df_monthly['Bulan'] = df_monthly['Bulan_Angka'].map(MONTH_MAP_NUM_TO_ID)
        df_monthly = df_monthly.sort_values('Bulan_Angka')
        
        fig_cycle = go.Figure()
        fig_cycle.add_trace(go.Scatter(
            x=df_monthly['Bulan'], y=df_monthly['Luas_Tanam'],
            mode='lines+markers', name='Rata-rata Luas Tanam (🌱)',
            line=dict(color='#10b981', width=3)
        ))
        fig_cycle.add_trace(go.Scatter(
            x=df_monthly['Bulan'], y=df_monthly['Luas_Panen'],
            mode='lines+markers', name='Rata-rata Luas Panen (🌾)',
            line=dict(color='#f59e0b', width=3, dash='dash')
        ))
        fig_cycle.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#cbd5e1',
            margin=dict(l=20, r=20, t=20, b=20),
            height=300,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_cycle, use_container_width=True)
        
    else:
        st.warning("Data historis tidak dapat dimuat.")

# ------------------------------------------
# TAB 4: EDUCATIONAL MODULE (GRAIN PRICES & SELLING STRATEGY)
# ------------------------------------------
with tab_education:
    st.markdown("### 💡 Edukasi Strategi Penjualan Gabah & Waktu Panen")
    st.markdown(
        """
        Sering kali petani terpaksa menjual gabah dengan harga sangat murah karena waktu panen raya bersamaan. 
        <b>Jateng Harvest</b> menyajikan modul edukasi siklus tren harga gabah regional agar Anda dapat merencanakan penjualan terbaik.
        """
    )
    
    # Static realistic price trend simulation based on Central Java local GKP (Gabah Kering Panen) standards
    prices_data = pd.DataFrame({
        'Bulan': MONTHS_ID,
        'Harga_GKP_PerKg': [7200, 7500, 6400, 6000, 6200, 6800, 7100, 6300, 6500, 7400, 7800, 8100],
        'Volume_Panen_Nasional': [15, 20, 75, 90, 80, 45, 30, 60, 50, 25, 15, 10]
    })
    
    col_edu_info, col_edu_chart = st.columns([1, 2])
    
    with col_edu_info:
        st.markdown(
            """
            <div class="glass-card" style="border-left:5px solid #fbbf24;">
                <h4 style="color:#fbbf24; margin-top:0;">💡 Tips Penjualan Gabah:</h4>
                <ul style="padding-left:20px; font-size:0.85rem; color:#cbd5e1; line-height:1.6;">
                    <li><b>Hindari Penjualan Saat Panen Raya (Maret - Mei):</b> Harga Gabah Kering Panen (GKP) cenderung mencapai titik terendah karena pasokan melimpah.</li>
                    <li><b>Gunakan Lantai Jemur Pasca Panen:</b> Mengeringkan gabah Anda hingga kadar air 14% (menjadi Gabah Kering Giling - GKG) dapat meningkatkan nilai jual hingga 25-30% lebih tinggi.</li>
                    <li><b>Jendela Emas Penjualan (November - Januari):</b> Harga melonjak tinggi di musim tanam karena minimnya ketersediaan panen di pasaran.</li>
                    <li><b>Simpan dengan Baik:</b> Gunakan instruksi kapasitas gudang di halaman utama untuk mengamankan stok gabah Anda agar terhindar dari jamur.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col_edu_chart:
        # Plotly combination chart: line chart for Price and area chart for harvest volume
        fig_edu = go.Figure()
        
        # Add bar chart representing harvest volume
        fig_edu.add_trace(go.Bar(
            x=prices_data['Bulan'], y=prices_data['Volume_Panen_Nasional'],
            name="Volume Pasokan Pasar (%)",
            marker_color='rgba(16, 185, 129, 0.25)',
            yaxis='y2'
        ))
        
        # Add line chart representing price per kg
        fig_edu.add_trace(go.Scatter(
            x=prices_data['Bulan'], y=prices_data['Harga_GKP_PerKg'],
            mode='lines+markers+text',
            name="Rata-rata Harga Gabah (IDR/Kg)",
            line=dict(color='#fbbf24', width=3),
            marker=dict(size=8, color='#d97706'),
            text=[f"Rp{val:,}" for val in prices_data['Harga_GKP_PerKg']],
            textposition="top center"
        ))
        
        fig_edu.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#cbd5e1',
            margin=dict(l=20, r=20, t=10, b=20),
            height=300,
            xaxis=dict(showgrid=False),
            yaxis=dict(title="Harga Gabah (IDR/Kg)", showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            yaxis2=dict(
                title="Saturasi Pasokan (%)",
                overlaying='y',
                side='right',
                showgrid=False,
                maxallowed=100
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_edu, use_container_width=True)

# Footer credit
st.markdown("<hr style='border: 1px solid rgba(255,255,255,0.05); margin-top:40px;'>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; padding-bottom: 20px;">
        © 2026 Jateng Harvest Dashboard. Sistem Informasi Manajemen Logistik & Perencanaan Pertanian Jawa Tengah.
    </div>
    """,
    unsafe_allow_html=True
)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# ==========================================
# 1. KONFIGURASI HALAMAN & STYLE (Disesuaikan agar tidak bentrok)
# ==========================================
# st.set_page_config(page_title="Jateng-Harvest Dashboard", layout="wide") # Dinonaktifkan karena st.set_page_config hanya boleh dipanggil sekali di awal script

st.markdown("""
    <style>
    .res-card { 
        padding: 24px; 
        border-radius: 16px; 
        background: rgba(30, 41, 59, 0.6) !important; 
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 20px; 
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. LOAD MODELS & ENCODERS
# ==========================================
@st.cache_resource
def load_assets():
    model = joblib.load('model_padi_final.pkl')
    le_kab = joblib.load('le_kab.pkl')
    le_kec = joblib.load('le_kec.pkl')
    # Load data riwayat untuk referensi (data_siap_ml.csv)
    try:
        data_history = pd.read_csv('data_siap_ml.csv')
    except:
        data_history = None
    return model, le_kab, le_kec, data_history

model, le_kab, le_kec, data_history = load_assets()

# ==========================================
# 3. HEADER & JUDUL (PERENCANAAN MANDIRI)
# ==========================================
st.markdown("<hr style='border: 1px solid rgba(255,255,255,0.05); margin-top:50px;'>", unsafe_allow_html=True)
st.title("🌾 Jateng-Harvest (Kalkulator Sederhana)")
st.subheader("Pengolahan Data Lapangan SIMOTANDI untuk Estimasi & Persiapan Panen")
st.info("Aplikasi ini membantu Anda mengestimasi luas panen dan memberikan rekomendasi persiapan logistik secara otomatis tanpa perlu login.")

# ==========================================
# 4. INPUT DATA (SIMOTANDI 2026 SELESAI)
# ==========================================
with st.sidebar:
    st.markdown("<hr style='border: 1px solid rgba(255,255,255,0.05); margin-top:20px;'>", unsafe_allow_html=True)
    st.header("📍 Pilih Wilayah (Kalkulator Sederhana)")
    kab_list = sorted(le_kab.classes_)
    selected_kab_simple = st.selectbox("Pilih Kabupaten", kab_list, key="kab_simple")
    
    # Filter kecamatan berdasarkan kabupaten (jika data history tersedia)
    if data_history is not None:
        kec_list = sorted(data_history[data_history['Kabupaten'] == selected_kab_simple]['Kecamatan'].unique())
    else:
        kec_list = sorted(le_kec.classes_)
    selected_kec_simple = st.selectbox("Pilih Kecamatan", kec_list, key="kec_simple")
    
    st.divider()
    st.header("📊 Input Data Simontadi (Kalkulator Sederhana)")
    bulan_dict = {
        "Januari": 1, "Februari": 2, "Maret": 3, "April": 4, 
        "Mei": 5, "Juni": 6, "Juli": 7, "Agustus": 8, 
        "September": 9, "Oktober": 10, "November": 11, "Desember": 12
    }
    selected_bulan_simple = st.selectbox("Estimasi untuk Bulan (2026)", list(bulan_dict.keys()), key="bulan_simple")
    luas_tanam_simple = st.number_input("Input Luas Tanam (Hektar)", min_value=0.0, step=0.1, key="luas_simple")

# ==========================================
# 5. LOGIKA PREDIKSI & TRANSFORMASI
# ==========================================
# Secara cerdas mengisi fase pertumbuhan (Lag-shift simulation)
# Dalam praktek nyata, user memasukkan data bulan berjalan, AI memprediksi panennya
fase_veg = luas_tanam_simple * 0.8  # Estimasi sederhana fase vegetatif
fase_gen = luas_tanam_simple * 0.5  # Estimasi sederhana fase generatif
fase_siap = luas_tanam_simple        # Luas tanam inilah yang akan jadi panen nantinya

if st.button("🔥 HITUNG ESTIMASI & PERSIAPAN PANEN", key="btn_simple"):
    # Encoding input
    kab_enc = le_kab.transform([selected_kab_simple])[0]
    kec_enc = le_kec.transform([selected_kec_simple])[0]
    bulan_val = bulan_dict[selected_bulan_simple]
    
    # Buat DataFrame Input
    input_df = pd.DataFrame([[
        kab_enc, kec_enc, bulan_val, luas_tanam_simple, 
        fase_veg, fase_gen, fase_siap
    ]], columns=['Kab_Enc', 'Kec_Enc', 'Bulan_Angka', 'Luas_Tanam', 
                 'Fase_Vegetatif_Est', 'Fase_Generatif_Est', 'Fase_Siap_Panen_Est'])
    
    # Prediksi dengan Log Transformation Handling
    pred_log = model.predict(input_df)
    hasil_prediksi = np.expm1(pred_log)[0]
    
    # ==========================================
    # 6. DISPLAY OUTPUT (ESTIMASI & TOLERANSI)
    # ==========================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="res-card">
            <h3 style="margin-top:0;">📈 Estimasi Luas Panen</h3>
            <h1 style="color: #10b981; margin: 15px 0;">{hasil_prediksi:.2f} Ha</h1>
            <p style="color:#94a3b8; margin:0;">Bulan: {selected_bulan_simple} 2026</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        # Menampilkan Rentang Estimasi Toleransi (Kejujuran Data)
        mae_val = 115.76
        batas_bawah = max(0, hasil_prediksi - mae_val)
        batas_atas = hasil_prediksi + mae_val
        st.markdown(f"""
        <div class="res-card">
            <h3 style="margin-top:0;">🛡️ Rentang Toleransi Perhitungan</h3>
            <p style="color:#cbd5e1; margin-bottom:5px;">Estimasi Real di Lapangan:</p>
            <h4 style="color:#fbbf24; font-size:1.4rem; margin:10px 0;">{batas_bawah:.2f} Ha - {batas_atas:.2f} Ha</h4>
            <small style="color:#64748b;">*Berdasarkan batas toleransi wajar wilayah 115.76 Ha</small>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ==========================================
    # 7. NOVELTY: MODUL PERSIAPAN PANEN (ACTIONABLE)
    # ==========================================
    st.header("🚜 Rekomendasi Persiapan Panen (Instruksi Digital)")
    
    # Asumsi: 1 Ha = 5 Ton GKG, 1 Ton = 20 Karung (50kg)
    total_ton = hasil_prediksi * 5
    total_karung = total_ton * 20
    butuh_buruh = np.ceil(hasil_prediksi / 2) # 1 orang per 2 Ha per hari
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="rec-card">
                <h4 style="margin:0 0 10px 0; color:#10b981;">📦 Logistik</h4>
                <p style="margin:5px 0;">- Siapkan <b>{int(total_karung):,}</b> karung gabah.</p>
                <p style="margin:5px 0;">- Siapkan gudang/lantai jemur kapasitas <b>{total_ton:.1f} Ton</b>.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with c2:
        st.markdown(
            f"""
            <div class="rec-card rec-card-gold">
                <h4 style="margin:0 0 10px 0; color:#f59e0b;">👥 Tenaga & Alat</h4>
                <p style="margin:5px 0;">- Butuh estimasi <b>{int(butuh_buruh)}</b> tenaga angkut.</p>
                <p style="margin:5px 0;">- Hubungi penyedia <i>Combine Harvester</i> 2 minggu sebelum.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with c3:
        st.markdown(
            f"""
            <div class="rec-card rec-card-blue">
                <h4 style="margin:0 0 10px 0; color:#3b82f6;">💰 Strategi Harga</h4>
                <p style="margin:5px 0;">- Cek HPP Gabah terbaru.</p>
                <p style="margin:5px 0;">- Hindari jual langsung ke tengkulak saat panen raya.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ==========================================
# 8. FOOTER DATA RIWAYAT
# ==========================================
if st.checkbox("Lihat Data Riwayat 2025 (Transparansi Data)", key="chk_simple"):
    if data_history is not None:
        st.write(f"Menampilkan data historis untuk {selected_kab_simple}:")
        st.dataframe(data_history[data_history['Kabupaten'] == selected_kab_simple].head(10))
    else:
        st.warning("File data_siap_ml.csv tidak ditemukan.")

st.sidebar.markdown("---")
st.sidebar.caption("Jateng-Harvest v1.0 | Proyek Transformasi Digital")