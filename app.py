import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os
import math
import csv
from datetime import datetime
import urllib.request
import urllib.parse
import json
import base64
from PIL import Image

# ==========================================
# LOGO LOADING
# ==========================================
LOGO_FILENAME = "30 Mei 2026, 11.51.02.png"
logo_img = None

# Coba cari file gambar logo di beberapa lokasi (lokal/repositori)
possible_logo_paths = [
    LOGO_FILENAME,
    os.path.join(os.path.dirname(__file__), LOGO_FILENAME),
    r"D:\Mat.6\proyek transdig\STREAMLITPYTHON\30 Mei 2026, 11.51.02.png"
]

for path in possible_logo_paths:
    if os.path.exists(path):
        try:
            logo_img = Image.open(path)
            break
        except Exception:
            pass

# ==========================================
# PAGE CONFIGURATION & THEME STYLE
# ==========================================
st.set_page_config(
    page_title="Jateng Harvest",
    page_icon=logo_img if logo_img is not None else "🌾",
    layout="centered"
)

# Load background image dynamically
bg_img_path = os.path.join(os.path.dirname(__file__), "image.png")
bg_img_css = ""
if os.path.exists(bg_img_path):
    with open(bg_img_path, "rb") as f:
        encoded_bg = base64.b64encode(f.read()).decode()
    bg_img_css = f"""
    /* Background image with fresh, light living agricultural glassmorphic overlay */
    .stApp {{
        background: linear-gradient(rgba(27, 67, 50, 0.93), rgba(8, 28, 21, 0.96)), url("data:image/png;base64,{encoded_bg}") !important;
        background-size: cover !important;
        background-position: center center !important;
        background-attachment: fixed !important;
        color: #F8F9FA !important;
    }}
    """
else:
    bg_img_css = """
    .stApp {
        background: linear-gradient(135deg, #1B4332 0%, #081C15 100%) !important;
        color: #F8F9FA !important;
    }
    """

# Custom premium styling with Glassmorphism, tailored animations, and warm harvest/high contrast colors
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        color: #F8F9FA !important;
    }

    /* Sidebar Glassmorphism styling */
    [data-testid="stSidebar"] {
        background-color: rgba(27, 67, 50, 0.95) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] * {
        color: #F8F9FA !important;
    }
    
    /* Header card */
    .header-card {
        background: rgba(45, 106, 79, 0.45) !important;
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 30px;
        border-radius: 20px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
        backdrop-filter: blur(12px);
    }

    /* General Glassmorphism container - Hijau Lumut (#2D6A4F with glass opacity) */
    .glass-card {
        background: rgba(45, 106, 79, 0.5) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        padding: 24px;
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        color: #F8F9FA !important;
    }

    /* Recommendation card micro-animations */
    .rec-card {
        background: rgba(27, 67, 50, 0.6) !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 5px solid #FFB703 !important;
        padding: 16px 20px;
        border-radius: 12px;
        margin-bottom: 12px;
        transition: all 0.3s ease;
        color: #F8F9FA !important;
    }
    .rec-card:hover {
        transform: translateY(-4px);
        background: rgba(45, 106, 79, 0.8) !important;
        box-shadow: 0 10px 20px -5px rgba(0, 0, 0, 0.5);
    }
    
    .rec-card-gold {
        border-left-color: #FFB703 !important;
    }
    .rec-card-blue {
        border-left-color: #3b82f6 !important;
    }

    /* Stat values */
    .metric-val {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFB703 !important;
        margin-bottom: 4px;
    }
    .metric-val-gold {
        color: #FFB703 !important;
    }
    
    /* Buttons custom styles: Kuning Padi Matang (#FFB703) full width bold */
    .stButton>button {
        background: #FFB703 !important;
        color: #000000 !important;
        border: none !important;
        font-weight: 800 !important;
        font-size: 1.1rem !important;
        padding: 14px 28px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 15px -3px rgba(255, 183, 3, 0.4) !important;
        transition: all 0.2s ease-in-out !important;
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px -5px rgba(255, 183, 3, 0.6) !important;
        background: #ffa700 !important;
        color: #000000 !important;
    }
    .stButton>button:active {
        transform: translateY(1px) !important;
    }

    /* Download PDF Button Styling - Round White Button */
    div.stDownloadButton > button {
        background-color: #FFFFFF !important;
        color: #1B4332 !important;
        border: 2px solid #2D6A4F !important;
        font-weight: 800 !important;
        border-radius: 50px !important;
        padding: 12px 30px !important;
        box-shadow: 0 4px 15px rgba(255,255,255,0.2) !important;
        text-transform: uppercase;
        width: auto !important;
        margin: 0 auto !important;
        display: block !important;
    }
    div.stDownloadButton > button:hover {
        background-color: #F8F9FA !important;
        color: #1B4332 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(255,255,255,0.4) !important;
    }

    /* Subtext formatting */
    .sub-text {
        font-size: 0.85rem;
        color: #F8F9FA;
        opacity: 0.8;
    }

    /* Input label and options text color */
    label, [data-testid="stWidgetLabel"] p {
        color: #F8F9FA !important;
        font-weight: 700 !important;
    }
    
    /* Selectbox styling */
    .stSelectbox div[data-baseweb="select"] {
        background-color: rgba(27, 67, 50, 0.8) !important;
        color: #F8F9FA !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    .stSelectbox div[data-baseweb="select"] * {
        color: #F8F9FA !important;
    }

    div[role="listbox"] {
        background-color: #2D6A4F !important;
        color: #F8F9FA !important;
    }
    
    div[role="option"] {
        background-color: transparent !important;
        color: #F8F9FA !important;
    }
    
    div[role="option"]:hover {
        background-color: #1B4332 !important;
    }
</style>
"""
st.markdown(f"<style>{bg_img_css}</style>", unsafe_allow_html=True)
st.markdown(custom_css, unsafe_allow_html=True)

from fpdf import FPDF
import io
import csv
import math

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
# TELEMETRI, FEEDBACK, & KARTU PDF (TUGAS 1-3)
# ==========================================
from streamlit_gsheets import GSheetsConnection
TELEMETRY_FILE = "telemetri_penggunaan.csv"
FEEDBACK_FILE = "log_feedback_petani.csv"
GOOGLE_SHEETS_API_URL = "https://script.google.com/macros/s/15WhpXDecY5QJQDFEu_Uh4fsSLDr3fA7cinSTlLAu8f8/exec"

SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/15WhpXDecY5QJQDFEu_Uh4fsSLDr3fA7cinSTlLAu8f8/edit#gid=0"

def log_anonymous_activity(kecamatan, kabupaten, luas_tanam, asumsi_prod, estimasi_ton):
    """Background logging to Google Sheets with double try-except fallback to local CSV."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    success = False
    
    # First level: Try to send to Google Sheets
    try:
        # Menghubungkan ke Google Sheets melalui GSheetsConnection
        conn = st.connection("gsheets", type=GSheetsConnection)
        
        # Membaca data yang sudah ada di worksheet "Sheet1" secara langsung
        try:
            existing_df = conn.read(spreadsheet=SPREADSHEET_URL, worksheet="Sheet1", ttl=0)
        except Exception:
            existing_df = pd.DataFrame()
            
        # Jika sheet masih kosong atau belum memiliki header yang tepat, inisialisasi kolom standar
        if existing_df.empty or "Waktu" not in existing_df.columns:
            existing_df = pd.DataFrame(columns=["Waktu", "Kabupaten", "Kecamatan", "Luas_Tanam", "Estimasi_Panen"])
        
        # Membangun baris data baru
        new_row = pd.DataFrame([{
            "Waktu": timestamp,
            "Kabupaten": str(kabupaten),
            "Kecamatan": str(kecamatan),
            "Luas_Tanam": float(luas_tanam),
            "Estimasi_Panen": float(estimasi_ton)
        }])
        
        # Menambahkan baris baru di bawah data yang sudah ada
        updated_df = pd.concat([existing_df, new_row], ignore_index=True)
        
        # Memperbarui spreadsheet dengan parameter spreadsheet eksplisit
        conn.update(spreadsheet=SPREADSHEET_URL, worksheet="Sheet1", data=updated_df)
        success = True
    except Exception:
        # Fails, will proceed to local backup
        pass
        
    # Second level: If Google Sheets fails, backup to local CSV
    if not success:
        try:
            file_exists = os.path.exists(TELEMETRY_FILE)
            with open(TELEMETRY_FILE, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Waktu", "Kabupaten", "Kecamatan", "Luas_Tanam", "Estimasi_Panen"])
                writer.writerow([timestamp, kabupaten, kecamatan, round(luas_tanam, 2), round(estimasi_ton, 2)])
        except Exception:
            pass

def log_user_feedback(kecamatan, kabupaten, bulan_tanam, luas_tanam, akurat):
    """Tugas 3: Modul Feedback Inklusif untuk Validasi Model Masa Depan."""
    try:
        file_exists = os.path.exists(FEEDBACK_FILE)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(FEEDBACK_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Kabupaten", "Kecamatan", "Bulan_Tanam", "Luas_Tanam_Ha", "Sesuai"])
            writer.writerow([timestamp, kabupaten, kecamatan, bulan_tanam, round(luas_tanam, 2), "Ya" if akurat else "Tidak"])
    except Exception:
        pass

class HarvestPDF(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 16)
        self.cell(0, 10, "KARTU PERSIAPAN LOGISTIK PANEN", ln=True, align="C")
        self.set_font("helvetica", "", 10)
        self.cell(0, 6, "Sistem Informasi Manajemen Logistik Pertanian Jawa Tengah", ln=True, align="C")
        self.line(10, 28, 200, 28)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.cell(0, 10, f"Halaman {self.page_no()} | Dokumen dicetak otomatis (Tanpa Login & Privasi Terlindungi)", align="C")

def generate_pdf_report(kabupaten, kecamatan, bulan_tanam, luas_tanam, as_prod, pred_results):
    """Tugas 1: Implementasi Fitur Cetak Laporan PDF (User Empowerment)."""
    pdf = HarvestPDF()
    pdf.add_page()
    
    # Rincian Input
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, "Rincian Lokasi & Input Lahan:", ln=True)
    pdf.set_font("helvetica", "", 11)
    pdf.cell(0, 6, f"Kabupaten: {kabupaten} | Kecamatan: {kecamatan}", ln=True)
    pdf.cell(0, 6, f"Bulan Tanam Terakhir: {bulan_tanam} | Luas Tanam: {luas_tanam:,.1f} Ha", ln=True)
    pdf.cell(0, 6, f"Asumsi Produktivitas: {as_prod:.1f} Ton/Ha", ln=True)
    pdf.ln(8)
    
    # Hasil Perhitungan & Rencana Logistik
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(0, 8, "Rencana Preskriptif Logistik 3 Bulan ke Depan:", ln=True)
    pdf.ln(4)
    
    for idx, pred in enumerate(pred_results):
        pdf.set_font("helvetica", "B", 11)
        pdf.cell(0, 6, f"Bulan Ke-{idx+1}: {pred['Bulan_Nama']} (Estimasi Panen: {pred['Prediksi_Ha']:,.1f} Ha)", ln=True)
        
        # Hitung logistik
        ha = pred['Prediksi_Ha']
        tons = ha * as_prod
        sacks = math.ceil(tons * 20)
        harvesters = math.ceil(ha / 15.0)
        drying = tons * 12
        storage = tons * 1.8
        
        pdf.set_font("helvetica", "", 10)
        pdf.cell(0, 5, f" - Segera pesan {sacks:,} lembar karung gabah (kapasitas 50 kg).", ln=True)
        pdf.cell(0, 5, f" - Siapkan {harvesters} unit combine harvester (untuk panen raya dalam 10 hari).", ln=True)
        pdf.cell(0, 5, f" - Luas lantai jemur minimal: {drying:,.0f} m² | Volume gudang: {storage:,.0f} m³.", ln=True)
        pdf.ln(6)
        
    return bytes(pdf.output())

# ==========================================
# LOGICAL HELPER FUNCTIONS (AUTOMATION CORE)
# ==========================================
def safe_transform(encoder, label, default_val=-1):
    """Safely encodes categorical labels with robust string cleaning."""
    try:
        clean_label = str(label).strip().title()
        if clean_label in encoder.classes_:
            return encoder.transform([clean_label])[0]
        else:
            return default_val
    except Exception:
        return default_val

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
# DEVELOPER DASHBOARD (SECRET VIEW)
# ==========================================
if st.query_params.get("view") == "developer-access":
    st.markdown(
        """
        <div class="header-card" style="text-align: center; padding: 20px;">
            <h1 style="margin: 0; color: #FFB703 !important;">👨‍💻 Autentikasi Internal Developer</h1>
            <p style="margin: 10px 0 0 0; color: #F8F9FA;">Akses terbatas. Silakan masukkan password/token pengembang.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Password Form
    col_auth_left, col_auth_right = st.columns([1, 2])
    with col_auth_left:
        password_input = st.text_input("Masukkan Token/Password:", type="password", key="dev_access_token")
    
    correct_password = "jateng_admin_2026"
    try:
        if "DEVELOPER_PASSWORD" in st.secrets:
            correct_password = st.secrets["DEVELOPER_PASSWORD"]
    except Exception:
        pass
        
    if not password_input:
        st.info("⚠️ Silakan isi password untuk melihat Dashboard Monitoring.")
        st.stop()
    elif password_input != correct_password:
        st.error("❌ Password salah! Akses ditolak.")
        st.stop()
        
    st.success("🔓 Akses Diberikan. Memuat data telemetri...")
    st.markdown("---")
    
    try:
        # Menghubungkan ke Google Sheets melalui GSheetsConnection
        conn = st.connection("gsheets", type=GSheetsConnection)
        
        # Membaca data yang sudah ada di worksheet "Sheet1" secara langsung
        dev_df = conn.read(spreadsheet=SPREADSHEET_URL, worksheet="Sheet1", ttl=0)
        
        if not dev_df.empty and "Waktu" in dev_df.columns:
            # Statistik total hitungan prediksi
            total_predictions = len(dev_df)
            st.markdown(
                f"""
                <div class="glass-card" style="text-align: center; border-top: 4px solid #FFB703 !important;">
                    <span style="font-size:1rem; color:#FFB703; font-weight:600; text-transform:uppercase;">Total Hitungan Prediksi Petani</span>
                    <div style="font-size:3rem; color:#F8F9FA; font-weight:800; margin:10px 0;">{total_predictions:,}</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            col_d1, col_d2 = st.columns([1, 2])
            
            with col_d1:
                st.markdown("### 📊 Top Kecamatan Akses Terbanyak")
                if "Kecamatan" in dev_df.columns:
                    top_kec = dev_df["Kecamatan"].value_counts().reset_index()
                    top_kec.columns = ["Kecamatan", "Jumlah_Akses"]
                    # Limit to top 10 for better visualization
                    top_kec = top_kec.head(10)
                    
                    fig_kec = px.bar(
                        top_kec, x="Jumlah_Akses", y="Kecamatan", orientation='h',
                        color="Jumlah_Akses", color_continuous_scale="Viridis"
                    )
                    fig_kec.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#F8F9FA',
                        yaxis={'categoryorder': 'total ascending'},
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=350
                    )
                    st.plotly_chart(fig_kec, use_container_width=True)
                else:
                    st.info("Kolom 'Kecamatan' belum tersedia di Google Sheets.")
                    
            with col_d2:
                col_title, col_btn = st.columns([2, 1])
                with col_title:
                    st.markdown("### 📋 Riwayat Telemetri Live")
                with col_btn:
                    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
                    st.link_button("🔗 Buka Google Sheets", SPREADSHEET_URL, use_container_width=True)
                # Show latest data first
                st.dataframe(dev_df.sort_values(by="Waktu", ascending=False), use_container_width=True, height=400)
                
        else:
            st.info("Koneksi Google Sheets berhasil, namun data aktivitas masih kosong.")
            
    except Exception as e:
        st.error(f"Gagal mengambil data dari Google Sheets API: {e}")
        st.warning("⚠️ Pastikan Google Sheets tidak sedang limit dan email Service Account sudah diset sebagai Editor.")

    # Menghentikan eksekusi kode di sini agar tampilan publik (Petani) tidak ikut di-render
    st.stop()

# ==========================================
# NAVIGATION & SIDEBAR MANAGEMENT
# ==========================================
st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 10px 0;">
        <h2 style="margin:0; font-size:1.8rem; color:#FFB703 !important;">
            Jateng Harvest
        </h2>
        <p style="color:#F8F9FA; font-size:0.85rem; font-weight:500; margin-top:5px; opacity:0.8;">Sistem Perencanaan & Perhitungan Panen</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("<hr style='border: 1px solid rgba(255,255,255,0.1); margin-top:0px;'>", unsafe_allow_html=True)

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
    <div style="background: rgba(45, 106, 79, 0.45); padding:15px; border-radius:12px; border: 1px solid rgba(255,255,255,0.1); margin-top: 15px;">
        <span style="font-size:0.75rem; color:#FFB703; font-weight:600; text-transform:uppercase; letter-spacing:1px;">Prinsip Aplikasi</span>
        <p style="font-size:0.8rem; color:#F8F9FA; margin: 5px 0 0 0; line-height: 1.4;">
            <b>Simontadi:</b> "Apa yang terjadi di lahan"<br>
            <b>Jateng Harvest:</b> "Apa yang harus Anda siapkan"
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================
# MAIN PAGE ROUTING & DASHBOARD BODY
# ==========================================

# Bagian 1: Header & Sambutan (Paling Atas)
col_logo_l, col_logo_m, col_logo_r = st.columns([1, 2, 1])
with col_logo_m:
    if logo_img is not None:
        st.image(logo_img, use_container_width=True)

st.markdown(
    """
    <div class="header-card" style="text-align: center; padding: 30px 20px; border-radius: 16px; margin-bottom: 25px;">
        <span style="background-color: #FFB703; color: #000000; padding: 6px 16px; border-radius: 20px; font-size: 0.85rem; font-weight: 800; text-transform: uppercase; letter-spacing: 1px;">
            Sugeng Rawuh Wonten ing Jateng Harvest! 🌾
        </span>
        <h1 style="margin: 20px 0 10px 0; font-size: 2.2rem; letter-spacing: -0.5px; color: #F8F9FA !important;">
            Jateng Harvest Dashboard
        </h1>
        <p style="margin: 0; color: #F8F9FA; font-size: 1.1rem; line-height: 1.6; opacity: 0.9;">
            Badhe ngecek persiapan panen teng kecamatan pundi dinten niki?<br>
            Aplikasi niki mbantu panjenengan ngitung kabutuhan logistik panen (karung, tenaga buruh, lan mesin) supados asil panenipun sae lan mboten rugi.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Bagian 2: Area Input (Tengah Atas)
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.markdown("<h3 style='margin-top:0; color:#FFB703;'>📍 Pilih Wilayah Lahan Anda</h3>", unsafe_allow_html=True)

if not df_hist.empty:
    list_kabupaten = sorted(df_hist['Kabupaten'].unique())
else:
    list_kabupaten = sorted(le_kab.classes_)

col_loc1, col_loc2 = st.columns(2)
with col_loc1:
    selected_kab = st.selectbox("Kabupaten:", list_kabupaten)

if not df_hist.empty:
    list_kecamatan = sorted(df_hist[df_hist['Kabupaten'] == selected_kab]['Kecamatan'].unique())
else:
    list_kecamatan = [k for k in sorted(le_kec.classes_) if k in df_hist['Kecamatan'].values]

with col_loc2:
    selected_kec = st.selectbox("Kecamatan:", list_kecamatan)

st.markdown("<hr style='border: 1px solid rgba(255,255,255,0.15); margin: 20px 0;'>", unsafe_allow_html=True)
st.markdown("<h3 style='margin-top:0; color:#FFB703;'>🌱 Kondisi Lahan Terkini</h3>", unsafe_allow_html=True)

col_inp1, col_inp2 = st.columns(2)
with col_inp1:
    selected_month = st.selectbox("Bulan Tanam Terakhir:", MONTHS_ID, index=4)

default_luas_tanam = 100.0
if not df_hist.empty:
    baseline = get_historical_luas_tanam(df_hist, selected_kab, selected_kec, MONTH_MAP_ID_TO_NUM[selected_month])
    if baseline > 0:
        default_luas_tanam = round(baseline, 2)

with col_inp2:
    manual_luas_tanam = st.number_input("Ketik luas lahan Anda (Hektar):", min_value=0.0, max_value=15000.0, value=default_luas_tanam, step=10.0)

hitung_btn = st.button("🚀 HITUNG ESTIMASI", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Run calculation globally
predictions = get_3_month_predictions(df_hist, selected_kab, selected_kec, selected_month, manual_luas_tanam)

# Calculate aggregate production for upcoming 3 months
total_est_ha = sum([p['Prediksi_Ha'] for p in predictions])
total_est_ton = total_est_ha * productivity_rate
total_karung = total_est_ton * 20
total_buruh = math.ceil(total_est_ha / 2.0)

# Log telemetry only when button is clicked
if hitung_btn:
    log_anonymous_activity(selected_kec, selected_kab, manual_luas_tanam, productivity_rate, total_est_ton)

# MAIN TABS
tab_predict, tab_community, tab_education = st.tabs([
    "🔮 Estimasi & Instruksi Pasca-Panen (Individu)",
    "📊 Pantau Wilayah Kecamatan (Tren Komunitas)",
    "💡 Edukasi Tren Harga & Literasi Finansial"
])

# ------------------------------------------
# TAB 1: INDIVIDUAL PREDICTOR
# ------------------------------------------
with tab_predict:
    st.markdown(f"### 🌾 Perkiraan Panen: Kecamatan **{selected_kec}**, {selected_kab}")
    
    st.markdown(
        f"""
        <div class="glass-card" style="text-align: center; margin-bottom: 25px;">
            📅 <b>Bulan Input:</b> {selected_month} &nbsp;|&nbsp; 
            🌱 <b>Luas Tanam:</b> {manual_luas_tanam:,.1f} Ha &nbsp;|&nbsp;
            ⚙️ <b>Asumsi Produktivitas:</b> {productivity_rate:.1f} Ton/Ha
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Bagian 3: Visualisasi Hasil (Kartu Informasi / Metrics Card Kontras Tinggi)
    st.markdown("<h4 style='margin-top:25px; margin-bottom:15px;'>📊 Ringkasan Kebutuhan 3 Bulan Ke Depan</h4>", unsafe_allow_html=True)
    
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown(
            f"""
            <div class="glass-card" style="border-top: 4px solid #FFB703 !important; text-align: center;">
                <span style="font-size:0.8rem; color:#FFB703; font-weight:700; text-transform:uppercase;">ESTIMASI HASIL</span>
                <div style="font-size:2.2rem; color:#FFB703; font-weight:800; margin:10px 0;">{total_est_ton:,.1f} Ton</div>
                <p style="font-size:0.85rem; color:#F8F9FA; margin:0; opacity: 0.9;">Dari total {total_est_ha:,.1f} Hektar panen</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with mc2:
        st.markdown(
            f"""
            <div class="glass-card" style="border-top: 4px solid #FFB703 !important; text-align: center;">
                <span style="font-size:0.8rem; color:#FFB703; font-weight:700; text-transform:uppercase;">BUTUH KARUNG</span>
                <div style="font-size:2.2rem; color:#F8F9FA; font-weight:800; margin:10px 0;">{int(total_karung):,} Lembar</div>
                <p style="font-size:0.85rem; color:#F8F9FA; margin:0; opacity: 0.9;">Kapasitas karung 50 Kg</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with mc3:
        st.markdown(
            f"""
            <div class="glass-card" style="border-top: 4px solid #FFB703 !important; text-align: center;">
                <span style="font-size:0.8rem; color:#FFB703; font-weight:700; text-transform:uppercase;">BUTUH BURUH</span>
                <div style="font-size:2.2rem; color:#F8F9FA; font-weight:800; margin:10px 0;">{total_buruh:,} Orang</div>
                <p style="font-size:0.85rem; color:#F8F9FA; margin:0; opacity: 0.9;">Tenaga panen & angkut</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Bagian 4: Grafik Tren (Bawah Hasil)
    st.markdown("<h4 style='margin-top:30px; margin-bottom:15px;'>📈 Grafik Garis Puncak Panen</h4>", unsafe_allow_html=True)
    
    months_plot = [p['Bulan_Nama'] for p in predictions]
    preds_plot = [p['Prediksi_Ha'] for p in predictions]
    
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=months_plot,
        y=preds_plot,
        line=dict(color='#FFB703', width=4),
        marker=dict(size=12, color='#FFB703'),
        mode='lines+markers+text',
        text=[f"{p:,.1f} Ha" for p in preds_plot],
        textfont=dict(color='#F8F9FA'),
        textposition="top center",
        name="Luas Panen"
    ))
    fig_line.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#F8F9FA',
        margin=dict(l=30, r=30, t=20, b=30),
        height=280,
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Hektar (Ha)", showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # Rincian per bulan
    st.markdown("#### 📅 Rincian Persiapan Tiap Bulan")
    inst_tabs = st.tabs([f"Bulan {p['Bulan_Nama']}" for p in predictions])
    for idx, pred in enumerate(predictions):
        with inst_tabs[idx]:
            ha = pred['Prediksi_Ha']
            tons = ha * productivity_rate
            sacks = math.ceil(tons * 20)
            harv = math.ceil(ha / 15.0)
            dry = tons * 12
            
            st.markdown(
                f"""
                <div style="background: rgba(45, 106, 79, 0.4); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); color: #F8F9FA;">
                    <p style="margin: 0 0 10px 0; font-size: 1.1rem; color: #FFB703;"><b>Instruksi Kerja Bulan {pred['Bulan_Nama']}:</b></p>
                    <ul style="padding-left: 20px; color: #F8F9FA; line-height: 1.8; opacity: 0.9;">
                        <li>Perkiraan panen seluas <b>{ha:,.1f} Hektar</b> (sekitar <b>{tons:,.1f} Ton</b> gabah basah).</li>
                        <li><b>Karung:</b> Pesan <b>{sacks:,}</b> lembar karung dari sekarang.</li>
                        <li><b>Alat Mesin:</b> Hubungi penyedia sewa untuk <b>{harv}</b> unit mesin <i>Combine Harvester</i>.</li>
                        <li><b>Tempat Jemur:</b> Siapkan area jemur minimal seluas <b>{dry:,.0f} m²</b>.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

    # Bagian 5: Tombol Aksi (Paling Bawah)
    st.markdown("<hr style='border: 1px solid rgba(255,255,255,0.1); margin-top:35px; margin-bottom:20px;'>", unsafe_allow_html=True)
    
    st.markdown("### 📥 Cetak Kartu Instruksi Anda")
    st.caption("Cetak rangkuman instruksi fisik untuk dibawa ke kelompok tani / penyedia alat mesin pertanian. Mobile-friendly!")
    
    pdf_data = generate_pdf_report(selected_kab, selected_kec, selected_month, manual_luas_tanam, productivity_rate, predictions)
    st.download_button(
        label="Download PDF Kartu Persiapan",
        data=pdf_data,
        file_name=f"Kartu_Persiapan_Panen_{selected_kec}_{selected_kab}.pdf",
        mime="application/pdf",
        key="btn_pdf_download"
    )
    
    st.markdown("#### 💬 Masukan Anda Sangat Berarti")
    st.caption("Apakah hasil perkiraan ini sesuai dengan kondisi lahan di daerah Anda?")
    feed_col1, feed_col2 = st.columns(2)
    with feed_col1:
        if st.button("✔️ Ya, Sesuai", key="btn_feed_yes", use_container_width=True):
            log_user_feedback(selected_kec, selected_kab, selected_month, manual_luas_tanam, True)
            st.success("Maturnuwun atas masukan Anda!")
    with feed_col2:
        if st.button("❌ Tidak Sesuai", key="btn_feed_no", use_container_width=True):
            log_user_feedback(selected_kec, selected_kab, selected_month, manual_luas_tanam, False)
            st.success("Maturnuwun! Masukan Anda membantu kami menyempurnakan perhitungan.")

# ------------------------------------------
# TAB 2: COMMUNITY MONITOR (AUTOMATIC HISTORICAL DATA DETECTOR)
# ------------------------------------------
with tab_community:
    st.markdown(f"### 📊 Pola Tanam & Prediksi Panen Wilayah Kecamatan **{selected_kec}**")
    st.markdown(
        """
        Gunakan tab ini untuk melihat ringkasan pola tanam historis dan estimasi panen komunitas di wilayah kecamatan Anda.
        Sistem membaca data internal secara otomatis tanpa memerlukan unggah data manual.
        """
    )
    
    # Filter dataset for selected area
    df_kec = df_hist[
        (df_hist['Kabupaten'].str.lower() == selected_kab.lower()) & 
        (df_hist['Kecamatan'].str.lower() == selected_kec.lower())
    ]
    
    if not df_kec.empty:
        # High level stats
        tot_luas_tanam = df_kec['Luas_Tanam'].sum()
        tot_luas_panen = df_kec['Luas_Panen'].sum()
        avg_fase_siap = df_kec['Fase_Siap_Panen_Est'].mean()
        
        c_stats_1, c_stats_2, c_stats_3 = st.columns(3)
        with c_stats_1:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align: center; border-left: 5px solid #FFB703 !important;">
                    <span style="font-size:0.8rem; color:#FFB703; font-weight:700; text-transform:uppercase;">TOTAL LUAS TANAM WILAYAH</span>
                    <div class="metric-val" style="margin:10px 0 0 0;">{tot_luas_tanam:,.1f} Ha</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c_stats_2:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align: center; border-left: 5px solid #FFB703 !important;">
                    <span style="font-size:0.8rem; color:#FFB703; font-weight:700; text-transform:uppercase;">TOTAL LUAS PANEN WILAYAH</span>
                    <div class="metric-val" style="margin:10px 0 0 0;">{tot_luas_panen:,.1f} Ha</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with c_stats_3:
            st.markdown(
                f"""
                <div class="glass-card" style="text-align: center; border-left: 5px solid #FFB703 !important;">
                    <span style="font-size:0.8rem; color:#FFB703; font-weight:700; text-transform:uppercase;">RATA-RATA LAHAN SIAP PANEN</span>
                    <div class="metric-val" style="margin:10px 0 0 0;">{avg_fase_siap:,.1f} Ha</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        # Monthly Seasonal Cycle of Planting vs Harvesting for Kecamatan
        st.markdown("#### 📅 Pola Siklus Bulanan Kecamatan: Masa Tanam vs Masa Panen")
        df_kec_monthly = df_kec.groupby('Bulan_Angka')[['Luas_Tanam', 'Luas_Panen']].mean().reset_index()
        df_kec_monthly['Bulan'] = df_kec_monthly['Bulan_Angka'].map(MONTH_MAP_NUM_TO_ID)
        df_kec_monthly = df_kec_monthly.sort_values('Bulan_Angka')
        
        fig_cycle = go.Figure()
        fig_cycle.add_trace(go.Scatter(
            x=df_kec_monthly['Bulan'], y=df_kec_monthly['Luas_Tanam'],
            mode='lines+markers', name='Luas Tanam (🌱)',
            line=dict(color='#FFB703', width=3)
        ))
        fig_cycle.add_trace(go.Scatter(
            x=df_kec_monthly['Bulan'], y=df_kec_monthly['Luas_Panen'],
            mode='lines+markers', name='Luas Panen (🌾)',
            line=dict(color='#FFFFFF', width=3, dash='dash')
        ))
        fig_cycle.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#F8F9FA',
            margin=dict(l=20, r=20, t=20, b=20),
            height=300,
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_cycle, use_container_width=True)
        
        # Details table
        st.markdown("#### 📋 Rincian Data Historis Kecamatan")
        df_kec_disp = df_kec.sort_values(by="Bulan_Angka")
        st.dataframe(
            df_kec_disp[['Bulan', 'Luas_Tanam', 'Luas_Panen', 'Fase_Vegetatif_Est', 'Fase_Generatif_Est', 'Fase_Siap_Panen_Est']],
            use_container_width=True
        )
    else:
        st.warning(f"Data historis kecamatan {selected_kec} belum tersedia di database internal.")
        # Fallback to Kabupaten average
        df_kab = df_hist[df_hist['Kabupaten'].str.lower() == selected_kab.lower()]
        if not df_kab.empty:
            st.info(f"Menampilkan ringkasan rata-rata untuk Kabupaten {selected_kab} sebagai referensi:")
            df_kab_monthly = df_kab.groupby('Bulan_Angka')[['Luas_Tanam', 'Luas_Panen']].mean().reset_index()
            df_kab_monthly['Bulan'] = df_kab_monthly['Bulan_Angka'].map(MONTH_MAP_NUM_TO_ID)
            df_kab_monthly = df_kab_monthly.sort_values('Bulan_Angka')
            
            fig_kab = go.Figure()
            fig_kab.add_trace(go.Scatter(
                x=df_kab_monthly['Bulan'], y=df_kab_monthly['Luas_Tanam'],
                mode='lines+markers', name='Luas Tanam Rata-rata (🌱)',
                line=dict(color='#FFB703', width=3)
            ))
            fig_kab.add_trace(go.Scatter(
                x=df_kab_monthly['Bulan'], y=df_kab_monthly['Luas_Panen'],
                mode='lines+markers', name='Luas Panen Rata-rata (🌾)',
                line=dict(color='#FFFFFF', width=3, dash='dash')
            ))
            fig_kab.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#F8F9FA',
                margin=dict(l=20, r=20, t=20, b=20),
                height=300,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_kab, use_container_width=True)

# ------------------------------------------
# TAB 3: EDUCATIONAL MODULE (GRAIN PRICES & SELLING STRATEGY)
# ------------------------------------------
with tab_education:
    st.markdown("### 💡 Edukasi Strategi Penjualan Gabah & Waktu Panen")
    st.markdown(
        """
        Sering kali kita terpaksa menjual gabah dengan harga sangat murah karena berbarengan dengan panen raya di mana-mana. 
        **Jateng Harvest** menyediakan modul edukasi sederhana mengenai siklus tren harga gabah daerah, agar Bapak/Ibu sekalian bisa merencanakan waktu penjualan terbaik dengan harga yang pas.
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
            <div class="glass-card" style="border-left: 5px solid #FFB703 !important;">
                <h4 style="color:#FFB703 !important; margin-top:0; font-size:1.1rem;">💡 Tips Penting Penjualan Gabah:</h4>
                <ul style="padding-left:20px; font-size:0.9rem; color:#F8F9FA; line-height:1.7; opacity: 0.9;">
                    <li><b>Jangan Langsung Jual Semua saat Panen Raya (Maret - Mei):</b> Biasanya pas panen raya, harga gabah basah (GKP) bakal turun drastis karena stok melimpah. Jika ada kebutuhan mendesak, jual sebagian saja dahulu, sisanya disimpan.</li>
                    <li><b>Manfaatkan Lantai Jemur (Dikeringkan Dulu):</b> Gabah yang dikeringkan sampai kadar air sekitar 14% (menjadi Gabah Kering Giling/GKG) harganya bisa naik 25% hingga 30% lebih tinggi dibanding langsung dijual basah.</li>
                    <li><b>Incar "Waktu Emas" Penjualan (November - Januari):</b> Di bulan-bulan ini, stok gabah di pasar sangat sedikit karena petani baru mulai menanam. Harga gabah biasanya melonjak tinggi. Ini waktu terbaik untuk menjual simpanan gabah Anda.</li>
                    <li><b>Simpan dengan Baik agar Aman:</b> Pastikan gudang penyimpanan Bapak/Ibu kering dan bebas dari lembap. Gunakan panduan area jemur di Tab 1 agar gabah tidak berjamur dan kualitasnya tetap terjaga.</li>
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
            marker_color='rgba(255, 183, 3, 0.3)',
            yaxis='y2'
        ))
        
        # Add line chart representing price per kg
        fig_edu.add_trace(go.Scatter(
            x=prices_data['Bulan'], y=prices_data['Harga_GKP_PerKg'],
            mode='lines+markers+text',
            name="Rata-rata Harga Gabah (IDR/Kg)",
            line=dict(color='#FFB703', width=3),
            marker=dict(size=8, color='#FFB703'),
            text=[f"Rp{val:,}" for val in prices_data['Harga_GKP_PerKg']],
            textfont=dict(color='#F8F9FA'),
            textposition="top center"
        ))
        
        fig_edu.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#F8F9FA',
            margin=dict(l=20, r=20, t=10, b=20),
            height=300,
            xaxis=dict(showgrid=False),
            yaxis=dict(title="Harga Gabah (IDR/Kg)", showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
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