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

# ==========================================
# PAGE CONFIGURATION & THEME STYLE
# ==========================================
st.set_page_config(
    page_title="Jateng Harvest Dashboard | Sistem Perencanaan Logistik & Estimasi Panen",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load background image dynamically
bg_img_path = os.path.join(os.path.dirname(__file__), "image.png")
bg_img_css = ""
if os.path.exists(bg_img_path):
    with open(bg_img_path, "rb") as f:
        encoded_bg = base64.b64encode(f.read()).decode()
    bg_img_css = f"""
    /* Background image with living agricultural warm glassmorphic overlay */
    .stApp {{
        background: linear-gradient(rgba(10, 25, 15, 0.82), rgba(15, 23, 42, 0.94)), url("data:image/png;base64,{encoded_bg}") !important;
        background-size: cover !important;
        background-position: center center !important;
        background-attachment: fixed !important;
        color: #f1f5f9;
    }}
    """
else:
    bg_img_css = """
    .stApp {{
        background: radial-gradient(circle at 50% 10%, #1e293b 0%, #0f172a 70%);
        color: #f1f5f9;
    }}
    """

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

def save_developer_log(kabupaten, kecamatan, luas_tanam, hasil_prediksi):
    """Fungsi penulisan otomatis log ke Google Sheets menggunakan Service Account Streamlit Secrets."""
    try:
        # Menghubungkan ke Google Sheets melalui GSheetsConnection
        conn = st.connection("gsheets", type=GSheetsConnection)
        
        # Membaca data yang sudah ada di worksheet "Sheet1" dengan parameter spreadsheet eksplisit
        try:
            existing_df = conn.read(spreadsheet=SPREADSHEET_URL, worksheet="Sheet1", ttl=0)
        except Exception:
            existing_df = pd.DataFrame()
            
        # Jika sheet masih kosong atau belum memiliki header yang tepat, inisialisasi kolom standar
        if existing_df.empty or "Waktu" not in existing_df.columns:
            existing_df = pd.DataFrame(columns=["Waktu", "Kabupaten", "Kecamatan", "Luas_Tanam", "Estimasi_Panen"])
        
        # Membangun baris data baru
        new_row = pd.DataFrame([{
            "Waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Kabupaten": str(kabupaten),
            "Kecamatan": str(kecamatan),
            "Luas_Tanam": float(luas_tanam),
            "Estimasi_Panen": float(hasil_prediksi)
        }])
        
        # Menambahkan baris baru di bawah data yang sudah ada
        updated_df = pd.concat([existing_df, new_row], ignore_index=True)
        
        # Memperbarui spreadsheet dengan parameter spreadsheet eksplisit
        conn.update(spreadsheet=SPREADSHEET_URL, worksheet="Sheet1", data=updated_df)
    except Exception as e:
        # Menampilkan pesan error dan petunjuk otentikasi Editor secara sangat jelas
        st.error(f"Gagal mengirim data ke Google Sheets: {e}")
        st.warning(
            "⚠️ **Petunjuk Penting Otentikasi Google Sheets (Permission Denied):**\n\n"
            "Jika data gagal masuk, Anda **WAJIB** membuka file Google Sheets Anda (`jateng_proyek`), klik tombol **Share (Bagikan)** di pojok kanan atas, dan tambahkan email Service Account bot ini sebagai **Editor**:\n\n"
            "👉 `jateng-harvest-bot@jateng-harvest-monitoring.iam.gserviceaccount.com`"
        )

def log_anonymous_activity(kecamatan, kabupaten, luas_tanam, asumsi_prod, estimasi_ton):
    """Tugas 2 & Fitur 5: Anonymous Activity Logger & Telemetry ke Google Sheets (Tanpa Identitas Pribadi)."""
    # 1. Simpan ke CSV lokal
    try:
        file_exists = os.path.exists(TELEMETRY_FILE)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(TELEMETRY_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Kabupaten", "Kecamatan", "Luas_Tanam_Ha", "Asumsi_Prod_TonHa", "Estimasi_Total_Ton"])
            writer.writerow([timestamp, kabupaten, kecamatan, round(luas_tanam, 2), round(asumsi_prod, 2), round(estimasi_ton, 2)])
    except Exception:
        pass
        
    # 2. Kirim ke Google Sheets via Web App URL
    try:
        payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "kabupaten": kabupaten,
            "kecamatan": kecamatan,
            "luas_tanam": round(luas_tanam, 2),
            "estimasi_ton": round(estimasi_ton, 2)
        }
        data = urllib.parse.urlencode(payload).encode('utf-8')
        req = urllib.request.Request(GOOGLE_SHEETS_API_URL, data=data)
        urllib.request.urlopen(req, timeout=3)
    except Exception:
        pass # fail silently
        
    # 3. Eksekusi koneksi GSheetsConnection resmi Streamlit Secrets
    save_developer_log(kabupaten, kecamatan, luas_tanam, estimasi_ton)

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

def transform_realtime_simotandi(uploaded_file):
    """
    Advanced ETL function to parse raw SIMOTANDI files with multi-level merged headers.
    - Uses vertical scanning to identify columns, bypassing messy horizontal rows.
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            df_uploaded = pd.read_csv(uploaded_file, header=None)
        else:
            df_uploaded = pd.read_excel(uploaded_file, header=None)
            
        # 1. Tentukan indeks kolom menggunakan Vertical Scanning (20 baris pertama)
        col_indices = {}
        # Gunakan array dari berbagai kemungkinan nama kolom (cukup salah satu match = valid)
        col_matchers = {
            'kabupaten': ['kabupaten'],
            'kecamatan': ['kecamatan', 'nama wilayah'], # Hapus 'wilayah' agar tidak tabrakan dengan 'Kode Wilayah'
            'tanam': ['tanam (1', 'tanam(1', 'tanam'],
            'veg1': ['vegetatif 1', 'vegetatif1', '13 -', '13-'],
            'veg2': ['vegetatif 2', 'vegetatif2', '37 -', '37-'],
            'gen1': ['generatif 1', 'generatif1', '61 -', '61-'],
            'gen2': ['generatif 2', 'generatif2', '85 -', '85-'],
            'panen': ['panen'] 
        }
        
        num_cols = len(df_uploaded.columns)
        for c in range(num_cols):
            # Gabungkan 20 baris teratas dari kolom ini menjadi satu string panjang
            col_header_text = ' '.join(df_uploaded.iloc[:20, c].astype(str)).lower().replace('\n', ' ')
            
            # SANITASI: Hapus kata-kata pengecoh dari judul besar/agregat (misal judul file di baris 1)
            # Ini mencegah keyword 'tanam' menyangkut di kata 'pertanaman', dan 'panen' di 'luas panen'
            col_header_text = col_header_text.replace('pertanaman', '').replace('luas panen', '').replace('luas baku sawah', '')
            
            for key, keywords in col_matchers.items():
                if key not in col_indices:
                    # Cek apakah ada satupun keyword yang cocok (ANY)
                    if any(kw in col_header_text for kw in keywords):
                        col_indices[key] = c
                        
        # Pastikan kolom utama minimal ditemukan
        if 'kecamatan' not in col_indices or 'tanam' not in col_indices:
            # Kembalikan dataframe utuh (fallback ke mode lama) jika parser ini gagal total
            df_uploaded.columns = df_uploaded.iloc[0]
            df_uploaded = df_uploaded.iloc[1:].reset_index(drop=True)
            df_uploaded.columns = df_uploaded.columns.astype(str).str.lower().str.replace('\n', ' ').str.strip()
            return df_uploaded
            
        # 2. Cari baris pertama yang berisi data numerik pada kolom 'tanam'
        tanam_col_idx = col_indices['tanam']
        data_start_idx = -1
        for i in range(25):
            try:
                val = str(df_uploaded.iloc[i, tanam_col_idx]).strip()
                if val.replace('.', '', 1).isdigit() and val.lower() != 'nan':
                    data_start_idx = i
                    break
            except Exception:
                pass
                    
        if data_start_idx == -1:
            data_start_idx = 5 # Asumsi standar jika tidak terdeteksi
            
        # 3. Potong dataframe hanya untuk area data
        df_data = df_uploaded.iloc[data_start_idx:].reset_index(drop=True)
        
        # 4. Bangun dataframe baru yang sudah terstruktur rapi dengan nama standar
        df_clean = pd.DataFrame()
        for key, c_idx in col_indices.items():
            df_clean[key] = df_data.iloc[:, c_idx]
            
        # DUPLIKASI CERDAS AREA: Jika file SIMOTANDI hanya punya 1 kolom wilayah, isi ke dua-duanya!
        # Namun sebelum itu, coba deteksi Kabupaten dari teks metadata di 20 baris pertama
        detected_kabupaten = None
        for i in range(min(20, len(df_uploaded))):
            row_str = ' '.join(df_uploaded.iloc[i].astype(str)).lower()
            for kab in le_kab.classes_:
                if kab.lower() in row_str:
                    detected_kabupaten = kab
                    break
            if detected_kabupaten:
                break
                
        if 'kabupaten' not in df_clean.columns:
            if detected_kabupaten:
                df_clean['kabupaten'] = detected_kabupaten
            elif 'kecamatan' in df_clean.columns:
                df_clean['kabupaten'] = df_clean['kecamatan']
            
        # Jika kabupaten masih tidak ada, isi otomatis dengan nilai default
        if 'kabupaten' not in df_clean.columns:
            df_clean['kabupaten'] = "Tidak Diketahui"
            
        # Buang baris spasi atau nan
        df_clean = df_clean.dropna(subset=['kecamatan'])
        
        # FILTER TOTAL PROVINSI: Buang baris "Jawa Tengah" agar tidak diprediksi sebagai kecamatan
        invalid_areas = ['jawa tengah', 'total', 'jumlah', 'provinsi']
        df_clean = df_clean[~df_clean['kecamatan'].astype(str).str.lower().str.strip().isin(invalid_areas)]
            
        return df_clean
        
    except Exception as e:
        st.error(f"Gagal memparsing file mentah: {e}")
        return None
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
if st.query_params.get("view") == "developer":
    st.markdown(
        """
        <div class="header-card" style="text-align: center; padding: 20px;">
            <h1 style="margin: 0; color: #10b981;">👨‍💻 Dashboard Monitoring Internal Developer</h1>
            <p style="margin: 10px 0 0 0; color: #cbd5e1;">Pemantauan live data aktivitas prediksi dari Google Sheets jateng_proyek.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
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
                <div class="glass-card" style="text-align: center; border-top: 4px solid #10b981 !important;">
                    <span style="font-size:1rem; color:#94a3b8; font-weight:600; text-transform:uppercase;">Total Hitungan Prediksi Petani</span>
                    <div style="font-size:3rem; color:#10b981; font-weight:800; margin:10px 0;">{total_predictions:,}</div>
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
                        font_color='#cbd5e1',
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
        <h2 style="margin:0; font-size:1.8rem; background: linear-gradient(135deg, #10b981 0%, #f59e0b 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            Jateng Harvest
        </h2>
        <p style="color:#64748b; font-size:0.85rem; font-weight:500; margin-top:5px;">Sistem Perencanaan & Perhitungan Panen</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("<hr style='border: 1px solid rgba(255,255,255,0.05); margin-top:0px;'>", unsafe_allow_html=True)

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
st.markdown(
    """
    <div class="header-card" style="text-align: center; padding: 30px 20px; border-radius: 16px; margin-bottom: 25px;">
        <span style="background-color: #10b981; color: #0f172a; padding: 6px 16px; border-radius: 20px; font-size: 0.85rem; font-weight: 800; text-transform: uppercase; letter-spacing: 1px;">
            Sugeng Rawuh! Sistem Peringatan Dini Panen
        </span>
        <h1 style="margin: 20px 0 10px 0; font-size: 2.4rem; letter-spacing: -0.5px; color: #ffffff;">
            Mau cek persiapan panen di kecamatan mana hari ini?
        </h1>
        <p style="margin: 0; color: #cbd5e1; font-size: 1.1rem; line-height: 1.6;">
            Aplikasi publik ini menerjemahkan data pertanian menjadi instruksi persiapan logistik nyata bagi petani agar tidak rugi.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Handle file upload processing if present
simontadi_data = None
if uploaded_file is not None:
    simontadi_data = transform_realtime_simotandi(uploaded_file)
    if simontadi_data is not None:
        st.success(f"✔️ Berkas '{uploaded_file.name}' berhasil diunggah dan dibersihkan! Sistem siap memproses batch prediksi.")

# Bagian 2: Area Input (Tengah Atas)
st.markdown("<div style='background: rgba(30, 41, 59, 0.6); padding: 25px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.05); margin-bottom: 30px;'>", unsafe_allow_html=True)
st.markdown("<h3 style='margin-top:0; color:#10b981;'>📍 Pilih Wilayah Lahan Anda</h3>", unsafe_allow_html=True)

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

st.markdown("<hr style='border: 1px solid rgba(255,255,255,0.05); margin: 20px 0;'>", unsafe_allow_html=True)
st.markdown("<h3 style='margin-top:0; color:#f59e0b;'>🌱 Kondisi Lahan Terkini</h3>", unsafe_allow_html=True)

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

st.markdown("</div>", unsafe_allow_html=True)

# MAIN TABS
tab_predict, tab_batch, tab_summary, tab_education = st.tabs([
    "🔮 Estimasi & Instruksi Pasca-Panen",
    "📊 Estimasi Massal Laporan Simontadi 2026",
    "📈 Ringkasan Data Tanam Jateng 2025",
    "💡 Edukasi Tren Harga & Waktu Jual"
])

# ------------------------------------------
# TAB 1: INDIVIDUAL PREDICTOR
# ------------------------------------------
with tab_predict:
    st.markdown(f"### 🌾 Perkiraan Panen: Kecamatan **{selected_kec}**, {selected_kab}")
    
    col_sum_info, col_btn_run = st.columns([3, 1])
    with col_sum_info:
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
    with col_btn_run:
        hitung_btn = st.button("🚀 Hitung Persiapan Panen", use_container_width=True)

    # Run calculation
    predictions = get_3_month_predictions(df_hist, selected_kab, selected_kec, selected_month, manual_luas_tanam)
    
    # Calculate aggregate production for upcoming 3 months
    total_est_ha = sum([p['Prediksi_Ha'] for p in predictions])
    total_est_ton = total_est_ha * productivity_rate
    total_karung = total_est_ton * 20
    total_buruh = math.ceil(total_est_ha / 2.0)
    
    # Log telemetry
    log_anonymous_activity(selected_kec, selected_kab, manual_luas_tanam, productivity_rate, total_est_ton)
    if hitung_btn:
        st.success("✔️ Data berhasil tercatat di laporan developer!")
    
    # Bagian 3: Visualisasi Hasil (Kartu Informasi / Metrics Card Kontras Tinggi)
    st.markdown("<h4 style='margin-top:25px; margin-bottom:15px;'>📊 Ringkasan Kebutuhan 3 Bulan Ke Depan</h4>", unsafe_allow_html=True)
    
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.markdown(
            f"""
            <div class="glass-card" style="border-top: 4px solid #f59e0b !important; text-align: center;">
                <span style="font-size:0.8rem; color:#f59e0b; font-weight:700; text-transform:uppercase;">ESTIMASI HASIL</span>
                <div style="font-size:2.2rem; color:#facc15; font-weight:800; margin:10px 0;">{total_est_ton:,.1f} Ton</div>
                <p style="font-size:0.85rem; color:#cbd5e1; margin:0;">Dari total {total_est_ha:,.1f} Hektar panen</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with mc2:
        st.markdown(
            f"""
            <div class="glass-card" style="border-top: 4px solid #10b981 !important; text-align: center;">
                <span style="font-size:0.8rem; color:#10b981; font-weight:700; text-transform:uppercase;">BUTUH KARUNG</span>
                <div style="font-size:2.2rem; color:#facc15; font-weight:800; margin:10px 0;">{int(total_karung):,} Lembar</div>
                <p style="font-size:0.85rem; color:#cbd5e1; margin:0;">Kapasitas karung 50 Kg</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with mc3:
        st.markdown(
            f"""
            <div class="glass-card" style="border-top: 4px solid #3b82f6 !important; text-align: center;">
                <span style="font-size:0.8rem; color:#3b82f6; font-weight:700; text-transform:uppercase;">BUTUH BURUH</span>
                <div style="font-size:2.2rem; color:#facc15; font-weight:800; margin:10px 0;">{total_buruh:,} Orang</div>
                <p style="font-size:0.85rem; color:#cbd5e1; margin:0;">Tenaga panen & angkut</p>
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
        line=dict(color='#facc15', width=4),
        marker=dict(size=12, color='#10b981'),
        mode='lines+markers+text',
        text=[f"{p:,.1f} Ha" for p in preds_plot],
        textposition="top center",
        name="Luas Panen"
    ))
    fig_line.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#cbd5e1',
        margin=dict(l=30, r=30, t=20, b=30),
        height=280,
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Hektar (Ha)", showgrid=True, gridcolor='rgba(255,255,255,0.05)')
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
                <div style="background: rgba(255,255,255,0.02); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
                    <p style="margin: 0 0 10px 0; font-size: 1.1rem; color: #10b981;"><b>Instruksi Kerja Bulan {pred['Bulan_Nama']}:</b></p>
                    <ul style="padding-left: 20px; color: #cbd5e1; line-height: 1.8;">
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
    st.markdown("<hr style='border: 1px solid rgba(255,255,255,0.05); margin-top:35px; margin-bottom:20px;'>", unsafe_allow_html=True)
    
    st.markdown("### 📥 Cetak Kartu Instruksi Anda")
    st.caption("Cetak rangkuman instruksi fisik untuk dibawa ke kelompok tani / penyedia alat mesin pertanian. Mobile-friendly!")
    
    pdf_data = generate_pdf_report(selected_kab, selected_kec, selected_month, manual_luas_tanam, productivity_rate, predictions)
    st.download_button(
        label="CETAK KARTU PERSIAPAN (PDF)",
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
    st.markdown("#### 📥 Contoh/Template Struktur CSV Simontadi Operasional (HST)")
    template_data = pd.DataFrame([
        {
            "Kabupaten": "Banjarnegara", "Kecamatan": "Banjarmangu", 
            "Tanam (1 - 12 HST)": 65.24, "Vegetatif 1 (13 - 36 HST)": 100.5, "Vegetatif 2 (37 - 60 HST)": 150.2,
            "Generatif 1 (61 - 84 HST)": 200.0, "Generatif 2 (85 - 120 HST)": 180.5, "Panen": 90.0
        },
        {
            "Kabupaten": "Banyumas", "Kecamatan": "Ajibarang", 
            "Tanam (1 - 12 HST)": 405.46, "Vegetatif 1 (13 - 36 HST)": 300.0, "Vegetatif 2 (37 - 60 HST)": 250.0,
            "Generatif 1 (61 - 84 HST)": 150.0, "Generatif 2 (85 - 120 HST)": 100.0, "Panen": 400.0
        }
    ])
    st.dataframe(template_data, use_container_width=True)
    
    # Process uploaded file
    if simontadi_data is not None:
        st.markdown("#### 🔍 Pratinjau Data Simontadi yang Diunggah")
        st.dataframe(simontadi_data.head(10), use_container_width=True)
        
        # Verifikasi Kolom Hasil ETL
        required_cols = [
            'kabupaten', 'kecamatan', 'tanam', 
            'veg1', 'veg2', 'gen1', 'gen2', 'panen'
        ]
        
        actual_cols = list(simontadi_data.columns)
        missing_cols = [c for c in required_cols if c not in actual_cols]
        
        if missing_cols:
            st.error(f"Kolom berkas tidak dapat dipetakan secara otomatis. Kategori yang hilang: {missing_cols}. Pastikan format SIMOTANDI dapat terbaca.")
        else:
            run_batch = st.button("🔮 Jalankan Proses Perhitungan Estimasi & Logistik Massal")
            
            if run_batch:
                with st.spinner("Menghitung estimasi luas panen dan rincian logistik untuk seluruh baris data..."):
                    results = []
                    
                    for index, row_data in simontadi_data.iterrows():
                        row_kab = str(row_data['kabupaten'])
                        row_kec = str(row_data['kecamatan'])
                        
                        # Pastikan tidak memproses baris yang bernilai nan
                        if row_kec.lower() == 'nan':
                            continue
                            
                        # ETL Feature Alignment
                        try:
                            tanam = float(str(row_data['tanam']).replace(',', '.'))
                            veg1 = float(str(row_data['veg1']).replace(',', '.'))
                            veg2 = float(str(row_data['veg2']).replace(',', '.'))
                            gen1 = float(str(row_data['gen1']).replace(',', '.'))
                            gen2 = float(str(row_data['gen2']).replace(',', '.'))
                            panen_val = float(str(row_data['panen']).replace(',', '.'))
                        except ValueError:
                            # Skip row if it has non-numeric data that can't be converted
                            continue
                        
                        fase_vegetatif_est = tanam + veg1 + veg2
                        fase_generatif_est = gen1 + gen2
                        fase_siap_panen_est = panen_val
                        luas_tanam = tanam + veg1
                        bulan_angka = datetime.now().month
                        
                        kab_enc = safe_transform(le_kab, row_kab)
                        kec_enc = safe_transform(le_kec, row_kec)
                        
                        feature_vector = [
                            kab_enc,
                            kec_enc,
                            bulan_angka,
                            luas_tanam,
                            fase_vegetatif_est,
                            fase_generatif_est,
                            fase_siap_panen_est
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
                            "Bulan Data (Ekstrak)": bulan_angka,
                            "Luas Tanam Aktual (Ha)": round(luas_tanam, 2),
                            "Estimasi Panen Selanjutnya (Ha)": round(pred_ha, 1),
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