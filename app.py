# =============================================================================
# app.py — Flask REST API Sunucusu
# =============================================================================
# GÖREV   : Eğitilmiş modelleri ve checkpoint'i yükleyerek HTTP üzerinden
#           fenomen önerisi sunan REST API'yi ayağa kaldırır.
#
# ÇALIŞMA : python app.py  →  http://localhost:5000
#
# ENDPOINT'LER:
#   POST /recommend              → Marka metnine göre fenomen öner
#   GET  /influencers            → Tüm fenomen listesi (filtreli)
#   GET  /influencers/<n>/similar→ Benzer fenomenler (K-Means cluster)
#   GET  /campaigns              → 10 kampanya tanımı
#   GET  /stats                  → Sistem istatistikleri
#   GET  /                       → Frontend arayüzü
#
# BAĞIMLILIKLAR (pipeline/ dizininden yüklenir):
#   influencer_summary_checkpoint.pkl  — 244 fenomenin önceden hesaplanmış skorları
#   best_model_xgb.pkl / lgbm.pkl      — XGBoost & LightGBM sınıflandırıcıları
#   label_encoder.pkl / feature_columns.pkl
#
# SKOR FORMÜLLERİ:
#   NFS   = Ridge(engagement_rate, FGR, posts_per_month) → eng_auth etiketi (pipeline/nfs_scoring.py)
#   SFS   = cosine_sim(SBERT(marka), SBERT(fenomen)) × 100
#   BAS   = SFS×0.35 + NFS×0.30 + positive_ratio×0.25 + (100-fake_risk)×0.10
#   FINAL = SFS×0.35 + NFS×0.25 + CFS×0.20 + positive_ratio×0.10 + (100-fake_risk)×0.10
# =============================================================================

# =============================================================================
# app.py — Flask REST API Sunucusu
# Notebook ile uyumlu versiyon — 6 Türkçe kampanya, views bazlı metrikler
# =============================================================================

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pickle
import re
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

try:
    from db.mongo_client import get_collection, is_mongo_available
except ImportError:
    def get_collection(): return None
    def is_mongo_available(): return False

# ── Uygulama ayarları ────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="frontend/static", template_folder="frontend")
CORS(app)

BASE_DIR     = Path(__file__).parent
PIPELINE_DIR = BASE_DIR / "pipeline"

# ── Kampanya sabitleri — notebook ile birebir aynı ───────────────────────────
CAMPAIGN_NAMES = [
    "spor_kampanyasi",
    "moda_kampanyasi",
    "teknoloji_kampanyasi",
    "yemek_kampanyasi",
    "annebebek_kampanyasi",
    "oyun_kampanyasi",
]

CAMPAIGN_TEXTS = {
    "spor_kampanyasi": (
        "Spor giyim markasıyız. Fitness, spor salonu, koşu, yoga ve aktif yaşam tarzı "
        "içerikleri üreten fenomenlerle çalışmak istiyoruz. Antrenman rutinleri, spor "
        "beslenme önerileri, motivasyon içerikleri ve sağlıklı yaşam paylaşımları "
        "yapan içerik üreticileri arıyoruz."
    ),
    "moda_kampanyasi": (
        "Moda ve güzellik markasıyız. Stil önerileri, kombin paylaşımları, makyaj "
        "ve güzellik içerikleri üreten fenomenlerle çalışmak istiyoruz. Sokak modası, "
        "trend takibi, aksesuar ve kıyafet incelemeleri yapan içerik üreticileri arıyoruz."
    ),
    "teknoloji_kampanyasi": (
        "Teknoloji ve elektronik markasıyız. Cihaz incelemeleri, yazılım geliştirme, "
        "yapay zeka, kodlama ve dijital inovasyon içerikleri üreten fenomenlerle "
        "çalışmak istiyoruz. Ürün karşılaştırmaları ve teknoloji haberleri paylaşan "
        "içerik üreticileri arıyoruz."
    ),
    "yemek_kampanyasi": (
        "Gıda ve mutfak markasıyız. Yemek tarifleri, restoran incelemeleri, sağlıklı "
        "beslenme ve mutfak içerikleri üreten fenomenlerle çalışmak istiyoruz. "
        "Ev yemekleri, gurme deneyimler ve pratik tarifler paylaşan içerik "
        "üreticileri arıyoruz."
    ),
    "annebebek_kampanyasi": (
        "Anne ve bebek ürünleri markasıyız. Annelik deneyimleri, bebek bakımı, "
        "çocuk gelişimi ve aile yaşamı içerikleri üreten fenomenlerle çalışmak "
        "istiyoruz. Hamilelik süreci, emzirme, bebek beslenmesi ve ebeveynlik "
        "önerileri paylaşan içerik üreticileri arıyoruz."
    ),
    "oyun_kampanyasi": (
        "Oyun ve e-spor markasıyız. Oyun incelemeleri, canlı yayın, e-spor turnuvaları "
        "ve gaming setup içerikleri üreten fenomenlerle çalışmak istiyoruz. "
        "Strateji oyunları, FPS oyunları ve oyun dünyası haberleri paylaşan "
        "içerik üreticileri arıyoruz."
    ),
}

SIM_COLS = [f"sim_{c}" for c in CAMPAIGN_NAMES]

# ── Kampanya keyword'leri — Türkçe ──────────────────────────────────────────
CAMPAIGN_KEYWORDS: dict[str, list[str]] = {
    "spor_kampanyasi": [
        "spor", "antrenman", "fitness", "gym", "koşu", "yoga", "pilates",
        "egzersiz", "futbol", "basketbol", "voleybol", "maç", "turnuva",
        "atletizm", "kondisyon", "kas", "protein", "beslenme", "diyetisyen",
    ],
    "moda_kampanyasi": [
        "moda", "kombin", "makyaj", "skincare", "trend", "kıyafet", "güzellik",
        "parfüm", "stil", "aksesuar", "ootd", "fashion", "beauty", "makeup",
        "saç", "cilt", "kozmetik", "gardırop", "butik", "koleksiyon",
    ],
    "teknoloji_kampanyasi": [
        "teknoloji", "yazılım", "kodlama", "yapay zeka", "uygulama", "dijital",
        "telefon", "bilgisayar", "inceleme", "unboxing", "gadget", "samsung",
        "iphone", "review", "ai", "robot", "drone", "kamera", "internet",
    ],
    "yemek_kampanyasi": [
        "yemek", "tarif", "mutfak", "restoran", "lezzet", "gastronomi", "şef",
        "pişirme", "kahve", "tatlı", "börek", "pasta", "gurme", "nefis",
        "kahvaltı", "akşam yemeği", "vegan", "diyet", "beslenme",
    ],
    "annebebek_kampanyasi": [
        "anne", "bebek", "çocuk", "hamilelik", "doğum", "annelik", "aile",
        "babalık", "emzirme", "oyuncak", "okul", "kreş", "gebelik",
        "momlife", "parenting", "kids", "baby", "motherhood",
    ],
    "oyun_kampanyasi": [
        "oyun", "gaming", "gamer", "e-spor", "twitch", "yayın", "canlı",
        "playstation", "xbox", "minecraft", "valorant", "pubg", "streamer",
        "esports", "konsol", "pc", "fps", "strateji", "elraenn",
    ],
}

# ── Kampanya-kategori uyum tablosu ───────────────────────────────────────────
CAMPAIGN_CATEGORY_BONUS: dict[str, dict[str, float]] = {
    "spor_kampanyasi":      {"spor": 12, "saglik": 6, "lifestyle": 2, "yemek": -6, "oyun": -6, "moda": -4},
    "moda_kampanyasi":      {"moda": 12, "lifestyle": 4, "saglik": 2, "spor": -6, "oyun": -8, "teknoloji": -4},
    "teknoloji_kampanyasi": {"teknoloji": 12, "oyun": 4, "egitim": 4, "spor": -6, "yemek": -8, "moda": -4},
    "yemek_kampanyasi":     {"yemek": 12, "lifestyle": 2, "saglik": 2, "spor": -6, "oyun": -8, "teknoloji": -6},
    "annebebek_kampanyasi": {"anne-bebek": 12, "saglik": 4, "lifestyle": 2, "oyun": -8, "spor": -4, "teknoloji": -4},
    "oyun_kampanyasi":      {"oyun": 12, "teknoloji": 4, "lifestyle": -4, "yemek": -8, "moda": -6, "anne-bebek": -6},
}

CAMPAIGN_CATEGORY_ALLOWLIST: dict[str, list[str]] = {
    "spor_kampanyasi":      ["spor", "saglik", "lifestyle"],
    "moda_kampanyasi":      ["moda", "lifestyle", "saglik"],
    "teknoloji_kampanyasi": ["teknoloji", "oyun", "egitim"],
    "yemek_kampanyasi":     ["yemek", "lifestyle", "saglik"],
    "annebebek_kampanyasi": ["anne-bebek", "saglik", "lifestyle"],
    "oyun_kampanyasi":      ["oyun", "teknoloji"],
}

# ── Veri yükleme ─────────────────────────────────────────────────────────────
print("Veri ve modeller yükleniyor...")

_ckpt_candidates = [
    PIPELINE_DIR / "influencer_summary_checkpoint_safe.pkl",
    PIPELINE_DIR / "influencer_summary_checkpoint.pkl",
]
_ckpt_path = next((p for p in _ckpt_candidates if p.exists()), _ckpt_candidates[-1])
try:
    with open(_ckpt_path, "rb") as f:
        influencer_summary: pd.DataFrame = pickle.load(f)
    print(f"OK  Checkpoint yüklendi: {len(influencer_summary)} fenomen")
except FileNotFoundError:
    print("HATA: Checkpoint bulunamadı.", file=sys.stderr)
    sys.exit(1)
except Exception as _e:
    print(f"HATA: Checkpoint yüklenemedi: {_e}", file=sys.stderr)
    sys.exit(1)

is_mongo_active = False
_mongo_col = None
try:
    _mongo_col = get_collection()
    is_mongo_active = _mongo_col is not None
except Exception as _e:
    print(f"UYARI: MongoDB kontrolü başarısız ({_e}). pkl checkpoint aktif.")

if is_mongo_active:
    try:
        _mongo_count = _mongo_col.count_documents({})
        if _mongo_count > 0:
            _mongo_records = list(_mongo_col.find({}, {"_id": 0}))
            influencer_summary = pd.DataFrame(_mongo_records)
            print(f"OK  MongoDB aktif: {_mongo_count} fenomen")
        else:
            print("UYARI: MongoDB boş. pkl aktif.")
    except Exception as _e:
        print(f"UYARI: MongoDB okuma başarısız ({_e}). pkl aktif.")

# ── Eksik sütunları güvenli tamamla ─────────────────────────────────────────
_defaults = {
    "estimated_gender"     : "belirsiz",
    "gender_confidence"    : 0.5,
    "risk_category"        : "bilinmiyor",
    "fake_followers_risk"  : 0.0,
    "similarity_cluster"   : 0,
    "positive_ratio"       : 50.0,
    "negative_ratio"       : 20.0,
    "avg_sentiment_score"  : 0.5,
    "avg_signed_sentiment" : 0.0,
    "clean_tags_all"       : "",
    "data_source"          : "instagram",
    "avg_views"            : 0.0,
    "engagement_rate"      : 0.0,
    "FGR"                  : 0.0,
    "posts_per_month"      : 0.0,
    "NFS"                  : 0.0,
    "BAS"                  : 0.0,
}
for col, val in _defaults.items():
    if col not in influencer_summary.columns:
        influencer_summary[col] = val

# data_source: @influencerX formatı synthetic, diğerleri instagram
if "influencer_name" in influencer_summary.columns:
    inferred = influencer_summary["influencer_name"].astype(str).apply(
        lambda n: "synthetic" if re.match(r"^@influencer\d+$", n) else "instagram"
    )
    influencer_summary["data_source"] = np.where(
        influencer_summary["data_source"].isin(["instagram", "synthetic"]),
        influencer_summary["data_source"],
        inferred,
    )

# Eksik sim sütunlarını sıfırla
for c in SIM_COLS:
    if c not in influencer_summary.columns:
        influencer_summary[c] = 0.0

# ── ML modeli yükle ──────────────────────────────────────────────────────────
_xgb_path  = PIPELINE_DIR / "best_model_xgb.pkl"
_le_path   = PIPELINE_DIR / "label_encoder.pkl"
_feat_path = PIPELINE_DIR / "feature_columns.pkl"

xgb_model       = joblib.load(_xgb_path)  if _xgb_path.exists()  else None
label_encoder   = joblib.load(_le_path)   if _le_path.exists()   else None
feature_columns = joblib.load(_feat_path) if _feat_path.exists() else None

if xgb_model:
    print("OK  XGBoost pipeline yüklendi")
else:
    print("UYARI: XGBoost modeli bulunamadı")

# ── SBERT ────────────────────────────────────────────────────────────────────
_SBERT_DEVICE = "cpu"
sbert_model = SentenceTransformer(
    "paraphrase-multilingual-MiniLM-L12-v2",
    local_files_only=True,
    device=_SBERT_DEVICE,
)

def _sbert_encode(texts):
    return sbert_model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        device=_SBERT_DEVICE,
    )

print("Kampanya embedding'leri hazırlanıyor...")
CAMPAIGN_EMBEDDINGS: dict[str, np.ndarray] = {
    name: _sbert_encode([text])[0]
    for name, text in CAMPAIGN_TEXTS.items()
}

print("Influencer embedding cache hazırlanıyor...")
_EMBEDDING_COL = "sbert_embedding"
if _EMBEDDING_COL in influencer_summary.columns:
    try:
        _INFLUENCER_EMBEDDINGS = np.vstack(
            influencer_summary[_EMBEDDING_COL]
            .apply(lambda x: np.asarray(x, dtype=np.float32))
            .to_numpy()
        )
        print(f"OK  Embedding cache yüklendi: {_INFLUENCER_EMBEDDINGS.shape}")
    except Exception as _e:
        print(f"UYARI: Embedding cache kullanılamadı ({_e}); yeniden hesaplanıyor.")
        _INFLUENCER_EMBEDDINGS = _sbert_encode(
            influencer_summary["clean_tags_all"].fillna("").tolist()
        ).astype(np.float32)
else:
    _INFLUENCER_EMBEDDINGS = _sbert_encode(
        influencer_summary["clean_tags_all"].fillna("").tolist()
    ).astype(np.float32)
    print("UYARI: sbert_embedding yok; açılışta hesaplandı.")

_INFLUENCER_EMBEDDING_NORMS = np.linalg.norm(_INFLUENCER_EMBEDDINGS, axis=1) + 1e-10

# ── TF-IDF (KFS) ─────────────────────────────────────────────────────────────
print("TF-IDF matrisi hazırlanıyor...")
_tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True,
    token_pattern=r"(?u)\b\w+\b",
)
_tfidf_matrix = _tfidf_vectorizer.fit_transform(
    influencer_summary["clean_tags_all"].fillna("").tolist()
)
print(f"OK  TF-IDF: {_tfidf_matrix.shape[0]} x {_tfidf_matrix.shape[1]}")

# CF matrisi (opsiyonel)
_cf_matrix = None
_cf_candidates = [
    PIPELINE_DIR / "cf_similarity_matrix_safe.pkl",
    PIPELINE_DIR / "cf_similarity_matrix.pkl",
]
_cf_path = next((p for p in _cf_candidates if p.exists()), _cf_candidates[-1])
try:
    _cf_matrix = joblib.load(_cf_path)
    print(f"OK  CF matrisi yüklendi: {_cf_matrix.shape[0]}x{_cf_matrix.shape[1]}")
except FileNotFoundError:
    print("UYARI: CF matrisi bulunamadı.")
except Exception as _e:
    print(f"UYARI: CF matrisi yüklenemedi: {_e}")

print(f"OK  {len(influencer_summary)} fenomen yüklendi, API hazır\n")


# ════════════════════════════════════════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ════════════════════════════════════════════════════════════════════════════

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-10))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def calculate_bas(sfs: float, nfs: float, signed_sentiment: float, fake_risk: float) -> float:
    """
    BAS = SFS*0.35 + NFS*0.30 + SignedSentiment_normalized*0.25 + (100-FakeRisk)*0.10
    signed_sentiment: -1 ile +1 arası → 0-100'e normalize edilir
    """
    sentiment_normalized = ((float(signed_sentiment) + 1) / 2) * 100
    bas = (
        float(sfs)              * 0.35 +
        float(nfs)              * 0.30 +
        sentiment_normalized    * 0.25 +
        (100.0 - float(fake_risk)) * 0.10
    )
    return round(float(np.clip(bas, 0.0, 100.0)), 2)


def compute_brand_campaign_weights(
    brand_embedding: np.ndarray, brand_text: str = ""
) -> dict[str, float]:
    """Hibrit kampanya ağırlık hesaplama: SBERT cosine + keyword bonus → softmax"""
    raw_scores = np.array([
        cosine_sim(brand_embedding, CAMPAIGN_EMBEDDINGS[name])
        for name in CAMPAIGN_NAMES
    ])

    if brand_text:
        brand_lower = brand_text.lower()
        for i, name in enumerate(CAMPAIGN_NAMES):
            matches = sum(1 for kw in CAMPAIGN_KEYWORDS.get(name, []) if kw in brand_lower)
            raw_scores[i] += min(matches * 0.05, 0.25)

    weights = _softmax(raw_scores * 7)
    return {name: float(w) for name, w in zip(CAMPAIGN_NAMES, weights)}


def calculate_cfs(brand_weights: dict[str, float], row: pd.Series) -> float:
    """CFS = Σ brand_weight[k] * sim_kampanya[k] * 100"""
    cfs = sum(
        brand_weights.get(name, 0.0) * float(row.get(f"sim_{name}", 0.0))
        for name in CAMPAIGN_NAMES
    ) * 100.0
    return round(float(np.clip(cfs, 0.0, 100.0)), 2)


def calculate_final_score(
    sfs: float, nfs: float, cfs: float,
    signed_sentiment: float, fake_risk: float
) -> float:
    """
    FINAL = SFS*0.40 + NFS*0.20 + CFS*0.20 + SignedSentiment*0.10 + (100-FakeRisk)*0.10
    SFS < 35 ise ceza uygulanır.
    """
    sentiment_normalized = ((float(signed_sentiment) + 1) / 2) * 100
    score = (
        float(sfs)           * 0.40 +
        float(nfs)           * 0.20 +
        float(cfs)           * 0.20 +
        sentiment_normalized * 0.10 +
        (100.0 - float(fake_risk)) * 0.10
    )
    if float(sfs) < 35.0:
        penalty = (35.0 - float(sfs)) * 0.4
        score = score - penalty
    return round(float(np.clip(score, 0.0, 100.0)), 2)


def predict_ml_label(row: pd.Series, campaign_name: str) -> str:
    """XGBoost pipeline ile uygunluk etiketi tahmin eder."""
    if xgb_model is None or label_encoder is None or feature_columns is None:
        return "bilinmiyor"
    try:
        # Pipeline ham veri bekliyor — num_cols + cat_cols formatında
        num_data = {
            "engagement_rate"      : float(row.get("engagement_rate", 0)),
            "FGR"                  : float(row.get("FGR", 0)),
            "posts_per_month"      : float(row.get("posts_per_month", 0)),
            "NFS"                  : float(row.get("NFS", 0)),
            "SFS"                  : float(row.get("sfs", 0)),
            "positive_ratio"       : float(row.get("positive_ratio", 50)),
            "negative_ratio"       : float(row.get("negative_ratio", 20)),
            "avg_sentiment_score"  : float(row.get("avg_sentiment_score", 0.5)),
            "avg_signed_sentiment" : float(row.get("avg_signed_sentiment", 0.0)),
        }
        cat_data = {
            "category"    : str(row.get("category", "diğer")),
            "account_type": str(row.get("account_type", "creator")),
            "campaign"    : campaign_name,
        }
        sample = pd.DataFrame([{**num_data, **cat_data}])
        pred_enc = xgb_model.predict(sample)[0]
        return str(label_encoder.inverse_transform([pred_enc])[0])
    except Exception:
        return "bilinmiyor"


def ml_label_adjustment(label: str) -> float:
    return {
        "uygun"      : 0.0,
        "orta"       : -6.0,
        "uygun_degil": -18.0,
        "bilinmiyor" : 0.0,
    }.get(str(label), 0.0)


def _top_campaign(brand_weights: dict[str, float]) -> str:
    return max(brand_weights, key=lambda k: brand_weights[k])


# ════════════════════════════════════════════════════════════════════════════
# ANA ÖNERİ FONKSİYONU
# ════════════════════════════════════════════════════════════════════════════

def get_top_n(brand_text: str, top_n: int = 5) -> dict:
    df = influencer_summary.copy()

    # 1) SFS — marka embedding ile fenomen embedding cosine benzerliği
    brand_embedding = _sbert_encode([brand_text])[0]
    brand_norm = norm(brand_embedding) + 1e-10
    df["sfs"] = (
        (_INFLUENCER_EMBEDDINGS @ brand_embedding)
        / (_INFLUENCER_EMBEDDING_NORMS * brand_norm)
        * 100.0
    ).clip(0, 100).round(2)

    # 2) Kampanya ağırlıkları + CFS
    brand_weights = compute_brand_campaign_weights(brand_embedding, brand_text)
    sim_matrix = df[SIM_COLS].fillna(0).to_numpy(dtype=float)
    weight_vec = np.array([brand_weights.get(name, 0.0) for name in CAMPAIGN_NAMES], dtype=float)
    sim_matrix    = df[SIM_COLS].fillna(0).to_numpy(dtype=float)
    weight_vec    = np.array([brand_weights.get(name, 0.0) for name in CAMPAIGN_NAMES], dtype=float)

    # 3) BAS — signed_sentiment kullan
    df["bas"] = df.apply(
        lambda r: calculate_bas(
            r["sfs"], r["NFS"],
            r.get("avg_signed_sentiment", 0.0),
            r["fake_followers_risk"]
        ), axis=1
    )

    # 4) Final Score
    df["final_score"] = df.apply(
        lambda r: calculate_final_score(
            r["sfs"], r["NFS"], r["cfs"],
            r.get("avg_signed_sentiment", 0.0),
            r["fake_followers_risk"]
        ), axis=1
    )

    # 4b) KFS — TF-IDF keyword frekans bonusu (max +10)
    brand_tfidf  = _tfidf_vectorizer.transform([brand_text])
    kfs_raw      = sklearn_cosine(brand_tfidf, _tfidf_matrix).flatten()
    kfs_max      = kfs_raw.max() if kfs_raw.max() > 1e-9 else 1.0
    df["kfs"]    = (kfs_raw / kfs_max * 100).clip(0, 100).round(2)
    df["final_score"] = (df["final_score"] + kfs_raw / kfs_max * 10.0).clip(0, 100).round(2)

    # 5) En yakın kampanya ve kategori filtresi
    closest_camp = _top_campaign(brand_weights)
    allowed_cats = CAMPAIGN_CATEGORY_ALLOWLIST.get(closest_camp, [])

    df["category_match"] = (
        df["category"].isin(allowed_cats) if allowed_cats else True
    )
    df["semantic_match"] = df["sfs"].astype(float) >= 30.0

    # Kampanya-kategori bonusu
    cat_bonus_map = CAMPAIGN_CATEGORY_BONUS.get(closest_camp, {})
    if cat_bonus_map:
        df["cat_bonus"] = df["category"].map(cat_bonus_map).fillna(0.0)
        df["final_score"] = (df["final_score"] + df["cat_bonus"]).clip(0, 100).round(2)

    # SFS düşük olanları cezalandır
    df["relevance_ok"] = df["category_match"] & df["semantic_match"]
    df["niche_penalty"] = np.where(~df["relevance_ok"], 20.0, 0.0)
    df["final_score"] = (df["final_score"] - df["niche_penalty"]).clip(0, 100).round(2)

    # 6) Filtrele
    df_filtered = df[df["relevance_ok"]].copy()
    if len(df_filtered) < top_n:
        # Kategori filtresini esnet, semantic koru
        df_filtered = df[df["semantic_match"]].copy()
    if len(df_filtered) < top_n:
        df_filtered = df.copy()

    # Instagram önceliği
    instagram_pool = df_filtered[df_filtered["data_source"] == "instagram"].copy()
    if len(instagram_pool) >= min(5, top_n):
        df_filtered = instagram_pool

    # 7) ML etiketi
    df_filtered["ml_label"] = df_filtered.apply(
        lambda r: predict_ml_label(r, closest_camp), axis=1
    )
    df_filtered["ai_adjustment"] = df_filtered["ml_label"].apply(ml_label_adjustment)
    df_filtered["final_score"] = (
        df_filtered["final_score"].astype(float) + df_filtered["ai_adjustment"].astype(float)
    ).clip(0, 100).round(2)

    # Instagram bonusu
    df_filtered["source_bonus"] = np.where(df_filtered["data_source"] == "instagram", 3.0, 0.0)
    df_filtered["ranking_score"] = (
        df_filtered["final_score"].astype(float) + df_filtered["source_bonus"]
    ).clip(0, 100).round(2)

    top_df = (
        df_filtered
        .sort_values(["ranking_score", "final_score"], ascending=[False, False])
        .head(top_n)
        .copy()
    )

    top3_camps = sorted(brand_weights.items(), key=lambda x: x[1], reverse=True)[:3]

    # Sonuç sütunları
    base_cols = [
        "influencer_name", "category", "account_type",
        "NFS", "sfs", "cfs", "bas", "final_score",
        "ai_adjustment", "ml_label",
        "positive_ratio", "avg_signed_sentiment",
        "fake_followers_risk", "risk_category",
        "avg_views", "engagement_rate",
    ]
    optional_cols = [
        "estimated_gender", "similarity_cluster", "country",
        "data_source", "kfs",
    ]
    result_cols = base_cols + [c for c in optional_cols if c in top_df.columns]

    records = (
        top_df[result_cols]
        .rename(columns={"bas": "campaign_bas"})
        .to_dict(orient="records")
    )

    return {
        "recommendations"       : records,
        "brand_campaign_weights": {k: round(v, 4) for k, v in brand_weights.items()},
        "closest_campaign"      : closest_camp,
        "top3_campaigns"        : [{"campaign": k, "weight": round(v, 4)} for k, v in top3_camps],
    }


# ════════════════════════════════════════════════════════════════════════════
# FLASK ENDPOINT'LERİ
# ════════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api", methods=["GET"])
def api_info():
    return jsonify({
        "message"  : "Fenomen-Marka Eşleştirme API — Notebook Uyumlu v4",
        "kampanyalar": CAMPAIGN_NAMES,
        "formulas": {
            "NFS"  : "Ridge(engagement_rate, FGR, posts_per_month) — korelasyon ağırlıklı",
            "SFS"  : "cosine_sim(marka_embedding, fenomen_embedding) * 100",
            "CFS"  : "SUM(brand_weight[k] * sim_kampanya[k]) * 100",
            "BAS"  : "SFS*0.35 + NFS*0.30 + SignedSentiment_norm*0.25 + (100-FakeRisk)*0.10",
            "FINAL": "SFS*0.40 + NFS*0.20 + CFS*0.20 + SignedSentiment_norm*0.10 + (100-FakeRisk)*0.10",
        },
    })


@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json(silent=True)
    if not data or "brand_text" not in data:
        return jsonify({"error": "brand_text alanı gerekli"}), 400

    brand_text = str(data["brand_text"]).strip()
    if len(brand_text) < 10:
        return jsonify({"error": "brand_text en az 10 karakter olmalı"}), 400

    top_n = max(1, min(int(data.get("top_n", 5)), 50))

    try:
        result = get_top_n(brand_text, top_n=top_n)
        return jsonify({
            "success"               : True,
            "brand_text"            : brand_text,
            "count"                 : len(result["recommendations"]),
            "closest_campaign"      : result["closest_campaign"],
            "top3_campaigns"        : result["top3_campaigns"],
            "brand_campaign_weights": result["brand_campaign_weights"],
            "recommendations"       : result["recommendations"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/influencers", methods=["GET"])
def list_influencers():
    category = request.args.get("category", None)
    min_nfs  = float(request.args.get("min_nfs", 0))
    max_risk = float(request.args.get("max_risk", 100))
    sort_by  = request.args.get("sort_by", "NFS")
    limit    = min(int(request.args.get("limit", 50)), 200)

    if _mongo_col is not None:
        query = {
            "NFS"               : {"$gte": min_nfs},
            "fake_followers_risk": {"$lte": max_risk},
        }
        if category:
            query["category"] = {"$regex": category, "$options": "i"}
        sort_field = sort_by if sort_by in ("NFS", "positive_ratio", "BAS") else "NFS"
        cursor = _mongo_col.find(
            query,
            {"_id": 0, "influencer_name": 1, "category": 1,
             "account_type": 1, "NFS": 1, "positive_ratio": 1,
             "risk_category": 1, "fake_followers_risk": 1, "data_source": 1},
        ).sort(sort_field, -1).limit(limit)
        records = list(cursor)
        return jsonify({"success": True, "source": "mongodb",
                        "count": len(records), "influencers": records})

    df = influencer_summary.copy()
    if category:
        df = df[df["category"].str.lower() == category.lower()]
    df = df[df["NFS"] >= min_nfs]
    df = df[df["fake_followers_risk"] <= max_risk]

    sort_col = sort_by if sort_by in ("NFS", "BAS", "positive_ratio") else "NFS"
    cols = ["influencer_name", "category", "account_type",
            "NFS", "BAS", "positive_ratio", "risk_category",
            "fake_followers_risk", "avg_views", "engagement_rate"]
    avail_cols = [c for c in cols if c in df.columns]
    result = df[avail_cols].sort_values(sort_col, ascending=False).head(limit)

    return jsonify({
        "success"    : True,
        "source"     : "pkl",
        "count"      : len(result),
        "influencers": result.to_dict(orient="records"),
    })


@app.route("/influencers/<string:name>/similar", methods=["GET"])
def similar_influencers(name: str):
    df  = influencer_summary
    row = df[df["influencer_name"].str.lower() == name.lower()]

    if row.empty:
        return jsonify({"error": f"'{name}' bulunamadı"}), 404

    actual_name = row.iloc[0]["influencer_name"]

    if _cf_matrix is not None and actual_name in _cf_matrix.index:
        sim_series = _cf_matrix[actual_name].drop(labels=[actual_name])
        top_names  = sim_series.sort_values(ascending=False).head(10).index.tolist()
        top_scores = sim_series.sort_values(ascending=False).head(10).values.tolist()

        similar_df = df[df["influencer_name"].isin(top_names)].copy()
        similar_df["cf_similarity"] = similar_df["influencer_name"].map(
            dict(zip(top_names, top_scores))
        )
        cols = ["influencer_name", "category", "account_type", "NFS",
                "positive_ratio", "risk_category", "cf_similarity"]
        result = similar_df[[c for c in cols if c in similar_df.columns]] \
                            .sort_values("cf_similarity", ascending=False)

        return jsonify({
            "success"      : True,
            "influencer"   : actual_name,
            "method"       : "item_based_cf",
            "similar_count": len(result),
            "similar"      : result.to_dict(orient="records"),
        })

    # Fallback: K-Means cluster
    cluster_id = int(row.iloc[0]["similarity_cluster"])
    similar    = df[
        (df["similarity_cluster"] == cluster_id) &
        (df["influencer_name"].str.lower() != name.lower())
    ]
    cols = ["influencer_name", "category", "account_type", "NFS",
            "positive_ratio", "risk_category", "similarity_cluster"]
    result = similar[[c for c in cols if c in similar.columns]] \
                     .sort_values("NFS", ascending=False).head(10)

    return jsonify({
        "success"      : True,
        "influencer"   : actual_name,
        "method"       : "kmeans_cluster",
        "cluster_id"   : cluster_id,
        "similar_count": len(result),
        "similar"      : result.to_dict(orient="records"),
    })


@app.route("/stats", methods=["GET"])
def stats():
    df = influencer_summary

    camp_avg = {}
    for c in SIM_COLS:
        if c in df.columns:
            camp_avg[c.replace("sim_", "")] = round(float(df[c].mean()), 4)

    return jsonify({
        "success"            : True,
        "total_influencers"  : int(len(df)),
        "categories"         : df["category"].value_counts().to_dict(),
        "account_types"      : df["account_type"].value_counts().to_dict()
                               if "account_type" in df.columns else {},
        "data_sources"       : df["data_source"].value_counts().to_dict()
                               if "data_source" in df.columns else {},
        "avg_NFS"            : round(float(df["NFS"].mean()), 2),
        "avg_engagement"     : round(float(df["engagement_rate"].mean()), 2)
                               if "engagement_rate" in df.columns else None,
        "avg_views"          : round(float(df["avg_views"].mean()), 0)
                               if "avg_views" in df.columns else None,
        "risk_distribution"  : df["risk_category"].value_counts().to_dict()
                               if "risk_category" in df.columns else {},
        "cluster_distribution": df["similarity_cluster"].value_counts().sort_index().to_dict()
                               if "similarity_cluster" in df.columns else {},
        "campaign_avg_scores": camp_avg,
    })


@app.route("/campaigns", methods=["GET"])
def campaigns():
    df = influencer_summary
    result = []
    for name in CAMPAIGN_NAMES:
        col = f"sim_{name}"
        top_inf = (
            df[["influencer_name", "category", col]]
            .sort_values(col, ascending=False)
            .head(5)
            .rename(columns={col: "similarity"})
            .to_dict(orient="records")
        ) if col in df.columns else []

        result.append({
            "campaign"       : name,
            "description"    : CAMPAIGN_TEXTS[name],
            "avg_similarity" : round(float(df[col].mean()), 4) if col in df.columns else 0,
            "top_influencers": top_inf,
        })

    return jsonify({"success": True, "campaigns": result})


@app.route("/debug/campaign", methods=["POST"])
def debug_campaign():
    data = request.get_json(silent=True)
    if not data or "brand_text" not in data:
        return jsonify({"error": "brand_text alanı gerekli"}), 400

    brand_text    = str(data["brand_text"]).strip()
    brand_lower   = brand_text.lower()
    brand_embedding = _sbert_encode([brand_text])[0]

    sbert_scores = {
        name: round(float(cosine_sim(brand_embedding, CAMPAIGN_EMBEDDINGS[name])), 6)
        for name in CAMPAIGN_NAMES
    }

    keyword_detail = {}
    raw_scores     = np.array([sbert_scores[n] for n in CAMPAIGN_NAMES], dtype=float)

    for i, name in enumerate(CAMPAIGN_NAMES):
        kws     = CAMPAIGN_KEYWORDS.get(name, [])
        matched = [kw for kw in kws if kw in brand_lower]
        bonus   = round(min(len(matched) * 0.05, 0.25), 4)
        raw_scores[i] += bonus
        keyword_detail[name] = {
            "matched"   : matched,
            "bonus"     : bonus,
        }

    weights_arr     = _softmax(raw_scores * 7)
    softmax_weights = {
        name: round(float(w) * 100, 4)
        for name, w in zip(CAMPAIGN_NAMES, weights_arr)
    }

    sorted_w     = sorted(softmax_weights.items(), key=lambda x: x[1], reverse=True)
    winner, w1   = sorted_w[0]
    runner, w2   = sorted_w[1]

    return jsonify({
        "brand_text"    : brand_text,
        "sbert_scores"  : dict(sorted(sbert_scores.items(), key=lambda x: x[1], reverse=True)),
        "kw_bonuses"    : keyword_detail,
        "softmax_pct"   : dict(sorted(softmax_weights.items(), key=lambda x: x[1], reverse=True)),
        "winner"        : {"campaign": winner, "pct": w1, "margin": round(w1 - w2, 4)},
    })


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    _default_port = 5001 if sys.platform == "darwin" else 5000
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", _default_port)))