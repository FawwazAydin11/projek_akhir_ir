import io
import os
import re
import zipfile
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Utilities
# =========================

ALLOWED_TEXT_COLS = ["Lyric", "Lyrics", "lyrics", "lyric", "text", "content", "body"]
ALLOWED_ARTIST_COLS = ["Artist", "artist", "singer", "author"]
ALLOWED_TITLE_COLS = ["Title", "title", "song", "name"]

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower().replace("’", "'")
    # keep latin letters, digits, space and Hangul
    s = re.sub(r"[^a-z0-9\s\uac00-\ud7af]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _choose_col(cols: List[str], cand: List[str]) -> Optional[str]:
    for c in cand:
        if c in cols:
            return c
    return None

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    tcol = _choose_col(cols, ALLOWED_TEXT_COLS)
    if tcol is None:
        raise ValueError(
            f"Tidak menemukan kolom lirik. Sediakan salah satu dari: {ALLOWED_TEXT_COLS}"
        )
    acol = _choose_col(cols, ALLOWED_ARTIST_COLS) or "Artist"
    ccol = _choose_col(cols, ALLOWED_TITLE_COLS) or "Title"
    out = pd.DataFrame({
        "Artist": df.get(acol, "Unknown"),
        "Title": df.get(ccol, "Unknown"),
        "Lyric": df[tcol].astype(str),
    })
    return out

def read_tabular_file(name: str, buffer: bytes) -> Optional[pd.DataFrame]:
    try:
        bio = io.BytesIO(buffer)
        # try csv first
        try:
            df = pd.read_csv(bio)
        except Exception:
            # then tsv
            bio.seek(0)
            try:
                df = pd.read_csv(bio, sep="\t")
            except Exception:
                # then infer sep
                bio.seek(0)
                df = pd.read_csv(bio, sep=None, engine="python")
        return standardize_columns(df)
    except Exception as e:
        st.warning(f"Gagal membaca {name}: {e}")
        return None

# =========================
# Highlight helpers
# =========================

def _compile_query_regex(query: str) -> Optional[re.Pattern]:
    q = query.strip()
    if not q:
        return None
    # highlight tiap kata OR seluruh frasa
    words = [w for w in re.split(r"\s+", q) if w]
    parts = [re.escape(q)]
    parts.extend(re.escape(w) for w in words if len(w) > 1)
    pattern = r"(" + "|".join(parts) + r")"
    try:
        return re.compile(pattern, flags=re.IGNORECASE)
    except re.error:
        return None

def highlight_query_terms(text: str, query: str) -> str:
    """Balut kecocokan dengan <mark>…</mark>."""
    rx = _compile_query_regex(query)
    if not rx:
        return text
    return rx.sub(lambda m: f"<mark>{m.group(0)}</mark>", text)

def first_match_span(text: str, query: str) -> Optional[Tuple[int, int]]:
    """Cari posisi match pertama untuk snippet."""
    rx = _compile_query_regex(query)
    if not rx:
        return None
    m = rx.search(text)
    if not m:
        return None
    return (m.start(), m.end())

def make_snippet(text: str, query: str, window: int = 120) -> str:
    """Ambil potongan sekitar match pertama + highlight."""
    if not text:
        return ""
    span = first_match_span(text, query)
    if span is None:
        # fallback: potong awal teks
        snippet_raw = text[: window * 2]
    else:
        s, e = span
        start = max(0, s - window)
        end = min(len(text), e + window)
        prefix = "…" if start > 0 else ""
        suffix = "…" if end < len(text) else ""
        snippet_raw = prefix + text[start:end] + suffix
    return highlight_query_terms(snippet_raw, query)

# =========================
# Indexing & Search
# =========================

@st.cache_resource(show_spinner=False)
def build_index(
    texts: List[str],
    min_df_word: int = 2,
    min_df_char: int = 2,
    max_features_word: Optional[int] = None,
    max_features_char: Optional[int] = None,
):
    vc = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=min_df_char,
        sublinear_tf=True,
        dtype=np.float32,
        max_features=max_features_char,
    )
    vw = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=min_df_word,
        sublinear_tf=True,
        stop_words="english",
        dtype=np.float32,
        max_features=max_features_word,
    )
    Xc = vc.fit_transform(texts)
    Xw = vw.fit_transform(texts)
    X = hstack([Xc, Xw], format="csr")
    return X, vc, vw

def encode_query(q: str, vc: TfidfVectorizer, vw: TfidfVectorizer) -> csr_matrix:
    qc = vc.transform([q])
    qw = vw.transform([q])
    return hstack([qc, qw], format="csr")

def search(query: str, X: csr_matrix, vc, vw, meta: pd.DataFrame, topk: int = 10) -> pd.DataFrame:
    qv = encode_query(normalize_text(query), vc, vw)
    sims = cosine_similarity(qv, X).ravel()
    idx = np.argsort(-sims)[:topk]
    out = meta.iloc[idx].copy()
    out.insert(0, "Score", sims[idx])
    return out

# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="Lyrics IR Search", page_icon="🎵", layout="wide")
st.title("🎵 Lyrics Information Retrieval (IR)")
st.caption("Cari lagu berdasarkan potongan lirik – TF-IDF char + word n-gram + highlight snippet")

with st.sidebar:
    st.header("1) Muat Dataset")
    st.write("App akan **auto-load** `lyric_csv.zip` bila ada. Kamu juga bisa upload CSV/TSV/ZIP.")
    files = st.file_uploader(
        "Upload file (opsional)",
        type=["csv", "tsv", "zip"],
        accept_multiple_files=True,
    )

    st.divider()
    st.header("2) Konfigurasi")
    min_df_word = st.number_input("min_df (word)", 1, 50, 2)
    min_df_char = st.number_input("min_df (char)", 1, 50, 2)
    max_features_word = st.number_input("max_features (word) – 0 = tanpa batas", 0, 1_000_000, 0, step=1000)
    max_features_char = st.number_input("max_features (char) – 0 = tanpa batas", 0, 1_000_000, 0, step=1000)
    topk = st.slider("Top-K hasil", 5, 50, 10)
    snippet_chars = st.slider("Panjang snippet (karakter total sekitar match)", 60, 300, 140, step=10)

# 1) Kumpulkan data ke frames
frames: List[pd.DataFrame] = []

# 1a) Auto-load ZIP lokal (hanya CSV)
ZIP_PATH = "lyric_csv.zip"
if os.path.exists(ZIP_PATH):
    try:
        with zipfile.ZipFile(ZIP_PATH) as z:
            added = 0
            for zi in z.infolist():
                if zi.is_dir():
                    continue
                if not zi.filename.lower().endswith(".csv"):
                    continue
                with z.open(zi) as fh:
                    buf = fh.read()
                    df = read_tabular_file(zi.filename, buf)
                    if df is not None:
                        frames.append(df)
                        added += len(df)
        if added > 0:
            st.success(f"Dataset auto-load dari `{ZIP_PATH}`: {added:,} baris CSV")
    except Exception as e:
        st.error(f"Gagal membuka ZIP default `{ZIP_PATH}`: {e}")

# 1b) Tambahkan dari upload user (ZIP/CSV/TSV)
if files:
    for f in files:
        name = f.name
        data = f.read()
        if name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    for zi in z.infolist():
                        if zi.is_dir():
                            continue
                        if not zi.filename.lower().endswith((".csv", ".tsv")):
                            continue
                        with z.open(zi) as fh:
                            buf = fh.read()
                            df = read_tabular_file(zi.filename, buf)
                            if df is not None:
                                frames.append(df)
            except Exception as e:
                st.error(f"Gagal membuka ZIP upload `{name}`: {e}")
        else:
            df = read_tabular_file(name, data)
            if df is not None:
                frames.append(df)

# 2) Validasi data
if not frames:
    st.info("Belum ada data. Pastikan `lyric_csv.zip` ada di folder yang sama, atau upload CSV/TSV/ZIP di sidebar.")
    st.stop()

corpus = pd.concat(frames, ignore_index=True)
corpus["Lyric_norm"] = corpus["Lyric"].map(normalize_text)

st.success(f"Dataset siap: {len(corpus):,} baris, kolom = {list(corpus.columns)}")
with st.expander("Lihat sampel data"):
    st.dataframe(corpus.head(20), use_container_width=True)

# 3) Bangun indeks
with st.spinner("Membangun indeks TF-IDF…"):
    X, vc, vw = build_index(
        corpus["Lyric_norm"].tolist(),
        min_df_word=min_df_word,
        min_df_char=min_df_char,
        max_features_word=(None if max_features_word == 0 else max_features_word),
        max_features_char=(None if max_features_char == 0 else max_features_char),
    )

st.success(f"Indeks: {X.shape[0]:,} dokumen × {X.shape[1]:,} fitur")

# 4) Pencarian + Highlight
st.header("🔎 Pencarian")
query = st.text_input("Masukkan potongan lirik / kueri", "i'm friends with the monster")

if query:
    results = search(query, X, vc, vw, corpus[["Artist", "Title", "Lyric"]], topk=topk)

    st.subheader("Hasil")
    # tampilkan per-item sebagai 'card' + snippet highlight
    for _, row in results.iterrows():
        artist = row["Artist"]
        title = row["Title"]
        score = row["Score"]
        lyric = row["Lyric"]

        # buat snippet di sekitar match + highlight
        snip_html = make_snippet(lyric, query, window=snippet_chars // 2)
        full_html = highlight_query_terms(lyric, query)

        st.markdown(
            f"""
**🎤 {artist} – {title}**  
<small>Score: {score:.4f}</small><br>
<div style="background:#0f0f0f;padding:10px;border-radius:8px;margin:6px 0;">
  {snip_html}
</div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Lihat seluruh lirik (di-highlight)"):
            st.markdown(
                f"<div style='white-space:pre-wrap'>{full_html}</div>",
                unsafe_allow_html=True,
            )

    # tombol unduh hasil (tabel ringkas)
    csv_bytes = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Unduh hasil (CSV)",
        data=csv_bytes,
        file_name="ir_results.csv",
        mime="text/csv",
    )

st.caption("Tip: Atur min_df & max_features untuk menyeimbangkan kualitas vs. memori. Snippet menunjukkan konteks di sekitar kecocokan.")
