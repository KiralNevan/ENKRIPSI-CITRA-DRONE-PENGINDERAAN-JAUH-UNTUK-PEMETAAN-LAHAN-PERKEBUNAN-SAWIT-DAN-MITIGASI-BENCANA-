"""Streamlit demo: Enkripsi citra AES dengan S-box hasil algoritma genetika."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import io
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.aes.constants import AES_SBOX, inverse_sbox
from src.aes.core import warm_up
from src.aes.modes import (
    cipher_to_image,
    decrypt_image_bytes,
    derive_key,
    encrypt_image_array,
    random_iv,
)
from src.image.io import image_to_png_bytes, load_image
from src.image.metrics import image_metrics, npcr_uaci
from src.sbox.io import load_sbox_csv
from src.sbox.metrics import evaluate_sbox, sbox_to_grid

RESULTS = ROOT / "results"
SBOX_DIR = RESULTS / "sbox"
METRICS_DIR = RESULTS / "metrics"
FIGURES_DIR = RESULTS / "figures"
DATA_DIR = ROOT / "data" / "processed"

st.set_page_config(
    page_title="AES + S-Box GA | Kriptografi",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    .main-title {font-size: 2.0rem; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 0;}
    .sub-title {color: #5b6472; font-size: 1.0rem; margin-top: 0.15rem; margin-bottom: 1.2rem;}
    .metric-card {
        border: 1px solid #e4e8ee; border-radius: 14px; padding: 14px 16px;
        background: #ffffff; box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
    }
    .pill {display:inline-block; padding: 3px 10px; border-radius: 999px; background:#eef4ff;
           color:#1d4ed8; font-size:0.78rem; font-weight:600; margin-right:6px;}
    .small-note {color:#6b7280; font-size:0.85rem;}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _warmup():
    warm_up()
    return True


@st.cache_data(show_spinner=False)
def _load_sbox(path_str: str, mtime: float) -> np.ndarray:
    del mtime
    return load_sbox_csv(path_str)


def get_sbox() -> tuple[np.ndarray, bool]:
    path = SBOX_DIR / "best_sbox.csv"
    if path.exists():
        return _load_sbox(str(path), path.stat().st_mtime), True
    return AES_SBOX, False


@st.cache_data(show_spinner=False)
def _eval_sbox_cached(sbox_bytes: bytes) -> dict:
    return evaluate_sbox(np.frombuffer(sbox_bytes, dtype=np.uint8))


def sbox_metrics(sbox: np.ndarray) -> dict:
    return _eval_sbox_cached(np.asarray(sbox, dtype=np.uint8).tobytes())


@st.cache_data(show_spinner=False)
def _read_csv(path_str: str, mtime: float) -> pd.DataFrame:
    del mtime
    return pd.read_csv(path_str)


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return _read_csv(str(path), path.stat().st_mtime)
    return None


@st.cache_data(show_spinner=False)
def _load_preset(path_str: str, mtime: float) -> np.ndarray:
    del mtime
    return load_image(path_str, (512, 512))


def load_preset_image(name: str) -> np.ndarray | None:
    mapping = {
        "Lena": DATA_DIR / "lena.png",
        "Mandrill": DATA_DIR / "mandrill.png",
        "Remote sensing (NASA Landsat)": DATA_DIR / "remote_512.png",
    }
    path = mapping.get(name)
    if path is None or not path.exists():
        return None
    return _load_preset(str(path), path.stat().st_mtime)


def load_uploaded_image(file) -> np.ndarray:
    img = Image.open(file).convert("RGB")
    max_side = 512
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
            Image.Resampling.LANCZOS,
        )
    return np.asarray(img, dtype=np.uint8)


def histogram_figure(plain: np.ndarray, cipher: np.ndarray) -> plt.Figure:
    channels = 3 if plain.ndim == 3 else 1
    fig, axes = plt.subplots(2, channels, figsize=(12, 5.5))
    if channels == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for c in range(channels):
        p = plain[:, :, c] if channels > 1 else plain
        q = cipher[:, :, c] if channels > 1 else cipher
        axes[0, c].hist(p.ravel(), bins=256, range=(0, 256), color="#3b82f6", alpha=0.85)
        axes[0, c].set_title(f"Plaintext kanal {c}" if channels > 1 else "Plaintext")
        axes[1, c].hist(q.ravel(), bins=256, range=(0, 256), color="#ef4444", alpha=0.85)
        axes[1, c].set_title(f"Ciphertext kanal {c}" if channels > 1 else "Ciphertext")
        for ax in (axes[0, c], axes[1, c]):
            ax.set_xlim(0, 255)
            ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def corr_figure(plain: np.ndarray, cipher: np.ndarray) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, img, title in zip(axes, [plain, cipher], ["Plaintext", "Ciphertext"]):
        g = img[:, :, 0] if img.ndim == 3 else img
        rng = np.random.default_rng(42)
        n = 3000
        ys = rng.integers(0, g.shape[0], n)
        xs = rng.integers(0, g.shape[1] - 1, n)
        ax.scatter(g[ys, xs], g[ys, xs + 1], s=2, alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel("pixel (x)")
        ax.set_ylabel("pixel (x+1)")
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 255)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def metric_card(label: str, value: str, help_text: str = "") -> None:
    st.markdown(
        f"""<div class="metric-card"><div class="small-note">{label}</div>
        <div style="font-size:1.45rem;font-weight:750;">{value}</div>
        <div class="small-note">{help_text}</div></div>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">Enkripsi Citra AES dengan S-box Hasil Algoritma Genetika</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Proyek Akhir Kriptografi · S-box 16×16 · AES-128 · ECB/CBC · '
    'Lena, Mandrill, dan citra penginderaan jauh</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<span class="pill">GA / Machine Learning</span>'
    '<span class="pill">6 kriteria S-box</span>'
    '<span class="pill">Entropy · NPCR · UACI · Korelasi</span>',
    unsafe_allow_html=True,
)

_warmup()
ga_sbox, has_ga = get_sbox()

tab_sbox, tab_metrics, tab_encrypt, tab_results, tab_about = st.tabs(
    ["S-Box", "Metrik S-Box", "Enkripsi Citra", "Hasil Eksperimen", "Tentang"]
)

# ---------------------------------------------------------------------------
# Tab 1: S-box
# ---------------------------------------------------------------------------
with tab_sbox:
    st.subheader("S-Box 16×16")
    if not has_ga:
        st.warning(
            "S-box GA belum ditemukan di `results/sbox/best_sbox.csv`. "
            "Jalankan `python -m src.experiments.run_ga` terlebih dahulu. "
            "Sementara ini ditampilkan S-box AES standar."
        )
    m = sbox_metrics(ga_sbox)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        metric_card("Nonlinearity", f"{m['nl']}", "ideal 112")
    with c2:
        metric_card("SAC", f"{m['sac']:.4f}", "ideal 0,5")
    with c3:
        metric_card("BIC-NL", f"{m['bic_nl']}", "ideal 112")
    with c4:
        metric_card("BIC-SAC", f"{m['bic_sac']:.4f}", "ideal 0,5")
    with c5:
        metric_card("LP", f"{m['lp']:.5f}", "mendekati 0")
    with c6:
        metric_card("DAP", f"{m['dap']:.5f}", "mendekati 0")

    st.markdown(
        f"**Bijective:** {'✅' if m['bijective'] else '❌'} &nbsp;&nbsp; "
        f"**Balanced:** {'✅' if m['balanced'] else '❌'} &nbsp;&nbsp; "
        f"**Fixed points:** {m['fixed_points']} &nbsp;&nbsp; "
        f"**Opposite fixed points:** {m['opposite_fixed_points']}"
    )

    grid = sbox_to_grid(ga_sbox)
    grid_df = pd.DataFrame(grid)
    grid_df.index = [f"0x{i:01X}" for i in range(16)]
    grid_df.columns = [f"{i:X}" for i in range(16)]
    st.dataframe(
        grid_df.style.background_gradient(cmap="viridis", axis=None).format("{:.0f}"),
        width="stretch",
        height=590,
    )

    csv_bytes = grid_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Unduh S-box (CSV)",
        data=csv_bytes,
        file_name="sbox_ga_16x16.csv",
        mime="text/csv",
    )

    with st.expander("Lihat S-box invers"):
        inv = inverse_sbox(ga_sbox)
        inv_df = pd.DataFrame(sbox_to_grid(inv))
        st.dataframe(inv_df, width="stretch", height=400)

    heatmap = FIGURES_DIR / "sbox_heatmap.png"
    if heatmap.exists():
        st.image(str(heatmap), caption="Heatmap S-box hasil GA", width="stretch")

# ---------------------------------------------------------------------------
# Tab 2: Metrik S-box
# ---------------------------------------------------------------------------
with tab_metrics:
    st.subheader("Perbandingan Metrik S-box")
    df_metrics = read_csv_if_exists(METRICS_DIR / "sbox_metrics.csv")
    if df_metrics is None:
        st.info("Belum ada `results/metrics/sbox_metrics.csv`. Jalankan `run_ga` terlebih dahulu.")
    else:
        st.dataframe(df_metrics, width="stretch", hide_index=True)
        metric_choice = st.selectbox(
            "Pilih metrik",
            ["nl", "sac", "bic_nl", "bic_sac", "lp", "dap"],
            format_func=lambda x: {
                "nl": "Nonlinearity",
                "sac": "SAC",
                "bic_nl": "BIC-NL",
                "bic_sac": "BIC-SAC",
                "lp": "LP",
                "dap": "DAP",
            }[x],
        )
        chart_df = (
            df_metrics[df_metrics["algorithm"] != "GA simpangan baku"]
            .set_index("algorithm")[[metric_choice]]
        )
        st.bar_chart(chart_df)

    st.markdown("### Target sesuai brief")
    target = pd.DataFrame(
        [
            ["Nonlinearity", "≥ 112", "semakin tinggi"],
            ["SAC", "≈ 0,5", "0,49–0,51"],
            ["BIC-NL", "112", "semakin tinggi"],
            ["BIC-SAC", "≈ 0,5", "0,49–0,51"],
            ["LP", "mendekati 0", "≤ 0,0156 (setara AES)"],
            ["DAP", "mendekati 0", "≤ 0,0156 (setara AES)"],
        ],
        columns=["Kriteria", "Nilai ideal", "Catatan"],
    )
    st.table(target)

    fig_path = FIGURES_DIR / "sbox_metrics.png"
    if fig_path.exists():
        st.image(str(fig_path), caption="Perbandingan metrik S-box", width="stretch")

# ---------------------------------------------------------------------------
# Tab 3: Enkripsi citra
# ---------------------------------------------------------------------------
with tab_encrypt:
    st.subheader("Enkripsi & Dekripsi Citra")
    left, right = st.columns([1.1, 1.9])

    with left:
        source = st.radio(
            "Sumber citra",
            ["Lena", "Mandrill", "Remote sensing (NASA Landsat)", "Upload"],
            horizontal=False,
        )
        uploaded = None
        if source == "Upload":
            uploaded = st.file_uploader("Unggah citra (PNG/JPG/TIFF)", type=["png", "jpg", "jpeg", "tif", "tiff"])

        algorithm = st.selectbox(
            "Algoritma",
            ["S-box GA", "AES standar"] if has_ga else ["AES standar"],
            index=0,
        )
        mode = st.selectbox("Mode", ["ECB", "CBC"])
        key_text = st.text_input("Kunci (teks)", value="kriptografi-2026")
        if mode == "CBC":
            if "iv_text" not in st.session_state:
                st.session_state["iv_text"] = random_iv().hex()

            def _randomize_iv() -> None:
                st.session_state["iv_text"] = random_iv().hex()

            iv_text = st.text_input("IV (hex 32 karakter)", key="iv_text")
            st.button("Acak IV", on_click=_randomize_iv)
        else:
            iv_text = None

        run = st.button("Enkripsi & Dekripsi", type="primary", width="stretch")
        st.caption("Demo akademik. Untuk produksi gunakan mode terautentikasi (mis. GCM) dan KDF yang tepat.")

    with right:
        if run:
            try:
                if source == "Upload":
                    if uploaded is None:
                        st.error("Unggah citra terlebih dahulu.")
                        st.stop()
                    plain = load_uploaded_image(uploaded)
                else:
                    plain = load_preset_image(source)
                    if plain is None:
                        st.error(f"Citra preset `{source}` belum tersedia. Jalankan preprocessing dataset.")
                        st.stop()

                sbox = ga_sbox if algorithm == "S-box GA" else AES_SBOX
                key = derive_key(key_text)
                iv = bytes.fromhex(iv_text) if mode == "CBC" else None
                if mode == "CBC" and (iv is None or len(iv) != 16):
                    st.error("IV harus 16 byte (32 karakter hex).")
                    st.stop()

                t0 = time.perf_counter()
                cipher, meta = encrypt_image_array(
                    plain, key, mode=mode, sbox=sbox, iv=iv, algorithm=algorithm
                )
                t_enc = time.perf_counter() - t0
                t0 = time.perf_counter()
                decrypted = decrypt_image_bytes(
                    cipher, key, plain.shape, mode=mode, sbox=sbox, iv=iv
                )
                t_dec = time.perf_counter() - t0
                cipher_img = cipher_to_image(cipher, plain.shape)
                metrics = image_metrics(plain, cipher_img, decrypted)

                # NPCR/UACI standar: bandingkan dua ciphertext.
                plain2 = plain.copy()
                cy, cx = 0, 0
                if plain2.ndim == 3:
                    plain2[cy, cx, 0] ^= 1
                else:
                    plain2[cy, cx] ^= 1
                cipher2, _ = encrypt_image_array(
                    plain2, key, mode=mode, sbox=sbox, iv=iv, algorithm=algorithm
                )
                npcr_p, uaci_p = npcr_uaci(cipher, cipher2)

                key2 = bytes([key[0] ^ 0x01]) + key[1:]
                cipher_k, _ = encrypt_image_array(
                    plain, key2, mode=mode, sbox=sbox, iv=iv, algorithm=algorithm
                )
                npcr_k, uaci_k = npcr_uaci(cipher, cipher_k)
                metrics.update(
                    {
                        "npcr": npcr_p,
                        "uaci": uaci_p,
                        "npcr_plain": npcr_p,
                        "uaci_plain": uaci_p,
                        "npcr_key": npcr_k,
                        "uaci_key": uaci_k,
                    }
                )

                st.session_state["result"] = {
                    "plain": plain,
                    "cipher_img": cipher_img,
                    "decrypted": decrypted,
                    "cipher": cipher,
                    "meta": meta,
                    "metrics": metrics,
                    "t_enc": t_enc,
                    "t_dec": t_dec,
                    "algorithm": algorithm,
                    "mode": mode,
                    "key_hex": key.hex(),
                    "iv_hex": iv.hex() if iv else None,
                    "source": source,
                }
            except Exception as exc:
                st.exception(exc)

        result = st.session_state.get("result")
        if result:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(result["plain"], caption="Plaintext", width="stretch")
            with c2:
                st.image(result["cipher_img"], caption="Ciphertext (visualisasi)", width="stretch")
            with c3:
                st.image(result["decrypted"], caption="Hasil dekripsi", width="stretch")

            mm = result["metrics"]
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                metric_card("Entropy ciphertext", f"{mm['entropy_cipher']:.4f}", "ideal 8")
            with k2:
                metric_card("NPCR", f"{mm['npcr']:.2f}%", "ideal ≈100%")
            with k3:
                metric_card("UACI", f"{mm['uaci']:.2f}%", "ideal ≈33%")
            with k4:
                metric_card("MSE", f"{mm['mse']:.4f}", "lossless jika 0")

            if result["mode"] == "ECB":
                st.caption(
                    "Catatan: NPCR/UACI di atas adalah sensitivitas plaintext (satu piksel). "
                    "ECB tidak memiliki difusi antarblok sehingga nilainya ~0%; "
                    "sensitivitas kunci tetap ~99,6%. Gunakan CBC untuk difusi penuh."
                )

            st.markdown(
                f"**Korelasi ciphertext:** H = {mm['corr_cipher_h']:.4f}, "
                f"V = {mm['corr_cipher_v']:.4f}, D = {mm['corr_cipher_d']:.4f}  \n"
                f"**Sensitivitas plaintext:** NPCR = {mm['npcr_plain']:.2f}%, "
                f"UACI = {mm['uaci_plain']:.2f}%  \n"
                f"**Sensitivitas kunci:** NPCR = {mm['npcr_key']:.2f}%, "
                f"UACI = {mm['uaci_key']:.2f}%  \n"
                f"**Waktu:** enkripsi {result['t_enc']*1000:.1f} ms, "
                f"dekripsi {result['t_dec']*1000:.1f} ms  \n"
                f"**Lossless:** {'✅ hash & piksel identik' if mm['hash_match'] else '❌ berbeda'}  \n"
                f"**Kunci (hex):** `{result['key_hex']}`  "
                + (f"**IV (hex):** `{result['iv_hex']}`" if result["iv_hex"] else "")
            )

            with st.expander("Histogram plaintext vs ciphertext", expanded=True):
                fig = histogram_figure(result["plain"], result["cipher_img"])
                st.image(fig_to_bytes(fig), width="stretch")
            with st.expander("Scatter korelasi piksel bertetangga"):
                fig = corr_figure(result["plain"], result["cipher_img"])
                st.image(fig_to_bytes(fig), width="stretch")

            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.download_button(
                    "Ciphertext PNG",
                    data=image_to_png_bytes(result["cipher_img"]),
                    file_name=f"cipher_{result['algorithm']}_{result['mode']}.png",
                    mime="image/png",
                    width="stretch",
                )
            with d2:
                st.download_button(
                    "Ciphertext BIN",
                    data=result["cipher"],
                    file_name=f"cipher_{result['algorithm']}_{result['mode']}.bin",
                    mime="application/octet-stream",
                    width="stretch",
                )
            with d3:
                st.download_button(
                    "Citra dekripsi",
                    data=image_to_png_bytes(result["decrypted"]),
                    file_name=f"decrypted_{result['algorithm']}_{result['mode']}.png",
                    mime="image/png",
                    width="stretch",
                )
            with d4:
                st.download_button(
                    "Metrik CSV",
                    data=pd.DataFrame([mm]).to_csv(index=False).encode("utf-8"),
                    file_name="metrik_citra.csv",
                    mime="text/csv",
                    width="stretch",
                )
        else:
            st.info("Pilih citra dan pengaturan, lalu klik **Enkripsi & Dekripsi**.")

# ---------------------------------------------------------------------------
# Tab 4: Hasil eksperimen
# ---------------------------------------------------------------------------
with tab_results:
    st.subheader("Hasil Eksperimen")
    bench = read_csv_if_exists(METRICS_DIR / "benchmark_results.csv")
    remote = read_csv_if_exists(METRICS_DIR / "remote_results_512.csv")
    frames = []
    if bench is not None:
        frames.append(bench)
    if remote is not None:
        frames.append(remote)
    if not frames:
        st.info("Belum ada hasil eksperimen. Jalankan `python -m src.experiments.run_benchmark`.")
    else:
        df = pd.concat(frames, ignore_index=True)
        f1, f2 = st.columns(2)
        with f1:
            image_filter = st.multiselect("Citra", sorted(df["image"].unique()), default=sorted(df["image"].unique()))
        with f2:
            algo_filter = st.multiselect("Algoritma", sorted(df["algorithm"].unique()), default=sorted(df["algorithm"].unique()))
        view = df[df["image"].isin(image_filter) & df["algorithm"].isin(algo_filter)]
        cols = [
            "image", "size", "algorithm", "mode", "entropy", "npcr", "uaci",
            "npcr_plain", "uaci_plain", "npcr_key", "uaci_key",
            "corr_cipher_h", "corr_cipher_v", "corr_cipher_d",
            "time_enc_s", "time_dec_s", "mse", "hash_match",
        ]
        st.dataframe(view[cols], width="stretch", hide_index=True)
        st.caption(
            "NPCR/UACI pada tabel adalah sensitivitas plaintext (ubah satu piksel). "
            "ECB tidak berdifusi sehingga NPCR plaintext ~0%; kolom `npcr_key`/`uaci_key` "
            "menunjukkan sensitivitas kunci yang tetap tinggi."
        )

        st.markdown("### Grafik")
        metric = st.selectbox(
            "Metrik",
            ["entropy", "npcr", "uaci", "npcr_plain", "uaci_plain", "time_enc_s"],
            format_func=lambda x: {
                "entropy": "Entropy ciphertext",
                "npcr": "NPCR (%)",
                "uaci": "UACI (%)",
                "npcr_plain": "NPCR sensitivitas plaintext (%)",
                "uaci_plain": "UACI sensitivitas plaintext (%)",
                "time_enc_s": "Waktu enkripsi (s)",
            }[x],
        )
        view = view.copy()
        view["label"] = view["image"] + " · " + view["algorithm"] + " · " + view["mode"]
        st.bar_chart(view.set_index("label")[[metric]])

        for name, caption in [
            ("benchmark_entropy.png", "Entropy ciphertext"),
            ("benchmark_npcr.png", "NPCR"),
            ("benchmark_uaci.png", "UACI"),
            ("benchmark_correlation.png", "Korelasi ciphertext"),
            ("benchmark_time.png", "Waktu enkripsi"),
        ]:
            path = FIGURES_DIR / name
            if path.exists():
                st.image(str(path), caption=caption, width="stretch")

# ---------------------------------------------------------------------------
# Tab 5: Tentang
# ---------------------------------------------------------------------------
with tab_about:
    st.subheader("Tentang Proyek")
    st.markdown(
        """
Proyek ini membangun **S-box 16×16** dengan **algoritma genetika** (pendekatan
machine learning yang diizinkan dosen), menggantikan S-box AES pada
**SubBytes**, **SubWord**, dan **InvSubBytes**, lalu menguji enkripsi/dekripsi
citra pada mode **ECB** dan **CBC**.

**Alur:**
1. GA membangun permutasi 0–255 (bijective dan balanced).
2. Fitness menggabungkan enam kriteria: NL, SAC, BIC-NL, BIC-SAC, LP, DAP.
3. S-box terbaik menggantikan S-box AES-128.
4. Citra dienkripsi/didekripsi; dihitung entropy, histogram, NPCR, UACI,
   korelasi, waktu, MSE, dan hash SHA-256.
5. Hasil ditampilkan di Streamlit dan dilaporkan dalam PDF.

**Reproduksi:**
```bash
python -m src.experiments.run_ga
python -m src.experiments.run_benchmark
python -m src.experiments.run_remote
python -m src.experiments.make_figures
streamlit run app/streamlit_app.py
```

**Catatan:**
- Implementasi AES-128 dibuat dari nol dan divalidasi dengan FIPS-197 serta
  PyCryptodome (ECB/CBC).
- Kunci/IV di aplikasi adalah demo, bukan untuk produksi.
- Citra benchmark: Lena dan Mandrill (OpenCV samples). Citra penginderaan
  jauh: Landsat Kalimantan, NASA Earth Observatory (public domain).
- Jika target S-box tidak tercapai, nilai sebenarnya tetap dilaporkan.
        """
    )
    st.markdown("---")
    st.caption(
        "Dibuat untuk Proyek Akhir Kriptografi · Teknik Informatika · UNNES 2026"
    )
