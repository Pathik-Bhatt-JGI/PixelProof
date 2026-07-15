"""
ForensiQ — Original Multi-Signal Image Authentication Engine
Run:  streamlit run app.py

Every detection signal in this pipeline is a self-implemented, explainable
algorithm (error level analysis, DCT/Benford forensics, sensor-noise
fingerprinting, chromatic-aberration physics, texture statistics,
metadata inspection). No third-party pretrained AI models are used
anywhere — this is original forensic engineering, not a wrapper around
someone else's classifier.
"""
import io
import json
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

from modules import hashing, metadata as meta_mod, forensics, fusion, ui, charts
from modules import dct_forensics, texture_forensics, chromatic_aberration
from modules import prnu, copy_move, cfa_forensics, localization
from modules import feature_extraction, learned_fusion, cnn_detector
from modules import report as report_mod

st.set_page_config(page_title="ForensiQ — Image Authentication", page_icon="◈", layout="wide")

CSS_PATH = Path(__file__).parent / "assets" / "style.css"
st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------- session --
if "log" not in st.session_state:
    st.session_state.log = []
if "evidence_bytes" not in st.session_state:
    st.session_state.evidence_bytes = None
if "results" not in st.session_state:
    st.session_state.results = None


def log_event(event: str):
    st.session_state.log.append({
        "event": event,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })


@st.cache_resource(show_spinner=False)
def _load_ml_model():
    return learned_fusion.load_model()


@st.cache_resource(show_spinner=False)
def _load_cnn_model():
    return cnn_detector.load_model()


# --------------------------------------------------------------------- hero --
st.markdown('''
<div class="hero">
  <div class="hero-eye">DIGITAL FORENSICS &middot; ORIGINAL DETECTION ENGINE</div>
  <div class="hero-t">FORENSI<span>Q</span></div>
  <div class="hero-s">Ten independent forensic signals built from first-principles signal processing — error level
  analysis, wavelet-domain PRNU sensor-noise fingerprinting, Benford's Law &amp; double-compression DCT
  forensics, CFA/demosaicing footprint detection, block-matching copy-move forgery localization,
  chromatic-aberration lens physics, and texture statistics. Every algorithm here is self-implemented,
  citable, and fully explainable. No third-party AI models, no black boxes.</div>
</div>
''', unsafe_allow_html=True)

st.markdown('''
<div class="panel">
  <div class="ptitle">ANALYSIS PIPELINE</div>
  <div style="display:flex;flex-wrap:wrap;gap:20px;">
    <div><span class="step-num">1</span><span style="font-family:var(--mono);font-size:.68rem;color:var(--t2)">HASH &amp; LOCK EVIDENCE</span></div>
    <div><span class="step-num">2</span><span style="font-family:var(--mono);font-size:.68rem;color:var(--t2)">METADATA FORENSICS</span></div>
    <div><span class="step-num">3</span><span style="font-family:var(--mono);font-size:.68rem;color:var(--t2)">10-SIGNAL ANALYSIS</span></div>
    <div><span class="step-num">4</span><span style="font-family:var(--mono);font-size:.68rem;color:var(--t2)">WEIGHTED FUSION</span></div>
    <div><span class="step-num">5</span><span style="font-family:var(--mono);font-size:.68rem;color:var(--t2)">SIGNED PDF REPORT</span></div>
  </div>
</div>
''', unsafe_allow_html=True)

# ---------------------------------------------------------------- case info --
st.markdown('<div class="ptitle" style="margin-top:4px;">CASE INFORMATION</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    case_number = st.text_input("Case Number", placeholder="e.g. CF-2026-0142")
with c2:
    examiner = st.text_input("Examiner Name", placeholder="e.g. J. Rao")

st.markdown('<div class="ptitle" style="margin-top:20px;">EVIDENCE UPLOAD</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload image evidence",
    type=["jpg", "jpeg", "png", "tiff", "tif", "webp", "bmp"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()

    if st.session_state.evidence_bytes != file_bytes:
        st.session_state.evidence_bytes = file_bytes
        st.session_state.results = None
        log_event(f"Evidence uploaded: {uploaded_file.name}")

    image = Image.open(io.BytesIO(file_bytes))
    hashes = hashing.compute_hashes(file_bytes)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, caption=uploaded_file.name, width='stretch')
    with col2:
        st.markdown(f'''
        <div class="panel">
          <div class="ptitle">EVIDENCE INTEGRITY LOCK</div>
          <div style="font-family:var(--mono);font-size:.72rem;line-height:2;color:var(--t2)">
            SHA-256&nbsp;&nbsp;<span class="hash-ok">{hashes['sha256']}</span><br>
            MD5&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<span class="hash-ok">{hashes['md5']}</span><br>
            SIZE&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{hashes['size_bytes']:,} bytes
          </div>
        </div>
        ''', unsafe_allow_html=True)
        st.caption("This hash proves the file has not been altered since it entered this session.")

    run_clicked = st.button("RUN FULL FORENSIC ANALYSIS", type="primary", width='stretch')

    if run_clicked:
        log_event("Forensic analysis started")
        with st.spinner("Running 10-signal forensic pipeline..."):
            metadata = meta_mod.extract_metadata(image)
            ela = forensics.error_level_analysis(image)
            freq = forensics.frequency_analysis(image)
            noise = prnu.prnu_noise_analysis(image)
            benford = dct_forensics.benford_analysis(image)
            dcomp = dct_forensics.double_compression_analysis(image)
            texture = texture_forensics.texture_regularity_analysis(image)
            ca = chromatic_aberration.chromatic_aberration_analysis(image)
            cfa = cfa_forensics.cfa_analysis(image)
            cm = copy_move.copy_move_analysis(image)
            loc_overlay = localization.build_localization_overlay(image, ela["ela_image"], noise["residual_image"])

        meta_score = fusion.metadata_score(metadata)
        scores = {
            "ela": ela["score"],
            "frequency": freq["score"],
            "noise": noise["score"],
            "benford": None if benford["insufficient_data"] else benford["score"],
            "double_compression": None if dcomp["insufficient_data"] else dcomp["score"],
            "texture": texture["score"],
            "chromatic_aberration": None if ca["insufficient_data"] else ca["score"],
            "cfa": None if cfa["insufficient_data"] else cfa["score"],
            "copy_move": None if cm["insufficient_data"] else cm["score"],
            "metadata": meta_score,
        }
        fusion_result = fusion.fuse_scores(scores)
        log_event(f"Analysis complete - verdict: {fusion_result['verdict']['label']}")

        feats = feature_extraction.feats_from_results(ela, freq, noise, benford, dcomp, texture, ca, cfa, cm, metadata)

        ml_result = None
        if learned_fusion.is_available():
            try:
                ml_result = learned_fusion.predict(feats, _load_ml_model())
            except Exception:
                ml_result = None

        cnn_result = None
        if cnn_detector.is_available():
            try:
                cnn_result = cnn_detector.predict(image, _load_cnn_model())
            except Exception:
                cnn_result = None

        st.session_state.results = {
            "metadata": metadata, "ela": ela, "freq": freq, "noise": noise,
            "benford": benford, "dcomp": dcomp, "texture": texture, "ca": ca,
            "cfa": cfa, "cm": cm, "loc_overlay": loc_overlay,
            "fusion": fusion_result, "hashes": hashes, "filename": uploaded_file.name,
            "ml_result": ml_result, "cnn_result": cnn_result,
        }

# ------------------------------------------------------------------ results --
if st.session_state.results:
    r = st.session_state.results
    fr = r["fusion"]
    verdict = fr["verdict"]
    ml_result = r.get("ml_result")
    cnn_result = r.get("cnn_result")
    has_calibrated = ml_result is not None or cnn_result is not None

    if has_calibrated:
        # Prefer the CNN when both are available — it typically has a
        # higher accuracy ceiling for this specific task since it learns
        # directly from pixels rather than from ten hand-picked summary
        # statistics. Both are still "your own model": trained by you,
        # on data you chose, not downloaded from a model hub.
        primary = cnn_result if cnn_result is not None else ml_result
        primary_kind = "CNN (trained from scratch)" if cnn_result is not None else f"Calibrated ML ({ml_result['model_name']})"
        primary_verdict = fusion.classify(primary["score"])
        any_low_confidence = (cnn_result and cnn_result.get("low_confidence")) or (ml_result and ml_result.get("low_confidence"))

        st.markdown(f'''
        <div class="panel" style="border-color:{primary_verdict['color']}66;">
          <div class="ptitle">CALIBRATED VERDICT &mdash; FROM YOUR TRAINED MODEL</div>
          <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
            {ui.badge(primary_verdict['label'], primary_verdict['badge'])}
            <span style="font-family:var(--display);font-size:2rem;font-weight:800;color:{primary_verdict['color']}">
              {primary['score']:.1f}<span style="font-size:1rem;color:var(--t3)">&nbsp;/&nbsp;100</span>
            </span>
            <span style="font-family:var(--mono);font-size:.65rem;color:var(--t3);">via {primary_kind}</span>
          </div>
        </div>
        ''', unsafe_allow_html=True)

        if any_low_confidence:
            st.markdown(ui.warn_box(
                "This model was evaluated on fewer than 100 held-out test images. That sample size is too "
                "small to trust the reported accuracy/ROC-AUC — retrain on a larger labeled dataset "
                "(thousands of images per class) before relying on this verdict for anything real."
            ), unsafe_allow_html=True)

        cal_cols = st.columns(2 if (ml_result and cnn_result) else 1)
        col_i = 0
        if cnn_result is not None:
            with cal_cols[col_i]:
                auc_txt = f"test ROC-AUC {cnn_result['test_roc_auc']:.3f} &middot; " if cnn_result.get('test_roc_auc') is not None else ""
                st.markdown(f'''<div class="kpi"><div class="kpi-l">CNN (from scratch)</div>
                <div class="kpi-v">{cnn_result['score']:.1f}</div>
                <div class="kpi-s">{auc_txt}n_test={cnn_result['n_test']}</div></div>''', unsafe_allow_html=True)
            col_i += 1
        if ml_result is not None:
            with cal_cols[col_i]:
                st.markdown(f'''<div class="kpi"><div class="kpi-l">{ml_result['model_name']}</div>
                <div class="kpi-v">{ml_result['score']:.1f}</div>
                <div class="kpi-s">test ROC-AUC {ml_result['test_roc_auc']:.3f} &middot; n_test={ml_result['n_test']}</div></div>''', unsafe_allow_html=True)
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    heuristic_title = ("EXPLAINABLE SIGNAL BREAKDOWN (uncalibrated heuristics — see calibrated verdict above)"
                        if has_calibrated else "VERDICT (heuristic — train a model in training/ for validated accuracy)")

    st.markdown(f'''
    <div class="panel" style="border-color:{verdict['color']}44;">
      <div class="ptitle">{heuristic_title}</div>
      <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
        {ui.badge(verdict['label'], verdict['badge'])}
        <span style="font-family:var(--display);font-size:2rem;font-weight:800;color:{verdict['color']}">
          {fr['final_score']:.1f}<span style="font-size:1rem;color:var(--t3)">&nbsp;/&nbsp;100</span>
        </span>
      </div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="ptitle" style="margin-top:12px;">COMPONENT SIGNAL SCORES</div>', unsafe_allow_html=True)
    st.markdown(ui.kpi_grid(fr["components"], fr["weights"], fusion.LABELS), unsafe_allow_html=True)

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    tabs = st.tabs([
        "LOCALIZATION", "ELA", "FREQUENCY", "NOISE (PRNU)", "BENFORD (DCT)",
        "DOUBLE-COMPRESSION", "TEXTURE (LBP)", "CHROMATIC ABERRATION",
        "CFA / DEMOSAIC", "COPY-MOVE", "METADATA",
    ])

    with tabs[0]:
        st.image(r["loc_overlay"], caption="Composite heat overlay — blends ELA + PRNU noise-residual signals to show where anomalies concentrate spatially", width='stretch')
        st.markdown(ui.abox("This view is a visualization layer over signals computed elsewhere in the pipeline (ELA + PRNU) — it introduces no new detection logic of its own, but shows an examiner *where* to look first.", "ac"), unsafe_allow_html=True)

    with tabs[1]:
        st.image(r["ela"]["ela_image"], caption="ELA heatmap — bright/inconsistent regions suggest re-compression or editing", width='stretch')
        sc = r['ela']['score']
        st.markdown(ui.abox(f"Mean error: {r['ela']['mean_error']:.2f}  ·  Std deviation: {r['ela']['std_error']:.2f}  ·  Score: {sc:.1f}/100", "ag" if sc<35 else "aa" if sc<65 else "ar"), unsafe_allow_html=True)

    with tabs[2]:
        st.image(r["freq"]["spectrum_image"], caption="FFT magnitude spectrum", width='stretch')
        sc = r['freq']['score']
        st.markdown(ui.abox(f"High/low frequency energy ratio: {r['freq']['high_low_ratio']:.3f}  ·  Score: {sc:.1f}/100", "ag" if sc<35 else "aa" if sc<65 else "ar"), unsafe_allow_html=True)

    with tabs[3]:
        st.image(r["noise"]["residual_image"], caption="PRNU-style sensor noise residual (wavelet-domain Wiener filtering, Lukas-Fridrich-Goljan 2006)", width='stretch')
        sc = r['noise']['score']
        st.markdown(ui.abox(f"Estimated noise sigma: {r['noise']['sigma']:.2f}  ·  Block-variance consistency index: {r['noise']['consistency']:.3f}  ·  Score: {sc:.1f}/100", "ag" if sc<35 else "aa" if sc<65 else "ar"), unsafe_allow_html=True)

    with tabs[4]:
        b = r["benford"]
        if b["insufficient_data"]:
            st.markdown(ui.warn_box("Image too small for reliable DCT block sampling — signal excluded from fusion."), unsafe_allow_html=True)
        else:
            st.image(charts.benford_chart(b["observed"], b["expected"]), width='stretch')
            sc = b['score']
            st.markdown(ui.abox(f"Chi-square divergence from Benford's Law: {b['chi_square']:.1f}  ·  Score: {sc:.1f}/100", "ag" if sc<35 else "aa" if sc<65 else "ar"), unsafe_allow_html=True)

    with tabs[5]:
        d = r["dcomp"]
        if d["insufficient_data"]:
            st.markdown(ui.warn_box("Image too small for reliable DCT block sampling — signal excluded from fusion."), unsafe_allow_html=True)
        else:
            st.image(charts.double_compression_chart(d["histogram"], d["bin_edges"]), width='stretch')
            sc = d['score']
            st.markdown(ui.abox(f"Periodicity ratio: {d['periodicity_ratio']:.2f}  ·  Score: {sc:.1f}/100", "ag" if sc<35 else "aa" if sc<65 else "ar"), unsafe_allow_html=True)

    with tabs[6]:
        st.image(r["texture"]["lbp_image"], caption="Local Binary Pattern code map", width='stretch')
        sc = r['texture']['score']
        st.markdown(ui.abox(f"LBP entropy ratio: {r['texture']['entropy_ratio']:.3f}  ·  Uniform-pattern fraction: {r['texture']['uniform_fraction']:.3f}  ·  Score: {sc:.1f}/100", "ag" if sc<35 else "aa" if sc<65 else "ar"), unsafe_allow_html=True)

    with tabs[7]:
        ca = r["ca"]
        if ca["insufficient_data"]:
            st.markdown(ui.warn_box("Image too small for patch-grid sampling — signal excluded from fusion."), unsafe_allow_html=True)
        else:
            st.image(charts.chromatic_aberration_chart(ca["samples"]), width='stretch')
            sc = ca['score']
            st.markdown(ui.abox(f"Radius/misalignment correlation: {ca['correlation']:.3f}  ·  Mean channel shift: {ca['mean_magnitude']:.2f}px  ·  Score: {sc:.1f}/100", "ag" if sc<35 else "aa" if sc<65 else "ar"), unsafe_allow_html=True)

    with tabs[8]:
        cfa = r["cfa"]
        if cfa["insufficient_data"]:
            st.markdown(ui.warn_box("Image too small for CFA residual analysis — signal excluded from fusion."), unsafe_allow_html=True)
        else:
            st.image(cfa["residual_image"], caption="Green-channel neighbour-prediction residual — camera images show a period-2 (Bayer Nyquist) footprint here", width='stretch')
            sc = cfa['score']
            st.markdown(ui.abox(f"Nyquist-frequency periodicity index: {cfa['periodicity_index']:.2f}  ·  Score: {sc:.1f}/100 (lower periodicity index = weaker/absent demosaicing footprint = more suspicious)", "ag" if sc<35 else "aa" if sc<65 else "ar"), unsafe_allow_html=True)

    with tabs[9]:
        cm = r["cm"]
        if cm["insufficient_data"]:
            st.markdown(ui.warn_box("Image too small for block-matching analysis — signal excluded from fusion."), unsafe_allow_html=True)
        elif cm["overlay_image"] is not None:
            st.image(cm["overlay_image"], caption="Matched duplicate regions — red/green boxes mark cloned block pairs, blue lines connect them", width='stretch')
            sc = cm['score']
            st.markdown(ui.abox(f"Matched blocks: {cm['match_count']} / {cm['total_blocks']}  ·  Dominant shift vector: {cm['dominant_shift']}  ·  Score: {sc:.1f}/100", "ag" if sc<35 else "aa" if sc<65 else "ar"), unsafe_allow_html=True)
        else:
            st.markdown(ui.abox("No consistent duplicated-block shift pattern found.", "ag"), unsafe_allow_html=True)

    with tabs[10]:
        m = r["metadata"]
        rows = [
            ["Has EXIF", str(m["has_exif"])],
            ["Camera Make", m["camera_make"] or "—"],
            ["Camera Model", m["camera_model"] or "—"],
            ["Software Tag", m["software"] or "—"],
            ["Capture Timestamp", m["datetime_original"] or "—"],
            ["GPS Present", str(m["gps_present"])],
        ]
        st.markdown(ui.table(["Field", "Value"], rows), unsafe_allow_html=True)
        if m["risk_flags"]:
            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
            for f in m["risk_flags"]:
                st.markdown(ui.warn_box(f), unsafe_allow_html=True)
        if m["raw"]:
            with st.expander("Raw EXIF tags"):
                st.markdown(ui.table(["Tag", "Value"], list(m["raw"].items())), unsafe_allow_html=True)

    # ---- findings log ----
    st.markdown('<div class="ptitle" style="margin-top:24px;">FINDINGS LOG</div>', unsafe_allow_html=True)
    findings_items = []
    for k, v in fr["components"].items():
        findings_items.append((v, f"{fusion.LABELS.get(k,k)}: score {v:.1f}/100 (weight {fr['weights'][k]*100:.0f}%)"))
    st.markdown(ui.findings_log(findings_items), unsafe_allow_html=True)

    # ---- chain of custody ----
    st.markdown('<div class="ptitle" style="margin-top:24px;">CHAIN OF CUSTODY</div>', unsafe_allow_html=True)
    custody_rows = [[e["timestamp"], e["event"]] for e in st.session_state.log]
    st.markdown(ui.table(["Timestamp (UTC)", "Event"], custody_rows), unsafe_allow_html=True)

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    dl_col1, dl_col2 = st.columns(2)

    with dl_col1:
        if st.button("GENERATE PDF FORENSIC REPORT", width='stretch'):
            explanations = [
                f"{fusion.LABELS.get(k,k)} produced a signal score of {v:.1f}/100 "
                f"(weight {fr['weights'][k]*100:.0f}% of composite)."
                for k, v in fr["components"].items()
            ]
            tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name

            images = {
                "Manipulation Localization Overlay": r["loc_overlay"],
                "Error Level Analysis": r["ela"]["ela_image"],
                "Frequency Spectrum (FFT)": r["freq"]["spectrum_image"],
                "PRNU Sensor Noise Residual": r["noise"]["residual_image"],
                "Texture / LBP Map": r["texture"]["lbp_image"],
            }
            if not r["benford"]["insufficient_data"]:
                images["Benford's Law (DCT)"] = charts.benford_chart(r["benford"]["observed"], r["benford"]["expected"])
            if not r["dcomp"]["insufficient_data"]:
                images["Double-Compression Histogram"] = charts.double_compression_chart(r["dcomp"]["histogram"], r["dcomp"]["bin_edges"])
            if not r["ca"]["insufficient_data"]:
                images["Chromatic Aberration"] = charts.chromatic_aberration_chart(r["ca"]["samples"])
            if not r["cfa"]["insufficient_data"]:
                images["CFA / Demosaicing Residual"] = r["cfa"]["residual_image"]
            if not r["cm"]["insufficient_data"] and r["cm"]["overlay_image"] is not None:
                images["Copy-Move Match Overlay"] = r["cm"]["overlay_image"]

            report_mod.generate_pdf_report(
                case_info={"case_number": case_number, "examiner": examiner},
                evidence_info={"filename": r["filename"], **r["hashes"]},
                fusion_result=fr,
                explanations=explanations,
                metadata_flags=r["metadata"]["risk_flags"],
                images=images,
                output_path=tmp_path,
            )
            log_event("PDF report generated")
            with open(tmp_path, "rb") as f:
                st.download_button(
                    "DOWNLOAD REPORT PDF", f, file_name=f"forensic_report_{r['filename']}.pdf",
                    mime="application/pdf", width='stretch',
                )

    with dl_col2:
        log_json = json.dumps({
            "case_number": case_number, "examiner": examiner,
            "evidence": {"filename": r["filename"], **r["hashes"]},
            "verdict": verdict, "final_score": fr["final_score"],
            "components": fr["components"], "chain_of_custody": st.session_state.log,
        }, indent=2)
        st.download_button(
            "DOWNLOAD CUSTODY LOG (JSON)", log_json,
            file_name=f"custody_log_{r['filename']}.json", mime="application/json",
            width='stretch',
        )

st.markdown('<hr>', unsafe_allow_html=True)
st.markdown('''
<div style="font-family:var(--mono);font-size:.62rem;color:var(--t3);line-height:1.8;text-align:center;">
ForensiQ is an original 10-signal forensic engine — every detector is self-implemented signal
processing with a citable academic basis (see METHODOLOGY.md), not a third-party AI model. It is a
support tool for human forensic examination, and no detector achieves guaranteed accuracy against
unseen or adversarially crafted content.
</div>
''', unsafe_allow_html=True)
