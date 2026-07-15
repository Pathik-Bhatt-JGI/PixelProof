"""Generates a professional forensic PDF report."""
import io
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from PIL import Image as PILImage

from .fusion import LABELS


def _pil_to_rl_image(pil_img: PILImage.Image, width=9 * cm):
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    w, h = pil_img.size
    height = width * h / w
    return RLImage(buf, width=width, height=height)


def generate_pdf_report(case_info: dict, evidence_info: dict, fusion_result: dict,
                         explanations: list, metadata_flags: list, images: dict,
                         output_path: str) -> str:
    doc = SimpleDocTemplate(output_path, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                             leftMargin=1.8 * cm, rightMargin=1.8 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], fontSize=18)
    h2 = styles["Heading2"]
    h3 = styles["Heading3"]
    body = styles["BodyText"]

    elements = []
    elements.append(Paragraph("ForensiQ &mdash; Image Authentication Report", title_style))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body))
    elements.append(Spacer(1, 0.5 * cm))

    case_data = [
        ["Case Number", case_info.get("case_number") or "N/A"],
        ["Examiner", case_info.get("examiner") or "N/A"],
        ["Evidence Filename", evidence_info.get("filename", "N/A")],
        ["File Size", f"{evidence_info.get('size_bytes', 0):,} bytes"],
        ["SHA-256", evidence_info.get("sha256", "N/A")],
        ["MD5", evidence_info.get("md5", "N/A")],
    ]
    t = Table(case_data, colWidths=[4.2 * cm, 11.8 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#0A1420")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#00FFB3")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.6 * cm))

    verdict = fusion_result["verdict"]
    elements.append(Paragraph("Verdict", h2))
    elements.append(Paragraph(
        f'<font color="{verdict["color"]}"><b>{verdict["label"]}</b></font> '
        f'&mdash; Composite score: <b>{fusion_result["final_score"]:.1f} / 100</b>', body))
    elements.append(Spacer(1, 0.4 * cm))

    elements.append(Paragraph("Component Scores", h2))
    comp_data = [["Signal", "Score (0-100)", "Weight Used"]]
    for k, v in fusion_result["components"].items():
        comp_data.append([LABELS.get(k, k), f"{v:.1f}", f"{fusion_result['weights'][k]*100:.0f}%"])
    t2 = Table(comp_data, colWidths=[8.5 * cm, 3.5 * cm, 3.5 * cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0A1420")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#00FFB3")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
    ]))
    elements.append(t2)
    elements.append(Spacer(1, 0.6 * cm))

    elements.append(Paragraph("Explainability Notes", h2))
    for note in explanations:
        elements.append(Paragraph(f"&bull; {note}", body))
    elements.append(Spacer(1, 0.4 * cm))

    if metadata_flags:
        elements.append(Paragraph("Metadata Findings", h2))
        for f in metadata_flags:
            elements.append(Paragraph(f"&bull; {f}", body))
        elements.append(Spacer(1, 0.4 * cm))

    elements.append(PageBreak())
    elements.append(Paragraph("Visual Forensic Evidence", h2))
    for label, pil_img in images.items():
        if pil_img is None:
            continue
        elements.append(Paragraph(label, h3))
        elements.append(_pil_to_rl_image(pil_img))
        elements.append(Spacer(1, 0.4 * cm))

    elements.append(Spacer(1, 0.6 * cm))
    elements.append(Paragraph(
        "Disclaimer: This report is produced by an original, fully self-implemented multi-signal "
        "forensic engine (error level analysis, frequency-domain analysis, PRNU sensor-noise "
        "consistency, Benford's Law DCT analysis, double-compression periodicity, texture-regularity "
        "analysis, chromatic-aberration consistency, CFA/demosaicing footprint detection, copy-move "
        "forgery detection, and metadata inspection). No third-party pretrained classifiers are used "
        "anywhere in this pipeline. It is intended to support, not replace, expert human forensic "
        "examination and should be interpreted alongside case context. Full methodology and citations "
        "are documented in METHODOLOGY.md.",
        styles["Italic"]))

    doc.build(elements)
    return output_path
