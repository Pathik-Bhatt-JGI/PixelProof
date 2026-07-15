"""Small HTML-building helpers for the ForensiQ custom theme. Kept
separate from app.py so the Streamlit orchestration logic stays readable.
"""
import html


def kpi_class(score: float) -> str:
    if score >= 65:
        return "kpi kpi-red"
    elif score >= 35:
        return "kpi kpi-amb"
    return "kpi"


def kpi_grid(components: dict, weights: dict, labels: dict) -> str:
    cards = []
    for k, v in components.items():
        cls = kpi_class(v)
        w = weights.get(k, 0) * 100
        cards.append(f'''
        <div class="{cls}">
          <div class="kpi-l">{html.escape(labels.get(k, k))}</div>
          <div class="kpi-v">{v:.1f}</div>
          <div class="kpi-s">WEIGHT {w:.0f}%</div>
        </div>''')
    return (
        '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;">'
        + "".join(cards) + "</div>"
    )


def acard_class(score: float) -> str:
    if score >= 70:
        return "acard acard-del"
    elif score >= 50:
        return "acard acard-mod"
    elif score >= 30:
        return "acard acard-new"
    return "acard acard-unr"


def findings_log(items: list) -> str:
    """items: list of (score, text) tuples."""
    cards = []
    for score, text in items:
        cls = acard_class(score)
        cards.append(f'<div class="{cls}">{html.escape(text)}</div>')
    return "".join(cards)


def abox(text: str, variant: str = "ac") -> str:
    return f'<div class="abox {variant}">{html.escape(text)}</div>'


def warn_box(text: str) -> str:
    return f'<div class="warn">&#9888; {html.escape(text)}</div>'


def badge(text: str, variant: str = "bg") -> str:
    return f'<span class="bd {variant}">{html.escape(text)}</span>'


def table(headers: list, rows: list) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(c))}</td>" for c in row)
        body_rows.append(f"<tr>{cells}</tr>")
    return f'<table class="ft"><thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'


def panel_open(title: str) -> str:
    return f'<div class="panel"><div class="ptitle">{html.escape(title)}</div>'


def panel_close() -> str:
    return "</div>"
