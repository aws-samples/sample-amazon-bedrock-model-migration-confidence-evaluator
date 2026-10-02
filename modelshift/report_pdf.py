"""ModelShift PDF renderer — a print rendition of the web UI's Run details page.

Visual language is lifted from web/styles.css (same design tokens, semantic
colors, pills, bands, distribution bars, cards) so the PDF reads like the app.
Two themes: "dark" (default, matches the UI) and "light" (print-friendly).

Pure reportlab (no system libs) so it works locally and inside the container.
Consumes the dict produced by report.build_run_report(); never touches run state.
"""
from __future__ import annotations

import html
import io
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from reportlab.lib.colors import Color, HexColor
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    CondPageBreak, Flowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

# --------------------------------------------------------------------------
# Design tokens (web/styles.css :root) + a light print variant
# --------------------------------------------------------------------------
THEMES: Dict[str, Dict[str, str]] = {
    "dark": dict(bg="#0f1216", surface="#161a20", surface2="#1c222a", border="#262d37",
                 muted="#8b95a3", text="#e6eaf0", text_dim="#b6bfca", accent="#4f7cff",
                 pane="#0f1216", ok="#34d399", drift="#22d3ee", partial="#f5b544", bad="#f4685f"),
    "light": dict(bg="#ffffff", surface="#ffffff", surface2="#e9edf2", border="#dde2e8",
                  muted="#5d6775", text="#1a1f26", text_dim="#3a4350", accent="#3563e9",
                  pane="#f6f8fa", ok="#16956a", drift="#0b93ab", partial="#b7790c", bad="#d0433a"),
}

# UI text maps (web/app.js) so wording matches the app exactly.
LABELS = ["compatible", "compatible_with_drift", "partial", "incompatible"]
LABEL_TEXT = {"compatible": "Compatible", "compatible_with_drift": "Compatible w/ drift",
              "partial": "Partial", "incompatible": "Incompatible", "error": "Error"}
LABEL_EXPLAIN = {
    "compatible": "Drop-in interchangeable with the original response.",
    "compatible_with_drift": "Same meaning, but stylistically different (length/format drift).",
    "partial": "Usable but would need review before relying on it.",
    "incompatible": "Would break the downstream consumer - not a safe drop-in.",
}
BAND_TEXT = {"safe_drop_in": "Safe drop-in", "drop_in_with_prompt_tuning": "Drop-in with prompt tuning",
             "needs_work": "Needs work", "not_a_drop_in": "Not a drop-in"}
BAND_EXPLAIN = {
    "safe_drop_in": "Swap the model with no prompt changes.",
    "drop_in_with_prompt_tuning": "Swap after light prompt/parameter tuning.",
    "needs_work": "Material rework needed before switching.",
    "not_a_drop_in": "Not a safe replacement for this workload.",
}
ENDPOINT_TEXT = {"bedrock-runtime": "Bedrock Runtime", "bedrock-mantle": "Bedrock Mantle",
                 "litellm": "LiteLLM proxy"}
DIMENSIONS = [
    ("d1_semantic", "D1", "Semantic similarity",
     "Does the candidate mean the same thing as the original answer?", False),
    ("d2_format", "D2", "Format / schema", "Same structure (JSON validity + keys, list vs prose).", True),
    ("d3_factual", "D3", "Factual agreement", "Concrete facts match - numbers, phone numbers, emails, dates.", True),
    ("d4_verbosity", "D4", "Verbosity", "Response length is comparable - not materially longer or shorter.", False),
    ("d5_instruction", "D5", "Instruction-following",
     "Honors the system/instructions the original also followed.", False),
    ("d6_tool_call", "D6", "Tool-call match", "Same function name + argument schema (tool-call responses only).", True),
]
DIM_BY_KEY = {d[0]: d for d in DIMENSIONS}
JUDGE_REASON_TEXT = {
    "equivalent": "Judge found the candidate interchangeable with the original.",
    "missing_content": "Candidate dropped content the original included.",
    "factual_divergence": "Candidate and original disagree on a concrete fact.",
    "format_change": "Candidate changed the output format/structure.",
    "added_verbosity": "Candidate is materially more verbose than the original.",
    "instruction_violation": "Candidate did not follow the instructions.",
}
HARD_BREAK_TEXT = ("Hard break - a broken format, diverged fact, or mismatched tool call caps this "
                   "at Incompatible regardless of the other scores.")

PAGE_W, PAGE_H = letter
MARGIN_X = 0.6 * inch
TOP, BOTTOM = 0.95 * inch, 0.7 * inch
CW = PAGE_W - 2 * MARGIN_X - 12      # content width (frame has 6pt padding per side)
FRAME_H = PAGE_H - TOP - BOTTOM - 12  # content height


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
_TRANSLATE = str.maketrans({"→": "->", "←": "<-", "≥": ">=", "≤": "<=", "■": "", "✓": "v",
                            "✗": "x", "≈": "~", "•": "*"})


def _safe(s: Any) -> str:
    """Coerce to the WinAnsi charset the standard PDF fonts can draw."""
    s = "" if s is None else str(s)
    return s.translate(_TRANSLATE).encode("cp1252", "replace").decode("cp1252")


def esc(s: Any) -> str:
    return html.escape(_safe(s))


def one_line(s: Any, n: int) -> str:
    s = " ".join(_safe(s).split())
    return s if len(s) <= n else s[:n].rstrip() + "..."


def trunc(s: Any, n: int) -> Tuple[str, bool]:
    s = _safe(s).replace("\r", "")
    return (s, False) if len(s) <= n else (s[:n].rstrip(), True)


def pre(s: str) -> str:
    """Escape + keep line breaks/indentation for a mono 'diff-pane'."""
    out = []
    for line in html.escape(s).split("\n"):
        stripped = line.lstrip(" ")
        out.append("&nbsp;" * (len(line) - len(stripped)) + stripped)
    return "<br/>".join(out)


def mix(fg: Color, bg: Color, a: float) -> Color:
    """Alpha-blend fg over bg (stand-in for the UI's rgba tints)."""
    return Color(fg.red * a + bg.red * (1 - a), fg.green * a + bg.green * (1 - a),
                 fg.blue * a + bg.blue * (1 - a))


def hexs(c: Color) -> str:
    return "#%02x%02x%02x" % (round(c.red * 255), round(c.green * 255), round(c.blue * 255))


def pct(n) -> str:
    return "—" if n is None else f"{round(n)}%"


def money(n) -> str:
    return f"${(n or 0):.4f}"


def ms(n) -> str:
    return "—" if n is None else f"{round(n)} ms"


class Ctx:
    """Theme colors + paragraph styles shared by every builder."""

    def __init__(self, theme: str):
        self.t = {k: HexColor(v) for k, v in THEMES.get(theme, THEMES["dark"]).items()}
        t = self.t
        base = ParagraphStyle("base", fontName="Helvetica", fontSize=8.6, leading=12, textColor=t["text"])

        def st(name, **kw):
            return ParagraphStyle(name, parent=kw.pop("parent", base), **kw)

        self.S = {
            "base": base,
            "dim": st("dim", textColor=t["text_dim"]),
            "muted": st("muted", textColor=t["muted"], fontSize=7.8, leading=10.8),
            "small": st("small", textColor=t["muted"], fontSize=7, leading=9.4),
            "h1": st("h1", fontName="Helvetica-Bold", fontSize=20, leading=24),
            "h2": st("h2", fontName="Helvetica-Bold", fontSize=12.5, leading=15, spaceBefore=12, spaceAfter=7),
            "h3": st("h3", fontName="Helvetica-Bold", fontSize=10, leading=12.5),
            "strong": st("strong", fontName="Helvetica-Bold", fontSize=8.6, leading=11.5),
            "mono": st("mono", fontName="Courier", fontSize=7.4, leading=9.6, textColor=t["text_dim"]),
            "monosm": st("monosm", fontName="Courier", fontSize=6.9, leading=9, textColor=t["muted"]),
            "monosm_r": st("monosm_r", fontName="Courier", fontSize=6.9, leading=9, textColor=t["muted"],
                           alignment=TA_RIGHT),
            "th": st("th", fontName="Helvetica-Bold", fontSize=7.3, leading=9.5, textColor=t["muted"]),
            "td": st("td", fontSize=7.8, leading=10.2),
            "tdm": st("tdm", fontName="Courier", fontSize=7.4, leading=9.8),
            "big": st("big", fontName="Helvetica-Bold", fontSize=30, leading=31),
            "big_r": st("big_r", fontName="Helvetica-Bold", fontSize=34, leading=35, alignment=TA_RIGHT),
            "stat": st("stat", fontName="Helvetica-Bold", fontSize=17, leading=19),
            "muted_r": st("muted_r", textColor=t["muted"], fontSize=7.8, leading=10.8, alignment=TA_RIGHT),
            "td_r": st("td_r", fontSize=7.8, leading=10.2, alignment=TA_RIGHT),
            "small_r": st("small_r", textColor=t["muted"], fontSize=7, leading=9.4, alignment=TA_RIGHT),
            "hero": st("hero", fontName="Helvetica-Bold", fontSize=16, leading=20),
        }

    def label_color(self, label: str) -> Color:
        return {"compatible": self.t["ok"], "compatible_with_drift": self.t["drift"],
                "partial": self.t["partial"], "incompatible": self.t["bad"],
                "error": self.t["bad"]}.get(label or "", self.t["muted"])

    def band_color(self, band: str) -> Color:
        return {"safe_drop_in": self.t["ok"], "drop_in_with_prompt_tuning": self.t["drift"],
                "needs_work": self.t["partial"], "not_a_drop_in": self.t["bad"]}.get(band or "", self.t["muted"])

    def score_color(self, v) -> Color:
        if not isinstance(v, (int, float)):
            return self.t["muted"]
        return self.t["ok"] if v >= 0.85 else (self.t["partial"] if v >= 0.6 else self.t["bad"])

    def P(self, text: str, style: str = "base") -> Paragraph:
        return Paragraph(text, self.S[style])


# --------------------------------------------------------------------------
# Flowables: rounded card, progress bar, distribution bar, pill/band, swatch
# --------------------------------------------------------------------------
class Box(Flowable):
    """Rounded container (the UI .card / .diff-pane / .banner). Splits across
    pages only when it is taller than a full page; otherwise it moves whole."""

    def __init__(self, children: List[Flowable], fill: Color, stroke: Optional[Color] = None,
                 radius: float = 8, pad: Tuple[float, float, float, float] = (12, 12, 12, 12),
                 accent_left: Optional[Color] = None, accent_top: Optional[Color] = None,
                 min_h: float = 0, space_after: float = 0):
        super().__init__()
        self.children = [c for c in children if c is not None]
        self.fill, self.stroke, self.radius, self.pad = fill, stroke, radius, pad
        self.accent_left, self.accent_top, self.min_h = accent_left, accent_top, min_h
        self.spaceAfter = space_after
        self._layout: List[Tuple[Flowable, float, float, float]] = []

    def _clone(self, children):
        return Box(children, self.fill, self.stroke, self.radius, self.pad,
                   self.accent_left, self.accent_top, 0, self.spaceAfter)

    def _inner_w(self, aw):
        return aw - self.pad[1] - self.pad[3]

    def wrap(self, aw, ah):
        iw = self._inner_w(aw)
        self._layout = []
        h = self.pad[0] + self.pad[2]
        for i, c in enumerate(self.children):
            sb = c.getSpaceBefore() if i else 0
            sa = c.getSpaceAfter() if i < len(self.children) - 1 else 0
            _, ch = c.wrap(iw, 10 ** 6)
            self._layout.append((c, sb, ch, sa))
            h += sb + ch + sa
        self.width, self.height = aw, max(h, self.min_h)
        return self.width, self.height

    def split(self, aw, ah):
        full_h = self.wrap(aw, ah)[1]
        if full_h <= FRAME_H and ah < FRAME_H - 1:
            return []  # fits on a fresh page: move it whole (KeepTogether behaviour)
        iw = self._inner_w(aw)
        avail = ah - self.pad[0] - self.pad[2]
        if avail < 40:
            return []
        first: List[Flowable] = []
        rest = list(self.children)
        used = 0.0
        while rest:
            c = rest[0]
            sb = c.getSpaceBefore() if first else 0
            _, ch = c.wrap(iw, 10 ** 6)
            if used + sb + ch <= avail:
                first.append(rest.pop(0))
                used += sb + ch + c.getSpaceAfter()
                continue
            parts = c.split(iw, avail - used - sb)
            if parts and len(parts) > 1:
                first.append(parts[0])
                rest = list(parts[1:]) + rest[1:]
            break
        if not first or not rest:
            return []
        return [self._clone(first), self._clone(rest)]

    def draw(self):
        c = self.canv
        w, h, r = self.width, self.height, self.radius
        c.saveState()
        c.setFillColor(self.fill)
        if self.stroke is not None:
            c.setStrokeColor(self.stroke)
            c.setLineWidth(0.7)
        c.roundRect(0, 0, w, h, r, stroke=1 if self.stroke is not None else 0, fill=1)
        if self.accent_left is not None or self.accent_top is not None:
            p = c.beginPath()
            p.roundRect(0, 0, w, h, r)
            c.clipPath(p, stroke=0, fill=0)
            if self.accent_left is not None:
                c.setFillColor(self.accent_left)
                c.rect(0, 0, 3, h, stroke=0, fill=1)
            if self.accent_top is not None:
                c.setFillColor(self.accent_top)
                c.rect(0, h - 3, w, 3, stroke=0, fill=1)
        c.restoreState()
        y = h - self.pad[0]
        for child, sb, ch, sa in self._layout:
            y -= sb + ch
            child.drawOn(c, self.pad[3], y)
            y -= sa


class Bar(Flowable):
    """The UI .progress bar: rounded track + colored fill."""

    def __init__(self, frac, color: Color, track: Color, width: Optional[float] = None, height: float = 6):
        super().__init__()
        self.frac = max(0.0, min(1.0, frac if isinstance(frac, (int, float)) else 0.0))
        self.color, self.track, self.w, self.h = color, track, width, height

    def wrap(self, aw, ah):
        self.width, self.height = (self.w or aw), self.h
        return self.width, self.height

    def draw(self):
        c, w, h = self.canv, self.width, self.height
        c.saveState()
        c.setFillColor(self.track)
        c.roundRect(0, 0, w, h, h / 2, stroke=0, fill=1)
        if self.frac > 0:
            p = c.beginPath()
            p.roundRect(0, 0, w, h, h / 2)
            c.clipPath(p, stroke=0, fill=0)
            c.setFillColor(self.color)
            c.rect(0, 0, w * self.frac, h, stroke=0, fill=1)
        c.restoreState()


class DistBar(Flowable):
    """The UI .distbar: one rounded bar, segment per compatibility label."""

    def __init__(self, ctx: Ctx, dist: Dict[str, int], height: float = 8):
        super().__init__()
        self.ctx, self.dist, self.h = ctx, dist or {}, height

    def wrap(self, aw, ah):
        self.width, self.height = aw, self.h
        return aw, self.h

    def draw(self):
        c, w, h, t = self.canv, self.width, self.height, self.ctx.t
        total = sum(v for v in self.dist.values() if isinstance(v, (int, float))) or 0
        c.saveState()
        p = c.beginPath()
        p.roundRect(0, 0, w, h, h / 2)
        c.setFillColor(t["surface2"])
        c.drawPath(p, stroke=0, fill=1)
        if total:
            c.clipPath(p, stroke=0, fill=0)
            x = 0.0
            keys = LABELS + [k for k in self.dist if k not in LABELS]
            for k in keys:
                n = self.dist.get(k) or 0
                if not n:
                    continue
                seg = w * n / total
                c.setFillColor(self.ctx.label_color(k))
                c.rect(x, 0, seg + 0.3, h, stroke=0, fill=1)
                x += seg
        c.restoreState()


class Pill(Flowable):
    """UI .pill (label, with shaped swatch), .band (verdict) or .badge (outline)."""

    def __init__(self, ctx: Ctx, text: str, color: Color, kind: str = "pill",
                 label_key: str = "", size: float = 7.2):
        super().__init__()
        self.ctx, self.text, self.color, self.kind = ctx, _safe(text), color, kind
        self.label_key, self.size = label_key, size
        self.font = "Helvetica" if kind == "badge" else "Helvetica-Bold"
        sw = 8 if kind == "pill" else 0
        self.width = stringWidth(self.text, self.font, size) + (sw + 4 if sw else 0) + 12
        self.height = size + 6.5

    def wrap(self, aw, ah):
        return self.width, self.height

    def draw(self):
        c, t = self.canv, self.ctx.t
        w, h = self.width, self.height
        c.saveState()
        if self.kind == "badge":
            c.setStrokeColor(t["border"])
            c.setLineWidth(0.7)
            c.roundRect(0, 0, w, h, h / 2, stroke=1, fill=0)
            text_color = t["muted"]
        else:
            c.setFillColor(mix(self.color, t["surface"], 0.16))
            c.roundRect(0, 0, w, h, h / 2 if self.kind == "pill" else 4, stroke=0, fill=1)
            text_color = self.color
        x = 6
        if self.kind == "pill":
            s, cy = 6.0, h / 2
            c.setFillColor(self.color)
            k = self.label_key
            if k == "compatible":
                c.roundRect(x, cy - s / 2, s, s, 1.2, stroke=0, fill=1)
            elif k == "partial":  # diamond
                p = c.beginPath()
                p.moveTo(x + s / 2, cy + s / 2)
                p.lineTo(x + s, cy)
                p.lineTo(x + s / 2, cy - s / 2)
                p.lineTo(x, cy)
                p.close()
                c.drawPath(p, stroke=0, fill=1)
            else:
                c.circle(x + s / 2, cy, s / 2, stroke=0, fill=1)
            x += s + 4
        c.setFillColor(text_color)
        c.setFont(self.font, self.size)
        c.drawString(x, (h - self.size) / 2 + 1.2, self.text)
        c.restoreState()


class Swatch(Flowable):
    def __init__(self, color: Color, size: float = 7):
        super().__init__()
        self.color, self.width, self.height = color, size, size

    def wrap(self, aw, ah):
        return self.width, self.height

    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.roundRect(0, 0, self.width, self.height, 1.5, stroke=0, fill=1)


# --------------------------------------------------------------------------
# layout helpers
# --------------------------------------------------------------------------
def hrow(cells: List[Any], widths: List[float], valign: str = "TOP",
         aligns: Optional[List[str]] = None, bottom: float = 0) -> Table:
    """Invisible horizontal layout (flex row)."""
    t = Table([cells], colWidths=widths)
    style = [("VALIGN", (0, 0), (-1, -1), valign),
             ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
             ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), bottom)]
    for i, a in enumerate(aligns or []):
        style.append(("ALIGN", (i, 0), (i, 0), a))
    t.setStyle(TableStyle(style))
    return t


def grid(boxes: List[Box], width: float, cols: int, gap: float = 10) -> Optional[Table]:
    """UI .grid.cols-N with equal-height cards per row."""
    if not boxes:
        return None
    cw = (width - gap * (cols - 1)) / cols
    rows = []
    for i in range(0, len(boxes), cols):
        chunk = boxes[i:i + cols]
        tallest = max(b.wrap(cw, FRAME_H)[1] for b in chunk)
        for b in chunk:
            b.min_h = tallest
        cells: List[Any] = []
        for j in range(cols):
            if j:
                cells.append("")
            cells.append(chunk[j] if j < len(chunk) else "")
        rows.append(cells)
    widths: List[float] = []
    for j in range(cols):
        if j:
            widths.append(gap)
        widths.append(cw)
    t = Table(rows, colWidths=widths)
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                           ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), gap)]))
    return t


def ui_table(ctx: Ctx, header: List[str], rows: List[List[Any]], widths: List[float],
             center_from: Optional[int] = None) -> Table:
    """UI <table>: muted bold headers, hairline row separators, no fills."""
    data = [[ctx.P(esc(h), "th") for h in header]] + rows
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("LINEBELOW", (0, 0), (-1, -1), 0.6, ctx.t["border"]),
             ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
             ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]
    if center_from is not None:
        style.append(("ALIGN", (center_from, 0), (-1, -1), "LEFT"))
    t.setStyle(TableStyle(style))
    return t


def card(ctx: Ctx, children: List[Any], **kw) -> Box:
    kw.setdefault("fill", ctx.t["surface"])
    kw.setdefault("stroke", ctx.t["border"])
    kw.setdefault("radius", 9)
    kw.setdefault("space_after", 12)
    return Box(children, **kw)


def banner(ctx: Ctx, text: str, kind: str = "warn") -> Box:
    col = ctx.t["partial"] if kind == "warn" else (ctx.t["bad"] if kind == "err" else ctx.t["accent"])
    return Box([ctx.P(f'<font color="{hexs(col)}">{text}</font>', "base")],
               fill=mix(col, ctx.t["bg"], 0.10), stroke=mix(col, ctx.t["bg"], 0.35),
               radius=6, pad=(7, 10, 7, 10), space_after=8)


def section_title(ctx: Ctx, title: str, sub: Optional[str] = None) -> List[Any]:
    out: List[Any] = [CondPageBreak(1.3 * inch), ctx.P(esc(title), "h2")]
    if sub:
        out.append(ctx.P(sub, "muted"))
        out.append(Spacer(1, 6))
    return out


def card_head(ctx: Ctx, width: float, title: str, right: Optional[Flowable] = None) -> Table:
    left = ctx.P(f"<b>{esc(title)}</b>", "h3")
    if right is None:
        return hrow([left], [width])
    rw = min(width * 0.6, right.wrap(width, 100)[0] + 2)
    return hrow([left, right], [width - rw, rw], valign="MIDDLE", aligns=["LEFT", "RIGHT"])


# --------------------------------------------------------------------------
# page furniture
# --------------------------------------------------------------------------
def _page_decor(ctx: Ctx, report: Dict[str, Any], gen_ts: str, build: str):
    t = ctx.t

    def decorate(c, doc):
        c.saveState()
        c.setFillColor(t["bg"])
        c.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)
        # brand (rail .brand): gradient-ish logo tile + name
        y = PAGE_H - 0.58 * inch
        c.setFillColor(t["accent"])
        c.roundRect(MARGIN_X, y - 3, 18, 18, 4, stroke=0, fill=1)
        c.setFillColor(mix(HexColor("#7aa0ff"), t["accent"], 0.55))
        c.roundRect(MARGIN_X + 9, y - 3, 9, 18, 4, stroke=0, fill=1)
        c.setFillColor(HexColor("#ffffff"))
        c.setFont("Helvetica-Bold", 10)
        c.drawCentredString(MARGIN_X + 9, y + 2.2, "M")
        c.setFillColor(t["text"])
        c.setFont("Helvetica-Bold", 10.5)
        c.drawString(MARGIN_X + 25, y + 5, "ModelShift")
        c.setFillColor(t["muted"])
        c.setFont("Helvetica", 7)
        c.drawString(MARGIN_X + 25, y - 3.5, "Migration Confidence Report")
        # right: run chip (.topbar .env)
        chip = _safe(f"{report.get('run_id', '')} · {report.get('status', '')}")
        cw = stringWidth(chip, "Courier", 7) + 12
        c.setStrokeColor(t["border"])
        c.setLineWidth(0.7)
        c.roundRect(PAGE_W - MARGIN_X - cw, y - 3, cw, 14, 4, stroke=1, fill=0)
        c.setFillColor(t["muted"])
        c.setFont("Courier", 7)
        c.drawString(PAGE_W - MARGIN_X - cw + 6, y + 1.5, chip)
        c.setStrokeColor(t["border"])
        c.line(MARGIN_X, PAGE_H - 0.72 * inch, PAGE_W - MARGIN_X, PAGE_H - 0.72 * inch)
        # footer
        c.line(MARGIN_X, 0.5 * inch, PAGE_W - MARGIN_X, 0.5 * inch)
        c.setFont("Helvetica", 6.8)
        c.drawString(MARGIN_X, 0.34 * inch, _safe(
            f"{one_line(report.get('name', ''), 60)} · generated {gen_ts} · build {build}"))
        c.drawRightString(PAGE_W - MARGIN_X, 0.34 * inch, f"Page {doc.page}")
        c.restoreState()

    return decorate


# --------------------------------------------------------------------------
# page-1 sections (mirror renderResults)
# --------------------------------------------------------------------------
def _hero(ctx: Ctx, report: Dict[str, Any]) -> Optional[Box]:
    best = report.get("best")
    if not best:
        return None
    t = ctx.t
    band = best.get("verdict_band", "")
    bc = ctx.band_color(band)
    iw = CW - 31 - 2
    src = report.get("source") or {}
    n = len(report.get("candidates") or [])
    summary = (f"{n} candidate{'s' if n != 1 else ''} evaluated against "
               f"{src.get('evaluable', 0)} evaluations replayed from the existing production logs. "
               f"<b>{esc(best['model'])}</b> ranks first: {esc(BAND_TEXT.get(band, band))} with "
               f"{pct(best.get('migration_confidence'))} migration confidence and a "
               f"{pct((best.get('hard_break_rate') or 0) * 100)} hard-break rate.")
    left = [ctx.P('<font size="6.8">RECOMMENDED CANDIDATE</font>', "muted"),
            ctx.P(esc(best["model"]), "hero"),
            Spacer(1, 6),
            Pill(ctx, BAND_TEXT.get(band, band), bc, kind="band", size=7.6),
            Spacer(1, 7),
            ctx.P(summary, "dim")]
    conf = best.get("migration_confidence") or 0
    right = [ctx.P(f'<font color="{hexs(bc)}">{pct(conf)}</font>', "big_r"),
             ctx.P("Migration Confidence", "muted_r"), Spacer(1, 6),
             Bar(conf / 100.0, bc, t["surface2"], height=6)]
    body = hrow([left, "", right], [iw * 0.64, iw * 0.06, iw * 0.30], valign="MIDDLE")
    return card(ctx, [body], accent_left=bc, pad=(14, 14, 14, 17))


def _golden(ctx: Ctx, src: Dict[str, Any]) -> Box:
    t = ctx.t
    iw = CW - 27 - 2
    dropped = sum((src.get("dropped") or {}).values())
    lat = src.get("original_latency_ms") or {}
    lat_line = (f"avg {ms(lat.get('avg'))} · p50 {ms(lat.get('p50'))} · p95 {ms(lat.get('p95'))} "
                f"({lat.get('count')} timed)") if lat.get("count") else "not recorded in source logs"
    models = ", ".join(f"{m.get('model')} ({m.get('rows')})" for m in (src.get("detected_models") or [])) or "—"
    sources = ", ".join(s.replace("upload:", "") for s in (src.get("sources") or [])) or "—"

    def stat(label, val):
        return [ctx.P(esc(val), "stat"), ctx.P(esc(label), "muted")]

    stats = hrow([stat("Evaluations", src.get("evaluable", 0)), stat("Rows found", src.get("found", 0)),
                  stat("Dropped", dropped)], [iw / 3] * 3)
    kv = [("Source", sources, "tdm"), ("Existing model(s)", models, "td"), ("Original latency", lat_line, "td"),
          ("Avg tokens (prompt/total)",
           f"{src.get('avg_prompt_tokens') or '—'} / {src.get('avg_total_tokens') or '—'}", "td"),
          ("Redaction rate", pct((src.get("redaction_rate") or 0) * 100), "td")]
    tbl = Table([[ctx.P(esc(k), "muted"), ctx.P(esc(one_line(v, 300)), s)] for k, v, s in kv],
                colWidths=[iw * 0.3, iw * 0.7])
    tbl.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.6, t["border"]), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                             ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return card(ctx, [
        card_head(ctx, iw, "Golden / existing baseline", Pill(ctx, "source of truth", t["muted"], kind="badge")),
        Spacer(1, 3),
        ctx.P("The existing production logs replayed against each candidate. Candidate responses are "
              "scored against these.", "muted"),
        Spacer(1, 8), stats, Spacer(1, 8), tbl,
    ], accent_left=t["accent"], pad=(12, 12, 10, 15))


def _cand_card(ctx: Ctx, c: Dict[str, Any], width: float) -> Box:
    t = ctx.t
    iw = width - 24
    band = c.get("verdict_band", "")
    bc = ctx.band_color(band)
    dist = c.get("distribution") or {}
    name = ctx.P(f"<b>{esc(c['model'])}</b>", "strong")
    pill = Pill(ctx, BAND_TEXT.get(band, band), bc, kind="band", size=6.8)
    if stringWidth(_safe(c["model"]), "Helvetica-Bold", 8.6) + pill.width + 8 <= iw:
        head: List[Any] = [hrow([name, pill], [iw - pill.width, pill.width], valign="MIDDLE",
                                aligns=["LEFT", "RIGHT"])]
    else:
        head = [name, Spacer(1, 4), pill]
    ep = c.get("endpoint")
    dims = c.get("dimension_averages") or {}
    dim_rows = []
    for key, code, nm, _desc, _hard in DIMENSIONS:
        v = dims.get(key)
        if not isinstance(v, (int, float)):
            continue
        dim_rows.append([ctx.P(f"<b>{code}</b> {esc(nm)}", "small"),
                         Bar(v, ctx.score_color(v), t["surface2"], height=4.5),
                         ctx.P(f'<font color="{hexs(ctx.score_color(v))}"><b>{v:.2f}</b></font>', "small_r")])
    dim_tbl = None
    if dim_rows:
        dim_tbl = Table(dim_rows, colWidths=[iw * 0.50, iw * 0.36, iw * 0.14])
        dim_tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (2, 0), (2, -1), "RIGHT"),
                                     ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                                     ("RIGHTPADDING", (0, 0), (0, -1), 4),
                                     ("TOPPADDING", (0, 0), (-1, -1), 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6)]))
    counts = " · ".join(f"{dist.get(k, 0)} {LABEL_TEXT[k]}" for k in LABELS)
    meta = (f"hard-break {pct((c.get('hard_break_rate') or 0) * 100)} · {money(c.get('cost_total'))} · "
            f"{ms(c.get('latency_avg_ms'))} avg")
    kids: List[Any] = head + [
        ctx.P("via " + esc(ENDPOINT_TEXT.get(ep, ep)), "monosm") if ep else None,
        Spacer(1, 8),
        ctx.P(pct(c.get("migration_confidence")), "big"),
        ctx.P("Migration Confidence", "muted"),
        Spacer(1, 7),
        DistBar(ctx, dist, height=8),
        Spacer(1, 4),
        ctx.P(esc(counts), "monosm"),
        Spacer(1, 3),
        ctx.P(esc(meta), "muted"),
    ]
    if dim_tbl is not None:
        kids += [Spacer(1, 7), Box([], fill=t["border"], radius=0, pad=(0.35, 0, 0.35, 0)),
                 Spacer(1, 6), dim_tbl]
    if c.get("errored"):
        kids += [Spacer(1, 6), ctx.P(f'<font color="{hexs(t["bad"])}">{c["errored"]} call(s) errored</font>', "small")]
    return card(ctx, kids, accent_top=bc, pad=(13, 12, 12, 12), space_after=0)


def _side_by_side(ctx: Ctx, cands: List[Dict[str, Any]]) -> Box:
    iw = CW - 24
    widths = [78, 66, 52, 48, 42, 46, 46]  # leaves ~100pt for the verdict band
    widths.append(iw - sum(widths))
    rows = []
    for c in cands:
        band = c.get("verdict_band", "")
        rows.append([
            ctx.P(esc(c["model"]), "tdm"),
            ctx.P(esc(ENDPOINT_TEXT.get(c.get("endpoint"), c.get("endpoint") or "—")), "td"),
            ctx.P(f'<font color="{hexs(ctx.score_color((c.get("migration_confidence") or 0) / 100))}"><b>'
                  f'{pct(c.get("migration_confidence"))}</b></font>', "td"),
            ctx.P(pct((c.get("hard_break_rate") or 0) * 100), "td"),
            ctx.P(money(c.get("cost_total")), "td"),
            ctx.P(ms(c.get("latency_avg_ms")), "td"),
            ctx.P(ms(c.get("latency_p95_ms")), "td"),
            Pill(ctx, BAND_TEXT.get(band, band), ctx.band_color(band), kind="band", size=6.6),
        ])
    tbl = ui_table(ctx, ["Candidate", "Endpoint", "Confidence", "Hard break", "Cost",
                         "Latency avg", "Latency p95", "Verdict"], rows, widths)
    return card(ctx, [tbl], pad=(6, 12, 6, 12))


def _dimension_matrix(ctx: Ctx, cands: List[Dict[str, Any]]) -> Optional[Box]:
    present = [d for d in DIMENSIONS
               if any(isinstance((c.get("dimension_averages") or {}).get(d[0]), (int, float)) for c in cands)]
    if not present:
        return None
    t = ctx.t
    iw = CW - 24
    first = 150
    colw = (iw - first) / max(1, len(cands))
    rows = []
    for key, code, nm, _desc, hard in present:
        tag = f'  <font color="{hexs(t["bad"])}" size="6">HARD-BREAK</font>' if hard else ""
        row: List[Any] = [ctx.P(f"<b>{code}</b> {esc(nm)}{tag}", "td")]
        for c in cands:
            v = (c.get("dimension_averages") or {}).get(key)
            if not isinstance(v, (int, float)):
                row.append(ctx.P("—", "muted"))
                continue
            row.append(hrow([Bar(v, ctx.score_color(v), t["surface2"], height=5),
                             ctx.P(f'<font color="{hexs(ctx.score_color(v))}"><b>{v:.2f}</b></font>', "td_r")],
                            [colw - 46, 36], valign="MIDDLE", aligns=["LEFT", "RIGHT"]))
        rows.append(row)
    tbl = ui_table(ctx, ["Dimension"] + [one_line(c["model"], 28) for c in cands], rows,
                   [first] + [colw] * len(cands))
    return card(ctx, [tbl], pad=(6, 12, 6, 12))


def _failure_reasons(ctx: Ctx, cands: List[Dict[str, Any]]) -> Optional[Box]:
    by_reason: Dict[str, Dict[str, int]] = {}
    for c in cands:
        for fr in c.get("top_failure_reasons") or []:
            if isinstance(fr, dict) and fr.get("reason"):
                by_reason.setdefault(fr["reason"], {})[c["model"]] = fr.get("count", 0)
    if not by_reason:
        return None
    iw = CW - 24
    rows = []
    for reason, per in sorted(by_reason.items(), key=lambda kv: -sum(kv[1].values())):
        rows.append([ctx.P(esc(reason), "tdm"),
                     ctx.P(esc(JUDGE_REASON_TEXT.get(reason, "")), "td"),
                     ctx.P(esc(" · ".join(f"{m} {n}" for m, n in per.items())), "monosm")])
    tbl = ui_table(ctx, ["Reason", "What it means", "Count by candidate"], rows,
                   [110, iw - 110 - 150, 150])
    return card(ctx, [tbl], pad=(6, 12, 6, 12))


def _remediations(ctx: Ctx, rems: List[Dict[str, Any]]) -> Optional[Box]:
    if not rems:
        return None
    t = ctx.t
    kids: List[Any] = []
    for i, r in enumerate(rems):
        if i:
            kids += [Spacer(1, 8), Box([], fill=t["border"], radius=0, pad=(0.35, 0, 0.35, 0)), Spacer(1, 8)]
        head = (f'<font face="Courier">{esc(r.get("candidate"))}</font> · <b>{esc(r.get("reason") or "cluster")}</b>'
                f' · {r.get("evidence_count") or 0} evaluation{"" if r.get("evidence_count") == 1 else "s"}')
        kids.append(ctx.P(head, "base"))
        if r.get("expected_effect"):
            kids.append(ctx.P(esc(r["expected_effect"]), "muted"))
        if r.get("type") == "reasoning_effort":
            change = (f"Change: reasoning effort {esc(r.get('before') or '')} -> {esc(r.get('after') or '')}"
                      " (no prompt change)")
            kids += [Spacer(1, 3), ctx.P(change, "dim")]
        elif r.get("after"):
            where = "Prepend to" if r.get("placement") == "prepend" else "Append to"
            kids += [Spacer(1, 3), ctx.P(f"Change: {where} the system prompt", "dim"), Spacer(1, 3),
                     Box([ctx.P(pre(trunc(r["after"], 900)[0]), "mono")], fill=t["pane"], stroke=t["border"],
                         radius=5, pad=(6, 8, 6, 8))]
        rt = r.get("retest")
        if rt and rt.get("status") == "done":
            m = rt.get("moved") or {}

            def d(a, b):
                if a is None or b is None:
                    return "—"
                col = t["ok"] if b - a > 0.05 else (t["bad"] if b - a < -0.05 else t["muted"])
                return f'{pct(a)} -> <font color="{hexs(col)}"><b>{pct(b)}</b></font>'
            line = (f"<b>Re-tested:</b> cluster {d(rt.get('confidence_before'), rt.get('confidence_after'))} · "
                    f"overall (projected) {d(rt.get('confidence_run_before'), rt.get('confidence_run_after'))} · "
                    f"{m.get('to_compatible', 0)} fixed · {m.get('regressions', 0)} regressions in "
                    f"{rt.get('regression_checked', 0)} re-checked · {m.get('errors', 0)} errors · "
                    f"{money(rt.get('cost'))}")
            kids += [Spacer(1, 5), ctx.P(line, "base")]
        elif rt:
            kids += [Spacer(1, 5), ctx.P(f"Re-test {esc(rt.get('status'))}", "muted")]
        else:
            kids += [Spacer(1, 5), ctx.P("Not re-tested.", "muted")]
    return card(ctx, kids, pad=(12, 12, 12, 12))


def _legend(ctx: Ctx) -> Box:
    t = ctx.t
    iw = CW - 24
    half = (iw - 16) / 2

    def kv_rows(items, w):
        tb = Table(items, colWidths=[w * 0.42, w * 0.58])
        tb.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                                ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
        return tb

    scores = hrow([
        hrow([Swatch(t["ok"]), ctx.P("0.85 - 1.00 strong", "small")], [11, half / 3 - 11], valign="MIDDLE"),
        hrow([Swatch(t["partial"]), ctx.P("0.60 - 0.84 moderate", "small")], [11, half / 3 - 11], valign="MIDDLE"),
        hrow([Swatch(t["bad"]), ctx.P("below 0.60 weak", "small")], [11, half / 3 - 11], valign="MIDDLE"),
    ], [half / 3] * 3)
    bands = kv_rows([[Pill(ctx, BAND_TEXT[k], ctx.band_color(k), kind="band", size=6.6),
                      ctx.P(esc(BAND_EXPLAIN[k]), "small")] for k in BAND_TEXT], half)
    labels = kv_rows([[Pill(ctx, LABEL_TEXT[k], ctx.label_color(k), label_key=k, size=6.6),
                       ctx.P(esc(LABEL_EXPLAIN[k]), "small")] for k in LABELS], half)
    dims = Table([[ctx.P(f"<b>{code}</b> {esc(nm)}" + (
        f' <font color="{hexs(t["bad"])}" size="5.8">HARD-BREAK</font>' if hard else ""), "small"),
        ctx.P(esc(desc), "small")] for _k, code, nm, desc, hard in DIMENSIONS],
        colWidths=[half * 0.42, half * 0.58])
    dims.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                              ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                              ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
    left = [ctx.P("<b>Score colors</b> (bars and numbers)", "dim"), Spacer(1, 4), scores, Spacer(1, 10),
            ctx.P("<b>Migration verdicts</b>", "dim"), Spacer(1, 3), bands, Spacer(1, 10),
            ctx.P("<b>Evaluation labels</b>", "dim"), Spacer(1, 3), labels]
    right = [ctx.P("<b>Compatibility dimensions</b> (0.00 - 1.00, higher = more compatible)", "dim"),
             Spacer(1, 3), dims, Spacer(1, 8), banner(ctx, esc(HARD_BREAK_TEXT), "err")]
    return card(ctx, [card_head(ctx, iw, "How to read these scores"), Spacer(1, 8),
                      hrow([left, "", right], [half, 16, half])])


# --------------------------------------------------------------------------
# appendix (mirrors loadSamples table + showSampleDetail drill-down)
# --------------------------------------------------------------------------
def _eval_index(ctx: Ctx, rows: List[Dict[str, Any]]) -> Box:
    iw = CW - 24
    widths = [64, 102, 30, 30, 30, 30, 30]
    widths.append(iw - sum(widths))
    data = []
    for r in rows:
        dmap = {d.get("key"): d.get("score") for d in r.get("dimensions") or []}
        cells: List[Any] = [ctx.P(esc(r.get("sample_id")), "tdm"),
                            Pill(ctx, LABEL_TEXT.get(r.get("label"), r.get("label") or "—"),
                                 ctx.label_color(r.get("label")), label_key=r.get("label") or "", size=6.4)]
        for key in ("d1_semantic", "d2_format", "d3_factual", "d4_verbosity", "d5_instruction"):
            v = dmap.get(key)
            cells.append(ctx.P(f'<font color="{hexs(ctx.score_color(v))}">{v:.2f}</font>'
                               if isinstance(v, (int, float)) else "—", "td"))
        reason = r.get("judge_reason") or ("hard-break" if r.get("hard_break") else "—")
        if r.get("status") == "error":
            reason = "error: " + one_line(r.get("error"), 60)
        cells.append(ctx.P(esc(reason), "muted"))
        data.append(cells)
    return card(ctx, [ui_table(ctx, ["Request", "Label", "D1", "D2", "D3", "D4", "D5", "Reason"], data, widths)],
                pad=(6, 12, 6, 12))


def _pane(ctx: Ctx, title: str, sub: str, text: str, width: float, limit: int, err: bool = False) -> List[Any]:
    body, cut = trunc(text or "", limit)
    if not body.strip():
        body = "(empty)"
    style = "mono"
    para = ctx.P(f'<font color="{hexs(ctx.t["bad"])}">{pre(body)}</font>' if err else pre(body), style)
    kids: List[Any] = [para]
    if cut:
        kids += [Spacer(1, 3), ctx.P("... truncated - full text in the ModelShift app", "small")]
    max_chars = max(20, int((width - 4) / (6.9 * 0.6)))
    return [ctx.P(f"<b>{esc(title)}</b>", "strong"), Spacer(1, 1),
            ctx.P(esc(one_line(sub, max_chars)), "monosm"), Spacer(1, 4),
            Box(kids, fill=ctx.t["pane"], stroke=ctx.t["border"], radius=6, pad=(7, 8, 7, 8))]


def _eval_card(ctx: Ctx, idx: int, row: Dict[str, Any], model: str) -> Box:
    t = ctx.t
    iw = CW - 28
    label = row.get("label") or "—"
    lc = ctx.label_color(label)
    head = card_head(ctx, iw, f"#{idx} · Sample {row.get('sample_id', '')}",
                     Pill(ctx, LABEL_TEXT.get(label, label), lc, label_key=label, size=7))

    msgs = row.get("messages") or []
    req = ""
    if row.get("instructions"):
        req += f"[system/instructions]\n{row['instructions']}\n\n"
    req += "\n\n".join(f"[{m.get('role')}]\n{m.get('content')}" for m in msgs) or (row.get("prompt") or "")
    o_tok = f" · {row['original_tokens']} tok" if row.get("original_tokens") else ""
    c_tok = f" · {row['tokens']} tok" if row.get("tokens") else ""
    c_cost = f" · {money(row['cost_estimate'])}" if row.get("cost_estimate") else ""
    effort = f" · effort {row['reasoning_effort']}" if row.get("reasoning_effort") else ""
    half = (iw - 10) / 2
    is_err = row.get("status") == "error"
    cand_text = ("ERROR: " + (row.get("error") or "unknown")) if is_err else (row.get("response") or "")
    responses = hrow([
        _pane(ctx, "Original response",
              f"{row.get('original_model') or '—'} · {ms(row.get('original_latency_ms'))}{o_tok}",
              row.get("golden") or "", half, 1400),
        "",
        _pane(ctx, "Candidate response", f"{model}{effort} · {ms(row.get('latency_ms'))}{c_tok}{c_cost}",
              cand_text, half, 1400, err=is_err),
    ], [half, 10, half])

    lat_o, lat_c = row.get("original_latency_ms"), row.get("latency_ms")
    lat_line = f"Latency - original {ms(lat_o)} vs candidate {ms(lat_c)}"
    if lat_o and lat_c:
        lat_line += f" ({'faster' if lat_c <= lat_o else 'slower'} by {abs(lat_c - lat_o)} ms)"

    # compatibility scores block (scoresBlock)
    scores: List[Any] = [
        hrow([ctx.P("<b>Compatibility scores</b>", "strong"),
              ctx.P("0.00 - 1.00 · higher = more compatible", "muted_r")], [iw * 0.5, iw * 0.5], valign="MIDDLE"),
        Spacer(1, 5),
    ]
    lp = Pill(ctx, LABEL_TEXT.get(label, label), lc, label_key=label, size=6.8)
    scores.append(hrow([lp, ctx.P(esc(LABEL_EXPLAIN.get(label, "")), "muted")],
                       [lp.width + 8, iw - lp.width - 8], valign="MIDDLE"))
    if row.get("hard_break"):
        scores += [Spacer(1, 6), banner(ctx, esc(HARD_BREAK_TEXT), "err")]
    dim_rows = []
    for d in row.get("dimensions") or []:
        meta = DIM_BY_KEY.get(d.get("key"))
        code, nm, hard = (meta[1], meta[2], meta[4]) if meta else (d.get("label", ""), "", False)
        v = d.get("score")
        tag = f' <font color="{hexs(t["bad"])}" size="5.8">HARD</font>' if hard else ""
        dim_rows.append([
            ctx.P(f"<b>{esc(code)}</b> {esc(nm)}{tag}", "td"),
            Bar(v, ctx.score_color(v), t["surface2"], width=64, height=5),
            ctx.P(f'<font color="{hexs(ctx.score_color(v))}"><b>{v:.2f}</b></font>'
                  if isinstance(v, (int, float)) else "—", "td"),
            ctx.P(esc(one_line(d.get("reason"), 220)), "muted"),
        ])
    if dim_rows:
        dt = Table(dim_rows, colWidths=[132, 72, 34, iw - 238])
        dt.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                ("LINEBELOW", (0, 0), (-1, -1), 0.5, t["border"]),
                                ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                                ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]))
        scores += [Spacer(1, 6), dt]
    jr, jrat = row.get("judge_reason"), row.get("judge_rationale")
    if jr or jrat:
        jtxt = "<b>Judge verdict</b>"
        if jr:
            jtxt += f' · <font face="Courier">{esc(jr)}</font>'
            if JUDGE_REASON_TEXT.get(jr):
                jtxt += f" - {esc(JUDGE_REASON_TEXT[jr])}"
        kids = [ctx.P(jtxt, "dim")]
        if jrat:
            kids += [Spacer(1, 3), ctx.P(esc(one_line(jrat, 900)), "base")]
        scores += [Spacer(1, 8), Box(kids, fill=mix(t["accent"], t["surface"], 0.10),
                                     stroke=mix(t["accent"], t["surface"], 0.38), radius=6, pad=(7, 9, 7, 9))]

    return card(ctx, [head, Spacer(1, 9)]
                + _pane(ctx, "Request (replayed to both)", f"original model: {row.get('original_model') or '—'}",
                        req, iw, 1800)
                + [Spacer(1, 9), responses, Spacer(1, 6), ctx.P(esc(lat_line), "muted"), Spacer(1, 10)]
                + scores,
                accent_left=lc, pad=(12, 14, 12, 14), space_after=12)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def render_pdf(report: Dict[str, Any], theme: str = "dark") -> bytes:
    """Render a build_run_report() dict to PDF bytes in the UI's visual language."""
    ctx = Ctx(theme if theme in THEMES else "dark")
    t = ctx.t
    gen_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    try:
        from .version import BUILD_ID as build
    except Exception:  # pragma: no cover
        build = "dev"

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=MARGIN_X, rightMargin=MARGIN_X,
                            topMargin=TOP, bottomMargin=BOTTOM, author="ModelShift",
                            title=_safe(f"ModelShift Migration Report - {report.get('name', '')}"))
    story: List[Any] = []

    # topbar: run name + status
    st = report.get("status", "")
    st_col = t["ok"] if st == "done" else (t["bad"] if st == "failed" else t["partial"])
    status_pill = Pill(ctx, st.upper(), st_col, kind="band", size=7)
    story.append(hrow([ctx.P(esc(report.get("name") or "Untitled run"), "h1"), status_pill],
                      [CW - status_pill.width - 4, status_pill.width + 4], valign="MIDDLE",
                      aligns=["LEFT", "RIGHT"]))
    sub = f"Run details · {esc(report.get('run_id', ''))} · generated {gen_ts}"
    story.append(ctx.P(sub, "muted"))
    if report.get("description"):
        story.append(Spacer(1, 3))
        story.append(ctx.P(esc(one_line(report["description"], 600)), "dim"))
    story.append(Spacer(1, 12))

    cands = report.get("candidates") or []
    if st == "failed" or not cands:
        why = report.get("error") or "No candidate verdicts were produced."
        story.append(banner(ctx, "<b>Run failed.</b> " + esc(why), "err"))
        if report.get("source"):
            story.append(_golden(ctx, report["source"]))
        doc.build(story, onFirstPage=_page_decor(ctx, report, gen_ts, build),
                  onLaterPages=_page_decor(ctx, report, gen_ts, build))
        return buf.getvalue()

    hero = _hero(ctx, report)
    if hero is not None:
        story.append(hero)
    for cv in report.get("caveats") or []:
        story.append(banner(ctx, esc(cv), "warn"))
    errored = sum(c.get("errored") or 0 for c in cands)
    if errored:
        story.append(banner(ctx, f"{errored} candidate call(s) errored and were excluded from scoring.", "warn"))
    story.append(_golden(ctx, report.get("source") or {}))

    story += section_title(ctx, "Side-by-side", "Ranked best-first.")
    story.append(_side_by_side(ctx, cands))

    cols = min(len(cands), 3)
    g = grid([_cand_card(ctx, c, (CW - 10 * (cols - 1)) / cols) for c in cands], CW, cols)
    first_row_h = g.wrap(CW, FRAME_H)[1] / max(1, -(-len(cands) // cols)) if g is not None else 0
    story.append(CondPageBreak(first_row_h + 0.8 * inch))
    story += section_title(ctx, "Results", "One card per candidate - the same verdict cards as the app.")
    if g is not None:
        story.append(g)

    dm = _dimension_matrix(ctx, cands)
    if dm is not None:
        story += section_title(ctx, "Dimension breakdown",
                               "Average score per compatibility dimension, per candidate.")
        story.append(dm)

    fr = _failure_reasons(ctx, cands)
    if fr is not None:
        story += section_title(ctx, "Why candidates lost points", "Most common judge failure reasons.")
        story.append(fr)

    rm = _remediations(ctx, report.get("remediations") or [])
    if rm is not None:
        story += section_title(ctx, "Suggested remediations")
        story.append(rm)

    story += [CondPageBreak(3.2 * inch), Spacer(1, 4), _legend(ctx)]

    # ---- Appendix: Evaluations ----
    story.append(PageBreak())
    story.append(ctx.P("Appendix - Evaluations", "h1"))
    story.append(ctx.P("Every request evaluated for each candidate: the request replayed to both models, the "
                       "original (golden) vs candidate response with models and latency, the six compatibility "
                       "scores, and the LLM judge's assessment - the same drill-down as the app.", "muted"))
    story.append(Spacer(1, 10))
    for c in cands:
        ev = c.get("evaluations") or {}
        rows = ev.get("rows") or []
        total = ev.get("total", len(rows))
        story += section_title(ctx, f"Evaluations · {c['model']} ({total})")
        if ev.get("truncated"):
            story.append(banner(ctx, f"Showing the first {ev.get('limit', len(rows))} of <b>{total}</b> evaluations. "
                                     f"The full list is available in the ModelShift app: Results &gt; "
                                     f"{esc(c['model'])} &gt; View evaluations.", "info"))
        if not rows:
            story.append(ctx.P("No evaluations recorded for this candidate.", "muted"))
            continue
        story.append(_eval_index(ctx, rows))
        for i, r in enumerate(rows, start=1):
            story.append(_eval_card(ctx, i, r, c["model"]))

    decor = _page_decor(ctx, report, gen_ts, build)
    doc.build(story, onFirstPage=decor, onLaterPages=decor)
    return buf.getvalue()
