"""PDF report (A4, one page) built with fpdf2. Pure function: build_report(case) -> bytes."""
import io
from fpdf import FPDF
from PIL import Image
import content

GREEN, INK, MUTED = (47, 93, 74), (31, 45, 38), (79, 98, 88)
IVORY, LINE, MINT, AMBER = (251, 249, 238), (207, 221, 200), (238, 246, 234), (253, 243, 214)

def _a(s) -> str:
    for a, b in {"–": "-", "—": "-", "’": "'", "‘": "'", "“": '"', "”": '"', "·": "-", "…": "..."}.items():
        s = str(s).replace(a, b)
    return s.encode("latin-1", "replace").decode("latin-1")

class _PDF(FPDF):
    def footer(self):
        self.set_draw_color(*LINE); self.line(12, 272, 198, 272)
        self.set_font("Helvetica", "", 7.5); self.set_text_color(*MUTED)
        self.set_xy(12, 274); self.multi_cell(186, 3.4, _a(content.DISCLAIMER))
        self.set_xy(12, 287); self.cell(186, 4, f"DermaScan  -  Page {self.page_no()} of {{nb}}", align="R")

def _png(img: Image.Image) -> io.BytesIO:
    b = io.BytesIO(); img.convert("RGB").save(b, "PNG"); b.seek(0); return b

def _fit(pdf, img, x, y, w, h):
    s = min(w / img.width, h / img.height); iw, ih = img.width * s, img.height * s
    pdf.image(_png(img), x + (w - iw) / 2, y + (h - ih) / 2, iw, ih)

def _sec(pdf, title, y):
    pdf.set_xy(12, y); pdf.set_font("Helvetica", "B", 12); pdf.set_text_color(*GREEN)
    pdf.cell(186, 6, _a(title))

def _box(pdf, x, y, w, h, label, value, fill=None):
    pdf.set_draw_color(*LINE); pdf.set_fill_color(*(fill or (252, 253, 249)))
    pdf.rect(x, y, w, h, style="DF")
    pdf.set_xy(x + 3, y + 2); pdf.set_font("Helvetica", "B", 6.5); pdf.set_text_color(*MUTED); pdf.cell(w - 6, 3, _a(label))
    pdf.set_xy(x + 3, y + 6.5); pdf.set_font("Helvetica", "", 10.5); pdf.set_text_color(*INK); pdf.cell(w - 6, 5, _a(value))

def build_report(case: dict) -> bytes:
    p, m = case["pred"], case["meta"]
    info = content.UNSURE if p.is_unsure else content.CLASSES[p.label]
    pdf = _PDF("P", "mm", "A4"); pdf.alias_nb_pages(); pdf.set_auto_page_break(True, 26); pdf.add_page()

    # 1. header
    pdf.set_fill_color(*IVORY); pdf.rect(0, 0, 210, 28, style="F")
    pdf.set_draw_color(*GREEN); pdf.set_line_width(0.6); pdf.line(0, 28, 210, 28); pdf.set_line_width(0.2)
    pdf.set_line_width(0.7); pdf.ellipse(12, 8, 11, 11); pdf.set_line_width(0.2)
    pdf.set_fill_color(*GREEN); pdf.ellipse(16, 12, 3, 3, style="F")
    pdf.set_xy(27, 7); pdf.set_font("Helvetica", "B", 18); pdf.set_text_color(*GREEN); pdf.cell(80, 8, "DermaScan")
    pdf.set_xy(27, 15.5); pdf.set_font("Helvetica", "", 9); pdf.set_text_color(*MUTED); pdf.cell(80, 5, "Skin Lesion Analysis Report")
    pdf.set_xy(120, 6.5); pdf.set_font("Helvetica", "", 8.5); pdf.set_text_color(*INK)
    pdf.multi_cell(78, 4.6, _a(f"Report ID: {case['report_id']}\nDate: {case['created']:%d %b %Y, %H:%M}\nClinician: {case['clinician']}"), align="R")

    # 2. patient details
    _sec(pdf, "Patient details", 33)
    for x, w, lab, val in [(12, 40, "CASE ID", case["case_no"]), (54, 40, "AGE", "Not provided" if m["age"] is None else m["age"]),
                           (96, 40, "SEX", m["sex"]), (138, 60, "LESION SITE", m["site"])]:
        _box(pdf, x, 40, w, 14, lab, val)

    # 3. image + result
    pdf.set_font("Helvetica", "B", 12); pdf.set_text_color(*GREEN)
    pdf.set_xy(12, 60); pdf.cell(60, 6, "Uploaded image")
    pdf.set_xy(78, 60); pdf.cell(120, 6, "Classification result")
    pdf.set_fill_color(228, 238, 221); pdf.rect(12, 67, 60, 46, style="F"); _fit(pdf, case["image"], 12, 67, 60, 46)
    pdf.set_draw_color(*LINE); pdf.set_fill_color(*(AMBER if p.is_unsure else MINT)); pdf.rect(78, 67, 120, 46, style="DF")
    pdf.set_xy(84, 71); pdf.set_font("Helvetica", "B", 7); pdf.set_text_color(*MUTED); pdf.cell(100, 4, "PREDICTED CLASS")
    pdf.set_xy(84, 77); pdf.set_font("Helvetica", "B", 17); pdf.set_text_color(*GREEN)
    pdf.multi_cell(108, 7.5, _a("Not sure - refer to dermatologist" if p.is_unsure else f"{info['name']} ({p.label})"))
    pdf.set_xy(84, 96); pdf.set_font("Helvetica", "", 10); pdf.set_text_color(*INK)
    pdf.cell(108, 5, f"Confidence: {p.confidence:.0%}   |   Risk group: {'-' if p.is_unsure else info['risk']}")
    pdf.set_xy(84, 103); pdf.set_font("Helvetica", "", 8); pdf.set_text_color(*MUTED)
    pdf.multi_cell(108, 4, "Below the tuned confidence threshold the system does not name a class and recommends dermatologist review." if p.is_unsure
                   else "Decision: confirm clinically; refer to a dermatologist where indicated.")

    # 4. model views
    _sec(pdf, "Model views", 119)
    for i, (img, cap) in enumerate([(p.mask, "Segmentation mask"), (p.cam, "Grad-CAM heatmap"), (p.overlay, "Mask + Grad-CAM overlay")]):
        x = 12 + i * 63
        pdf.set_fill_color(228, 238, 221); pdf.rect(x, 126, 60, 42, style="F"); _fit(pdf, img, x, 126, 60, 42)
        pdf.set_xy(x, 169); pdf.set_font("Helvetica", "", 8); pdf.set_text_color(*INK); pdf.cell(60, 4, cap, align="C")

    # 5. scores
    _sec(pdf, "Scores", 177)
    _box(pdf, 12, 184, 58, 17, "ATTENTION-LESION OVERLAP (IoU)", f"{p.iou:.2f}")
    _box(pdf, 74, 184, 58, 17, "CLASSIFICATION CONFIDENCE", f"{p.confidence:.0%}")
    pdf.set_xy(136, 184); pdf.set_font("Helvetica", "", 7.5); pdf.set_text_color(*MUTED)
    pdf.multi_cell(62, 3.6, align="L", text="Overlap shows how much of the model's attention falls on the segmented lesion. Higher means the model focused on the lesion. It is not a measure of segmentation accuracy.")

    # 6. notes
    _sec(pdf, "About this finding" + ("" if p.is_unsure else f" - {info['name']}"), 207)
    pdf.set_xy(12, 214); pdf.set_font("Helvetica", "", 9.5); pdf.set_text_color(*INK)
    pdf.multi_cell(88, 4.8, _a(info["about"]), align="L")
    pdf.set_xy(108, 214); pdf.set_font("Helvetica", "B", 7); pdf.set_text_color(*MUTED); pdf.cell(90, 4, "SUGGESTED NEXT STEPS")
    pdf.set_xy(108, 219); pdf.set_font("Helvetica", "", 9.5); pdf.set_text_color(*INK)
    for s in info["steps"]:
        pdf.set_x(108); pdf.multi_cell(90, 4.8, _a("-  " + s))
    return bytes(pdf.output())