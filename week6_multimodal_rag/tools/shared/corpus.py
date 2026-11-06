# ---- corpus.py: shared helper code (canonical copy in week6_multimodal_rag/tools/shared/) ----
# A small multimodal document corpus with exact ground truth (Weeks 6-8).
#
# Six fictional company annual reports rendered as real PDFs with reportlab. Each has text sections, a
# quarterly results table, and charts that carry information that is NOT in the text (revenue by region,
# segment shares, headcount trend), so some questions can only be answered from the images. One report is
# "scanned" (pages rasterized, no text layer), so the parser must fall back to OCR.
# Optional extras: the first pages of a few real arXiv papers (vector figures, real tables) and your own PDFs
# from Drive (My Drive/datasets/rag_corpus/*.pdf). Those have no full ground truth; a few known facts are asked.
import io
import json
import random
import re
import shutil
import urllib.request
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np
import pymupdf
from PIL import Image, ImageFilter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

COMPANIES = [
    dict(doc_id="northwind_energy", name="Northwind Energy", ceo="Amara Okafor", hq="Oslo", founded=1998,
         industry="renewable energy", segments=["Wind", "Solar", "Hydro", "Grid services"], regions=["Europe", "North America", "Asia"]),
    dict(doc_id="contoso_health", name="Contoso Health", ceo="Daniel Brooks", hq="Boston", founded=2004,
         industry="digital health", segments=["Diagnostics", "Telemedicine", "Devices"], regions=["North America", "Europe", "Latin America"]),
    dict(doc_id="fabrikam_robotics", name="Fabrikam Robotics", ceo="Mei Lin Zhang", hq="Shenzhen", founded=2011,
         industry="industrial robotics", segments=["Robot arms", "Mobile robots", "Software", "Services"], regions=["Asia", "Europe", "North America"]),
    dict(doc_id="tailspin_air", name="Tailspin Air", ceo="Carlos Mendes", hq="Lisbon", founded=1987,
         industry="regional aviation", segments=["Passenger", "Cargo", "Maintenance"], regions=["Europe", "Africa", "South America"]),
    dict(doc_id="litware_software", name="Litware Software", ceo="Priya Raman", hq="Bengaluru", founded=2009,
         industry="enterprise software", segments=["Cloud", "Licences", "Consulting", "Support"], regions=["Asia", "North America", "Europe"]),
    dict(doc_id="proseware_foods", name="Proseware Foods", ceo="Hannah Keller", hq="Munich", founded=1972,
         industry="packaged foods", segments=["Dairy", "Bakery", "Beverages"], regions=["Europe", "Middle East", "Asia"], scanned=True),
]
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
YEARS = [2021, 2022, 2023, 2024, 2025]
REAL_PAPERS = {  # doc_id: (arXiv id, title, [(question, answer)] answerable from the first pages)
    "resnet": ("1512.03385", "Deep Residual Learning for Image Recognition",
               [("What error rate did the ResNet ensemble achieve on the ImageNet test set?", "3.57%")]),
    "unet": ("1505.04597", "U-Net: Convolutional Networks for Biomedical Image Segmentation",
             [("Which challenge did U-Net win in 2015?", "ISBI cell tracking challenge")]),
    "clip": ("2103.00020", "Learning Transferable Visual Models From Natural Language Supervision",
             [("How many (image, text) pairs were used to pre-train CLIP?", "400 million")]),
    "llava": ("2304.08485", "Visual Instruction Tuning",
              [("Which language-only model was used to generate LLaVA's multimodal instruction-following data?", "GPT-4")]),
    "react": ("2210.03629", "ReAct: Synergizing Reasoning and Acting in Language Models",
              [("On which two knowledge-intensive tasks is ReAct evaluated besides ALFWorld and WebShop?", ["HotpotQA", "Fever"])]),
}
FILLER = [
    "The company continued to invest in its people, its processes and its digital platforms throughout the year.",
    "Customer satisfaction remained high, and the board approved a multi-year efficiency programme.",
    "Sustainability targets were reviewed by an external auditor and published alongside this report.",
    "Supply-chain conditions normalised during the second half, easing pressure on margins.",
]


@dataclass
class ReportFacts:
    doc_id: str
    name: str
    ceo: str
    hq: str
    founded: int
    industry: str
    employees: int
    revenue: list          # Q1..Q4, USD millions
    op_profit: list        # Q1..Q4, USD millions
    segments: dict         # name -> % share (ints summing to 100)
    regions: dict          # name -> USD millions
    growth_target: int     # % for 2026
    headcount: dict | None  # year -> employees (only on some reports)
    scanned: bool = False

    @property
    def total_revenue(self) -> float:
        return round(sum(self.revenue), 1)

    @property
    def total_op_profit(self) -> float:
        return round(sum(self.op_profit), 1)

    @property
    def largest_segment(self) -> str:
        return max(self.segments, key=self.segments.get)

    @property
    def largest_region(self) -> str:
        return max(self.regions, key=self.regions.get)


def _shares_with_clear_max(rng: random.Random, n: int) -> list[int]:
    raw = sorted((rng.uniform(8, 30) for _ in range(n)), reverse=True)
    raw[0] += 12  # make the winner unambiguous on a chart
    pct = [int(round(100 * r / sum(raw))) for r in raw]
    pct[-1] += 100 - sum(pct)
    rng.shuffle(pct)
    return pct


def company_facts(c: dict, idx: int, seed: int) -> ReportFacts:
    rng = random.Random(f"{seed}-{c['doc_id']}")
    base = rng.uniform(80, 600)
    revenue = [round(base * rng.uniform(0.82, 1.25), 1) for _ in QUARTERS]
    op_profit = [round(r * rng.uniform(0.07, 0.22), 1) for r in revenue]
    total = sum(revenue)
    reg_pct = _shares_with_clear_max(rng, len(c["regions"]))
    regions = {r: round(total * p / 100, 1) for r, p in zip(c["regions"], reg_pct)}
    segments = dict(zip(c["segments"], _shares_with_clear_max(rng, len(c["segments"]))))
    employees = rng.randrange(800, 9000, 10)
    headcount = None
    if idx % 2 == 0:
        start = employees * rng.uniform(0.55, 0.8)
        headcount = {y: int(round(start + (employees - start) * i / 4, -1)) for i, y in enumerate(YEARS)}
        headcount[2025] = employees
    return ReportFacts(c["doc_id"], c["name"], c["ceo"], c["hq"], c["founded"], c["industry"], employees, revenue, op_profit,
                       segments, regions, rng.choice([4, 6, 8, 10, 12, 15]), headcount, c.get("scanned", False))


def chart_png(kind: str, f: ReportFacts, style: int = 0) -> bytes:
    """Render one chart. style=0 is the report version; style=1 is a restyled copy used as a query image."""
    palette = [plt.cm.tab10.colors, plt.cm.Set2.colors][style % 2]
    figsize = {"bar": (6, 3.5), "pie": (4.5, 4.5), "line": (6, 3)}[kind]
    if style:
        figsize = (figsize[0] * 0.8, figsize[1] * 0.8)
    fig, ax = plt.subplots(figsize=figsize)
    if kind == "bar":
        names, vals = list(f.regions), list(f.regions.values())
        bars = ax.bar(names, vals, color=palette[:len(vals)])
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in vals], fontsize=9)
        ax.set_title("Revenue by region, 2025 (USD millions)" if not style else "Regional revenue")
        ax.set_ylim(0, max(vals) * 1.2)
    elif kind == "pie":
        ax.pie(list(f.segments.values()), labels=list(f.segments), autopct="%d%%", colors=palette, startangle=90 + 40 * style)
        ax.set_title("Revenue share by segment, 2025" if not style else "Segment mix")
    else:
        ys, vs = list(f.headcount), list(f.headcount.values())
        ax.plot(ys, vs, marker="o", color=palette[0])
        for y, v in zip(ys, vs):
            ax.annotate(f"{v:,}", (y, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        ax.set_xticks(ys)
        ax.set_title("Employees 2021-2025")
        ax.set_ylim(0, max(vs) * 1.25)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _rl_image(png: bytes, width_cm: float) -> RLImage:
    w, h = Image.open(io.BytesIO(png)).size
    return RLImage(io.BytesIO(png), width=width_cm * cm, height=width_cm * cm * h / w)


def build_report_pdf(f: ReportFacts, path: Path) -> None:
    st = getSampleStyleSheet()
    cap = st["Italic"]
    body = st["BodyText"]
    rng = random.Random(f.doc_id)
    story = [
        Paragraph(f"{f.name} Annual Report 2025", st["Title"]),
        Paragraph("Company overview", st["Heading2"]),
        Paragraph(f"{f.name} is a {f.industry} company headquartered in {f.hq}, founded in {f.founded}. Chief executive "
                  f"officer {f.ceo} leads a workforce of {f.employees:,} employees across its operations.", body),
        Paragraph(" ".join(rng.sample(FILLER, 2)), body),
        Paragraph("Strategy", st["Heading2"]),
        Paragraph(f"In 2025 {f.name} focused on operational efficiency and on expanding its {f.industry} portfolio. "
                  + " ".join(rng.sample(FILLER, 2)), body),
        PageBreak(),
        Paragraph("Financial results", st["Heading2"]),
        Paragraph(f"Full-year revenue reached USD {f.total_revenue:,.1f} million and operating profit USD {f.total_op_profit:,.1f} "
                  "million. Table 1 shows the quarterly results; Figure 1 breaks revenue down by region.", body),
        Spacer(1, 0.3 * cm),
    ]
    rows = [["Quarter", "Revenue", "Operating profit"]] + [[q, f"{r:,.1f}", f"{p:,.1f}"] for q, r, p in zip(QUARTERS, f.revenue, f.op_profit)]
    rows.append(["Total", f"{f.total_revenue:,.1f}", f"{f.total_op_profit:,.1f}"])
    table = Table(rows, colWidths=[4 * cm, 4 * cm, 4.5 * cm])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                               ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                               ("ALIGN", (1, 0), (-1, -1), "RIGHT")]))
    story += [table, Paragraph("Table 1: Quarterly results 2025 (USD millions).", cap), Spacer(1, 0.6 * cm),
              _rl_image(chart_png("bar", f), 12), Paragraph("Figure 1: Revenue by region, 2025.", cap), PageBreak(),
              Paragraph("Business segments", st["Heading2"]),
              Paragraph(f"{f.name} operates {len(f.segments)} business segments: {', '.join(f.segments)}. "
                        "Figure 2 shows how 2025 revenue was split between them.", body),
              _rl_image(chart_png("pie", f), 8), Paragraph("Figure 2: Revenue share by segment, 2025.", cap)]
    if f.headcount:
        story += [Spacer(1, 0.3 * cm), _rl_image(chart_png("line", f), 11), Paragraph("Figure 3: Employees 2021-2025.", cap)]
    story += [Paragraph("Outlook", st["Heading2"]),
              Paragraph(f"Management targets revenue growth of {f.growth_target}% in 2026, supported by new contracts and a "
                        "disciplined cost base. " + rng.choice(FILLER), body)]
    SimpleDocTemplate(str(path), pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm, title=f"{f.name} Annual Report 2025").build(story)


def rasterize_pdf(path: Path, dpi: int = 150, seed: int = 0) -> None:
    """Turn a PDF into a 'scanned' one: page images only (slight skew, blur, noise), no text layer."""
    rng = np.random.default_rng(seed)
    pages = []
    for page in pymupdf.open(path):
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        img = img.rotate(float(rng.uniform(-0.8, 0.8)), resample=Image.BICUBIC, fillcolor="white").filter(ImageFilter.GaussianBlur(0.6))
        arr = np.asarray(img).astype(np.float32) + rng.normal(0, 6, (img.height, img.width, 1))
        pages.append(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)))
    pages[0].save(path, format="PDF", save_all=True, append_images=pages[1:], resolution=dpi)


def locate_pages(path: Path) -> dict:
    """Page number of each labelled element, found in the rendered PDF (before any rasterization)."""
    marks = {"overview": "Company overview", "table1": "Table 1:", "fig1": "Figure 1:", "fig2": "Figure 2:",
             "fig3": "Figure 3:", "outlook": "Outlook"}
    out = {}
    for pno, page in enumerate(pymupdf.open(path), 1):
        text = page.get_text()
        for key, needle in marks.items():
            if key not in out and needle in text:
                out[key] = pno
    return out


def download_paper(arxiv_id: str, path: Path, max_pages: int) -> bool:
    """First pages of an arXiv paper. Returns False (and the corpus simply skips it) when offline."""
    try:
        req = urllib.request.Request(f"https://arxiv.org/pdf/{arxiv_id}", headers={"User-Agent": "Mozilla/5.0 (course notebook)"})
        data = urllib.request.urlopen(req, timeout=60).read()
        doc = pymupdf.open(stream=data, filetype="pdf")
        doc.select(list(range(min(max_pages, doc.page_count))))
        doc.save(path)
        return True
    except Exception as exc:
        print(f"skipping arXiv {arxiv_id}: {exc!r}"[:160])
        return False


def _qa(qid, question, answer, qtype, sources, number=None, image_path=None) -> dict:
    return {"id": qid, "question": question, "answer": answer, "number": number, "qtype": qtype,
            "sources": [list(s) for s in sources], "image_path": image_path}


def make_qa(facts: list[ReportFacts], pages: dict, img_dir: Path, papers: list[str], seed: int) -> list[dict]:
    """Questions with expected answers and expected sources (doc_id, page, element type)."""
    rng = random.Random(seed)
    qa = []
    for f in facts:
        p, d = pages[f.doc_id], f.doc_id
        q = rng.randrange(4)
        seg = rng.choice(list(f.segments))
        qa += [
            _qa(f"{d}-ceo", f"Who is the chief executive officer of {f.name}?", f.ceo, "text", [(d, p["overview"], "text")]),
            _qa(f"{d}-hq", f"In which city is {f.name} headquartered?", f.hq, "text", [(d, p["overview"], "text")]),
            _qa(f"{d}-employees", f"How many employees does {f.name} have?", f"{f.employees:,}", "text", [(d, p["overview"], "text")], f.employees),
            _qa(f"{d}-growth", f"What revenue growth does {f.name} target for 2026?", f"{f.growth_target}%", "text",
                [(d, p["outlook"], "text")], f.growth_target),
            _qa(f"{d}-rev-{QUARTERS[q]}", f"What was {f.name}'s revenue in {QUARTERS[q]} 2025?", f"{f.revenue[q]:,.1f}", "table",
                [(d, p["table1"], "table")], f.revenue[q]),
            _qa(f"{d}-op-{QUARTERS[3 - q]}", f"What was {f.name}'s operating profit in {QUARTERS[3 - q]} 2025?",
                f"{f.op_profit[3 - q]:,.1f}", "table", [(d, p["table1"], "table")], f.op_profit[3 - q]),
            _qa(f"{d}-region", f"Which region generated the most revenue for {f.name} in 2025?", f.largest_region, "figure",
                [(d, p["fig1"], "figure")]),
            _qa(f"{d}-segment", f"Which business segment had the largest share of {f.name}'s 2025 revenue?", f.largest_segment, "figure",
                [(d, p["fig2"], "figure")]),
            _qa(f"{d}-segshare", f"What percentage of {f.name}'s 2025 revenue came from its {seg} segment?", f"{f.segments[seg]}%",
                "figure", [(d, p["fig2"], "figure")], f.segments[seg]),
        ]
        if f.headcount:
            qa.append(_qa(f"{d}-hc2021", f"How many employees did {f.name} have in 2021 according to its headcount chart?",
                          f"{f.headcount[2021]:,}", "figure", [(d, p["fig3"], "figure")], f.headcount[2021]))
        kind = rng.choice(["pie", "bar"])
        img_path = img_dir / f"query_{d}_{kind}.png"
        img_path.write_bytes(chart_png(kind, f, style=1))
        qa.append(_qa(f"{d}-img-{kind}", "Which company's annual report contains a chart showing the same data as this image?", f.name,
                      "text+image", [(d, p["fig2" if kind == "pie" else "fig1"], "figure")], image_path=str(img_path)))
    for f in rng.sample(facts, 3):
        qa.append(_qa(f"{f.doc_id}-unans-2019", f"What was {f.name}'s revenue in 2019?", None, "unanswerable", []))
        qa.append(_qa(f"{f.doc_id}-unans-cfo", f"Who is the chief financial officer of {f.name}?", None, "unanswerable", []))
    for _ in range(4):  # multi-hop (Week 7): two documents or two elements
        a, b = rng.sample(facts, 2)
        win = a if a.revenue[3] > b.revenue[3] else b
        qa.append(_qa(f"mh-q4-{a.doc_id}-{b.doc_id}", f"Which company had the higher revenue in Q4 2025: {a.name} or {b.name}?", win.name,
                      "multi-hop", [(a.doc_id, pages[a.doc_id]["table1"], "table"), (b.doc_id, pages[b.doc_id]["table1"], "table")]))
        win = a if a.employees > b.employees else b
        qa.append(_qa(f"mh-emp-{a.doc_id}-{b.doc_id}", f"Which company has more employees, {a.name} or {b.name}, and who is its CEO?",
                      [win.name, win.ceo], "multi-hop", [(a.doc_id, pages[a.doc_id]["overview"], "text"), (b.doc_id, pages[b.doc_id]["overview"], "text")]))
    for f in rng.sample(facts, 3):
        qa.append(_qa(f"mh-regseg-{f.doc_id}", f"For {f.name}, name its largest region by revenue and its largest business segment.",
                      [f.largest_region, f.largest_segment], "multi-hop",
                      [(f.doc_id, pages[f.doc_id]["fig1"], "figure"), (f.doc_id, pages[f.doc_id]["fig2"], "figure")]))
    best = max(facts, key=lambda x: x.total_revenue)
    qa.append(_qa("mh-top-revenue-ceo", "Who is the CEO of the company with the highest full-year 2025 revenue among all the annual reports?",
                  best.ceo, "multi-hop", [(f.doc_id, pages[f.doc_id]["table1"], "table") for f in facts] + [(best.doc_id, pages[best.doc_id]["overview"], "text")]))
    for pid in papers:
        for i, (question, answer) in enumerate(REAL_PAPERS[pid][2]):
            qa.append(_qa(f"{pid}-q{i}", question, answer, "text", [(pid, 1, "text")]))
    return qa


def build_corpus(out_dir: Path, seed: int = 7, real_papers: bool = True, paper_pages: int = 4, user_pdfs: bool = True) -> tuple[list, dict, list]:
    """Write every PDF to out_dir/pdfs and return (docs, facts by doc_id, qa). Deterministic for a seed."""
    pdf_dir, img_dir = out_dir / "pdfs", out_dir / "query_images"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    facts = [company_facts(c, i, seed) for i, c in enumerate(COMPANIES)]
    docs, pages = [], {}
    for f in facts:
        path = pdf_dir / f"{f.doc_id}.pdf"
        build_report_pdf(f, path)
        pages[f.doc_id] = locate_pages(path)
        if f.scanned:
            rasterize_pdf(path, seed=seed)
        docs.append({"doc_id": f.doc_id, "path": str(path), "title": f"{f.name} Annual Report 2025",
                     "kind": "scanned report" if f.scanned else "report"})
    papers = []
    if real_papers:
        for pid, (arxiv_id, title, _) in REAL_PAPERS.items():
            path = pdf_dir / f"{pid}.pdf"
            if path.exists() or download_paper(arxiv_id, path, paper_pages):
                papers.append(pid)
                docs.append({"doc_id": pid, "path": str(path), "title": title, "kind": "arXiv paper"})
    drive = mount_drive() if user_pdfs else None
    if drive is not None and (drive / "datasets" / "rag_corpus").is_dir():
        for src in sorted((drive / "datasets" / "rag_corpus").glob("*.pdf"))[:10]:
            doc_id = re.sub(r"[^a-z0-9]+", "_", src.stem.lower()).strip("_")
            dst = pdf_dir / f"{doc_id}.pdf"
            shutil.copy(src, dst)
            docs.append({"doc_id": doc_id, "path": str(dst), "title": src.stem, "kind": "your PDF"})
    qa = make_qa(facts, pages, img_dir, papers, seed)
    (out_dir / "corpus.json").write_text(json.dumps({"docs": docs, "facts": {f.doc_id: asdict(f) for f in facts}, "pages": pages,
                                                     "qa": qa}, indent=1, default=str), encoding="utf-8")
    return docs, {f.doc_id: f for f in facts}, qa
