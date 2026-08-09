#!/usr/bin/env python3
"""Convert Obsidian English thesis chapters (read-only) into Leeds LaTeX template chapters.

Does NOT modify files under the Obsidian vault. Writes only to Project/template/.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

OBSIDIAN = Path("/Users/chienchen/Desktop/ObsidianNote/英文專題")
OUT = Path("/Users/chienchen/Coach_ChatBot/Project/template/Chapters")
CONFIG = Path("/Users/chienchen/Coach_ChatBot/Project/template/config.tex")

CHAPTER_META = {
    1: ("Introduction", "ch:intro"),
    2: ("Background Research", "ch:background"),
    3: ("Software Requirements and System Design", "ch:design"),
    4: ("Software Implementation", "ch:implementation"),
    5: ("Software Testing and Evaluation", "ch:evaluation"),
    6: ("Conclusions and Future Work", "ch:conclusion"),
}

# (surname_fragment_lower, year) -> bib key. More specific first where needed.
CITE_MAP: dict[tuple[str, str], str] = {
    ("asgari", "2025"): "asgari2025clinical",
    ("azimi", "2025"): "azimi2025",
    ("bedi", "2025"): "bedi2025",
    ("chen", "2021"): "chen2021dialogsum",
    ("chew", "2022"): "chew2022chatbots",
    ("european parliament", "2016"): "gdpr2016",
    ("european union", "2016"): "gdpr2016",
    ("feng", "2022"): "feng2022dialsum",
    ("frantar", "2023"): "frantar2023gptq",
    ("gliwa", "2019"): "gliwa2019",
    ("guo", "2025"): "guo2025deepseekr1",
    ("haltaufderheide", "2024"): "haltaufderheide2024",
    ("ranisch", "2024"): "haltaufderheide2024",
    ("karpukhin", "2020"): "karpukhin2020dpr",
    ("kleppmann", "2019"): "kleppmann2019localfirst",
    ("laranjo", "2018"): "laranjo2018agents",
    ("lewis", "2020"): "lewis2020rag",
    ("li", "2013"): "li2013phr",
    ("li", "2024"): "li2024mobile",
    ("li", "2025"): "li2025numeracy",
    ("lin", "2024"): "lin2024awq",
    ("madden", "2016"): "madden2016ree",
    ("maharana", "2024"): "maharana2024locomo",
    ("maynez", "2020"): "maynez2020faithfulness",
    ("mifflin", "1990"): "mifflin1990ree",
    ("milne-ives", "2020"): "milneives2020",
    ("milne ives", "2020"): "milneives2020",
    ("nuttall", "2015"): "nuttall2015bmi",
    ("o'hara", "2026"): "ohara2026nutrition",
    ("o’hara", "2026"): "ohara2026nutrition",
    ("ohara", "2026"): "ohara2026nutrition",
    ("oh", "2021"): "oh2021chatbots",
    ("park", "2023"): "park2023generative",
    ("ponzo", "2024a"): "ponzo2024a",
    ("ponzo", "2024b"): "ponzo2024b",
    ("ponzo", "2024"): "ponzo2024a",
    ("reimers", "2019"): "reimers2019sbert",
    ("gurevych", "2019"): "reimers2019sbert",
    ("saad-falcon", "2024"): "saadfalcon2024ares",
    ("seo", "2025"): "seo2025",
    ("touvron", "2023"): "touvron2023llama2",
    ("wang", "2023"): "wang2023longmem",
    ("wang", "2025"): "wang2025recursive",
    ("world health organization", "2021"): "who2021aiethics",
    ("who", "2021"): "who2021aiethics",
    ("zhang", "2025"): "zhang2025memorysurvey",
    ("zhao", "2023"): "zhao2023llmsurvey",
    ("zhong", "2024"): "zhong2024memorybank",
}

# Obsidian wiki-image basename -> thesis figure filename (no path)
WIKI_IMAGE_MAP = {
    "截圖 2026-07-26 20.47.44 1.png": "fig_4_2_settings_status.png",
    "截圖 2026-07-26 20.48.01.png": "fig_4_5_memory_viewer.png",
    "截圖 2026-07-26 21.26.19.png": "fig_4_7_rag_source_chips.png",
    "截圖 2026-07-26 20.47.13.png": "fig_4_8_meal_plan.png",
    "截圖 2026-07-27 16.14.23.png": "fig_4_9_food_choice.png",
    "截圖 2026-07-26 20.46.34.png": "fig_4_10_dashboard.png",
    "截圖 2026-07-26 21.26.19 1.png": "fig_4_11_mobile.png",
}

# Caption keyword -> existing PNG for mermaid blocks
MERMAID_FIGURE_MAP = [
    (re.compile(r"overall system architecture|implementation architecture", re.I),
     "fig_2_1_production_architecture.png", "Overall system architecture."),
    (re.compile(r"memory lifecycle|memory implementation flow|cross-session memory", re.I),
     "fig_2_4_memory_lifecycle.png", "Cross-session memory lifecycle."),
    (re.compile(r"eval.*demo|data and prompt flow", re.I),
     "fig_2_2_eval_vs_demo_paths.png", "Data and prompt flow / evaluation paths."),
    (re.compile(r"frontend navigation|user interface", re.I),
     "fig_2_3_frontend_navigation_flow.png", "Frontend navigation flow."),
]

SECTION_LABELS = {
    "project aim": "sec:aim",
    "objectives": "sec:objectives",
    "deliverables": "sec:deliverables",
    "ethical, legal, and social issues": "sec:ethics",
    "ethical issues": "sec:ethics-ethical",
    "legal issues": "sec:ethics-legal",
    "social issues": "sec:ethics-social",
    "literature review": "sec:litreview",
    "methods and technologies": "sec:methods-tech",
    "method selection": "sec:choice",
    "software requirements": "sec:requirements",
    "system design": "sec:system-design",
    "evaluation overview": "sec:eval-overview",
    "evaluation methodology": "sec:eval-method",
    "software testing": "sec:software-testing",
    "cross-session memory evaluation": "sec:controlled-eval",
    "long-context stress testing": "sec:long-context",
    "rag retrieval evaluation": "sec:rag-eval",
    "deployment configuration evaluation": "sec:deploy-eval",
    "persona-based walkthrough evaluation": "sec:persona-eval",
    "evaluation summary": "sec:eval-summary",
}


def strip_refs(text: str) -> str:
    return re.split(
        r"(?im)^#+[^\n]*references[^\n]*\n",
        text,
        maxsplit=1,
    )[0]


def clean_heading(h: str) -> str:
    h = re.sub(r"^#+\s*", "", h)
    h = h.replace("**", "")
    # Drop leading numbering like 1.1. / 2.1.8 / 4.12
    h = re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", h)
    # Drop Chinese parenthetical notes
    h = re.sub(r"（[^）]*）", "", h)
    h = re.sub(r"\([^)]*SMART[^)]*\)", "", h, flags=re.I)
    return h.strip(" :")


def resolve_cite(author_chunk: str, year: str) -> str | None:
    raw = author_chunk.strip()
    low = raw.lower().replace("’", "'")
    low = re.sub(r"^the\s+", "", low)

    # Prefer longest matching fragment with exact year (incl. 2024a)
    candidates: list[tuple[int, str]] = []
    for (frag, y), key in CITE_MAP.items():
        if frag in low and y == year:
            candidates.append((len(frag), key))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]

    # Fall back: same calendar year without letter (2024a -> 2024)
    y4 = year[:4]
    candidates = []
    for (frag, y), key in CITE_MAP.items():
        if frag in low and y[:4] == y4 and len(y) == 4:
            candidates.append((len(frag), key))
    if candidates:
        candidates.sort(reverse=True)
        return candidates[0][1]
    return None


def parse_paren_cites(inner: str) -> list[str]:
    """Parse 'Chew, 2022; O’Hara et al., 2026' into bib keys."""
    keys: list[str] = []
    # Split on ; but keep a,b suffixes with author
    parts = re.split(r";\s*", inner)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Ponzo et al., 2024a
        m = re.match(
            r"(.+?),?\s+(\d{4}[a-z]?)$",
            part,
        )
        if not m:
            # maybe "2024a, 2024b" after author already consumed - skip orphan
            if re.match(r"^\d{4}[a-z]?(?:,\s*\d{4}[a-z]?)*$", part):
                continue
            continue
        author, year = m.group(1), m.group(2)
        # Handle "Ponzo et al., 2024a" already; also "Ponzo et al., 2024a, 2024b"
        # If author contains multiple years incorrectly:
        multi = re.findall(r"\d{4}[a-z]?", part)
        if len(multi) > 1 and "ponzo" in author.lower():
            for y in multi:
                k = resolve_cite("Ponzo", y)
                if k and k not in keys:
                    keys.append(k)
            continue
        k = resolve_cite(author, year)
        if k and k not in keys:
            keys.append(k)
        elif not k:
            keys.append(f"TODO:{author}:{year}")
    return keys


def escape_tex(s: str) -> str:
    """Escape LaTeX specials outside math; preserve already-inserted cite commands."""
    # Temporarily protect cite commands
    cites: list[str] = []

    def save_cite(m: re.Match) -> str:
        cites.append(m.group(0))
        return f"«CITE{len(cites) - 1}»"

    s = re.sub(r"\\cite[tp]?\{[^}]+\}", save_cite, s)

    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in s:
        out.append(repl.get(ch, ch))
    s = "".join(out)
    # Unicode punctuation
    s = s.replace("’", "'").replace("‘", "'").replace("“", "``").replace("”", "''")
    s = s.replace("–", "--").replace("—", "---").replace("…", "...")
    s = s.replace("×", r"$\times$").replace("→", r"$\rightarrow$")
    for i, c in enumerate(cites):
        s = s.replace(f"«CITE{i}»", c)
    return s


def format_inline(text: str) -> str:
    """Convert markdown inline formatting and citations."""
    placeholders: list[str] = []

    def protect(s: str) -> str:
        placeholders.append(s)
        return f"«PH{len(placeholders) - 1}»"

    def expand_ph(s: str) -> str:
        return re.sub(r"«PH(\d+)»", lambda m: placeholders[int(m.group(1))], s)

    def escape_keeping_cmds(s: str) -> str:
        """Escape plain text while preserving nested LaTeX command groups."""
        cmd_re = re.compile(r"\\(?:texttt|cite[tp]?|emph|textbf)\{")
        out = []
        i = 0
        while i < len(s):
            m = cmd_re.search(s, i)
            if not m:
                out.append(escape_tex(s[i:]))
                break
            out.append(escape_tex(s[i:m.start()]))
            # find matching closing brace
            j = m.end()
            depth = 1
            while j < len(s) and depth:
                if s[j] == "{":
                    depth += 1
                elif s[j] == "}":
                    depth -= 1
                j += 1
            out.append(s[m.start():j])
            i = j
        return "".join(out)

    text = re.sub(
        r"`([^`]+)`",
        lambda m: protect(f"\\texttt{{{escape_tex(m.group(1))}}}"),
        text,
    )

    def narr_multi(m: re.Match) -> str:
        author = m.group(1)
        years = re.findall(r"\d{4}[a-z]?", m.group(2))
        keys = []
        for y in years:
            k = resolve_cite(author, y)
            if k and k not in keys:
                keys.append(k)
        if not keys:
            return m.group(0)
        return protect(f"\\citet{{{','.join(keys)}}}")

    text = re.sub(
        r"((?:The\s+)?[A-Z][A-Za-z\-']+(?:\s+(?:and\s+[A-Z][A-Za-z\-']+|et\s+al\.?))*)\s*\((\d{4}[a-z]?(?:\s*,\s*\d{4}[a-z]?)+)\)",
        narr_multi,
        text,
    )

    def narr(m: re.Match) -> str:
        key = resolve_cite(m.group(1), m.group(2))
        if not key:
            return m.group(0)
        return protect(f"\\citet{{{key}}}")

    text = re.sub(
        r"((?:The\s+)?(?:World Health Organization|[A-Z][A-Za-z\-']+(?:\s+(?:and\s+[A-Z][A-Za-z\-']+|et\s+al\.?))*))\s*\((\d{4}[a-z]?)\)",
        narr,
        text,
    )

    def paren(m: re.Match) -> str:
        inner = m.group(1)
        if not re.search(r"\d{4}", inner) or not re.search(r"[A-Za-z]", inner):
            return m.group(0)
        keys = [k for k in parse_paren_cites(inner) if not k.startswith("TODO:")]
        if not keys:
            return m.group(0)
        return protect(f"\\cite{{{','.join(keys)}}}")

    text = re.sub(r"\(([^()\n]{3,200}?)\)", paren, text)

    def bold(m: re.Match) -> str:
        return protect("\\textbf{" + escape_keeping_cmds(expand_ph(m.group(1))) + "}")

    text = re.sub(r"\*\*([^*]+)\*\*", bold, text)
    text = re.sub(r"\*\*([^*]+)$", bold, text)

    def ital(m: re.Match) -> str:
        return protect("\\emph{" + escape_keeping_cmds(expand_ph(m.group(1))) + "}")

    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", ital, text)

    prev = None
    while prev != text:
        prev = text
        text = expand_ph(text)
    return escape_keeping_cmds(text)



def md_table_to_latex(rows: list[list[str]], caption: str, label: str) -> str:
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:]
    ncols = len(header)
    # Build column spec
    if ncols <= 2:
        colspec = "@{}p{1.6cm}X@{}" if ncols == 2 else "@{}X@{}"
        env = "tabularx"
        width = "{\\textwidth}"
    elif ncols <= 4:
        colspec = "@{}" + "X" * ncols + "@{}"
        env = "tabularx"
        width = "{\\textwidth}"
    else:
        colspec = "@{}" + "l" * ncols + "@{}"
        env = "tabular"
        width = ""

    def cell(c: str) -> str:
        c = c.strip()
        c = re.sub(r"^\*\*|\*\*$", "", c)
        return format_inline(c)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{escape_tex(caption)}}}",
        f"\\label{{{label}}}",
        "\\small",
    ]
    if env == "tabularx":
        lines.append(f"\\begin{{tabularx}}{width}{{{colspec}}}")
    else:
        lines.append("\\begin{adjustbox}{max width=\\textwidth}")
        lines.append(f"\\begin{{tabular}}{{{colspec}}}")

    lines.append("\\toprule")
    lines.append(" & ".join(f"\\textbf{{{cell(h)}}}" for h in header) + " \\\\")
    lines.append("\\midrule")
    for r in body:
        # pad
        while len(r) < ncols:
            r.append("")
        lines.append(" & ".join(cell(c) for c in r[:ncols]) + " \\\\")
    lines.append("\\bottomrule")
    lines.append(f"\\end{{{env}}}")
    if env != "tabularx":
        lines.append("\\end{adjustbox}")
    lines.append("\\end{table}")
    lines.append("")
    return "\n".join(lines)


def parse_table_block(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    rows = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        row = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        if all(re.match(r"^:?-+:?$", c) for c in row):
            i += 1
            continue
        rows.append(row)
        i += 1
    return rows, i


def look_caption(prev_lines: list[str]) -> str:
    for line in reversed(prev_lines[-5:]):
        m = re.match(r"\*\*Table\s+[\d.]+\.?\s*(.*?)\*\*", line.strip())
        if m:
            return m.group(1).strip() or "Table"
        m = re.match(r"\*\*Figure\s+[\d.]+\.?\s*(.*?)\*\*", line.strip())
        if m:
            return m.group(1).strip()
    return ""


def convert_chapter(n: int) -> str:
    raw = (OBSIDIAN / f"chapter{n}.md").read_text(encoding="utf-8")
    raw = strip_refs(raw)
    title, chlabel = CHAPTER_META[n]
    lines = raw.splitlines()

    out: list[str] = [f"\\chapter{{{title}}}", f"\\label{{{chlabel}}}", ""]
    i = 0
    table_count = 0
    fig_count = 0
    pending_skip_caption = False
    last_figure_caption = ""

    # Skip first chapter heading line
    while i < len(lines) and (not lines[i].strip() or lines[i].strip().startswith("#")):
        if lines[i].strip().startswith("#"):
            i += 1
            break
        i += 1

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip standalone table/figure caption lines (consumed when emitting float)
        if re.match(r"^\*\*Table\s+", stripped):
            i += 1
            continue

        # Mermaid or code fence
        if stripped.startswith("```"):
            lang = stripped[3:].strip().lower()
            i += 1
            block = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1
            code = "\n".join(block)
            if lang == "mermaid":
                fig_count += 1
                mapped = None
                for rx, png, default_cap in MERMAID_FIGURE_MAP:
                    if rx.search(last_figure_caption) or rx.search(code[:200]):
                        mapped = (png, default_cap)
                        break
                cap = last_figure_caption or (mapped[1] if mapped else "System diagram.")
                # Strip trailing period description duplication
                cap = re.sub(r"\.\s*Figure.*", ".", cap)
                if mapped:
                    out.append("\\begin{figure}[htbp]")
                    out.append("  \\centering")
                    out.append(f"  \\includegraphics[width=0.95\\textwidth]{{{mapped[0]}}}")
                    out.append(f"  \\caption{{{escape_tex(cap)}}}")
                    out.append(f"  \\label{{fig:ch{n}-{fig_count}}}")
                    out.append("\\end{figure}")
                    out.append("")
                else:
                    out.append(f"% Mermaid diagram omitted (no exported figure): {escape_tex(cap)}")
                    out.append("")
                last_figure_caption = ""
                continue
            # Code listing
            out.append("\\begin{verbatim}")
            out.append(code.rstrip() + "\n")
            # verbatim cannot have \end{verbatim} inside — trust source
            if out[-1].endswith("\n"):
                out[-1] = out[-1][:-1]
            out.append("\\end{verbatim}")
            out.append("")
            continue

        # Wiki image
        m = re.match(r"!\[\[(.+?)\]\]", stripped)
        if m:
            name = m.group(1).split("|", 1)[0].strip()
            fig_count += 1
            cap = last_figure_caption or Path(name).stem.replace("_", " ")
            cap = re.sub(r"\.\s*Figure.*", ".", cap)
            # Keep only first sentence if caption embeds prose
            if ". " in cap and cap.lower().startswith("figure") is False:
                # captions like "Settings/... Figure 4.2 shows..."
                pass
            # Prefer short caption before "Figure X shows"
            sm = re.match(r"^(.*?)\.\s+Figure\s+", cap)
            if sm:
                cap = sm.group(1).strip()
            out.append("\\begin{figure}[htbp]")
            out.append("  \\centering")
            out.append("  \\fbox{\\parbox[c][0.25\\textheight][c]{0.85\\textwidth}{\\centering Figure placeholder}}")
            out.append(f"  \\caption{{{escape_tex(cap)}}}")
            out.append(f"  \\label{{fig:ch{n}-{fig_count}}}")
            out.append("\\end{figure}")
            out.append("")
            last_figure_caption = ""
            i += 1
            continue

        # Figure caption line (may be followed by prose on same line)
        fm = re.match(r"^\*\*Figure\s+([\d.]+)\.?\s*(.*?)\*\*(.*)$", stripped)
        if fm:
            rest_title = fm.group(2).strip().rstrip(".")
            after = fm.group(3).strip()
            last_figure_caption = rest_title
            if after:
                # Remaining prose on same line
                out.append(format_inline(after))
                out.append("")
            i += 1
            # If next is not image/mermaid, caption may apply to following fence
            continue

        # Headings
        if stripped.startswith("#"):
            level = len(re.match(r"^#+", stripped).group(0))
            heading = clean_heading(stripped)
            if not heading or heading.lower().startswith("chapter"):
                i += 1
                continue
            label_key = heading.lower()
            label = SECTION_LABELS.get(label_key, "")
            cmd = {2: "section", 3: "subsection", 4: "subsubsection"}.get(level, "subsubsection")
            if label:
                out.append(f"\\{cmd}{{{escape_tex(heading)}}}")
                out.append(f"\\label{{{label}}}")
            else:
                out.append(f"\\{cmd}{{{escape_tex(heading)}}}")
            out.append("")
            i += 1
            continue

        # Table
        if stripped.startswith("|"):
            rows, ni = parse_table_block(lines, i)
            table_count += 1
            cap = look_caption(lines[max(0, i - 5) : i]) or f"Table {n}.{table_count}"
            # remove bold markdown from caption leftovers
            cap = re.sub(r"\*\*", "", cap)
            out.append(md_table_to_latex(rows, cap, f"tab:ch{n}-{table_count}"))
            i = ni
            continue

        # Blank
        if not stripped:
            out.append("")
            i += 1
            continue

        # Numbered list item
        nm = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if nm:
            # Start enumerate if not open
            items = []
            while i < len(lines):
                s2 = lines[i].strip()
                nm2 = re.match(r"^(\d+)\.\s+(.*)$", s2)
                if not nm2:
                    # continuation indented lines
                    if s2 and (lines[i].startswith("    ") or lines[i].startswith("\t")):
                        items[-1] = items[-1] + " " + s2
                        i += 1
                        continue
                    break
                items.append(nm2.group(2))
                i += 1
                # gather indented continuation
                while i < len(lines) and (lines[i].startswith("    ") or lines[i].startswith("\t")) and lines[i].strip():
                    items[-1] = items[-1] + " " + lines[i].strip()
                    i += 1
            out.append("\\begin{enumerate}")
            for it in items:
                # split bold title from body if ".** " pattern
                out.append(f"  \\item {format_inline(it)}")
            out.append("\\end{enumerate}")
            out.append("")
            continue

        # Bullet list
        if stripped.startswith("- ") or stripped.startswith("* "):
            items = []
            while i < len(lines) and lines[i].strip().startswith(("- ", "* ")):
                items.append(lines[i].strip()[2:])
                i += 1
            out.append("\\begin{itemize}")
            for it in items:
                out.append(f"  \\item {format_inline(it)}")
            out.append("\\end{itemize}")
            out.append("")
            continue

        # Ordinary paragraph — gather consecutive non-empty non-special lines
        para = [stripped]
        i += 1
        while i < len(lines):
            s2 = lines[i].strip()
            if not s2:
                break
            if s2.startswith("#") or s2.startswith("|") or s2.startswith("```") or s2.startswith("![["):
                break
            if s2.startswith("- ") or s2.startswith("* ") or re.match(r"^\d+\.\s+", s2):
                break
            if s2.startswith("**Table") or s2.startswith("**Figure"):
                break
            para.append(s2)
            i += 1
        out.append(format_inline(" ".join(para)))
        out.append("")

    # Cleanup excess blank lines
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Fix broken verbatim newlines
    text = text.replace("\\begin{verbatim}\n\n", "\\begin{verbatim}\n")
    return text.strip() + "\n"


def update_graphicspath() -> None:
    cfg = CONFIG.read_text(encoding="utf-8")
    new = (
        r"\graphicspath{{../../thesis/figures/ch02-system-design/}"
        r"{../../thesis/figures/ch02-ui-screenshots/}"
        r"{../../thesis/figures/ch04-results/}"
        r"{../../thesis/figures/ch04-implementation/}}"
    )
    cfg2, n = re.subn(
        r"\\graphicspath\{(?:\{[^}]*\})+\}",
        lambda _m: new,
        cfg,
    )
    if n:
        CONFIG.write_text(cfg2, encoding="utf-8")
        print("Updated graphicspath in config.tex")


def main() -> None:
    for n in range(1, 7):
        tex = convert_chapter(n)
        dest = OUT / f"chapter{n}.tex"
        dest.write_text(tex, encoding="utf-8")
        print(f"Wrote {dest} ({len(tex)} chars)")


if __name__ == "__main__":
    main()
