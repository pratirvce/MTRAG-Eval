#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
MD_DIR = ROOT / "md_files"

EXPERIMENTS_COMPLETE_LISTING = MD_DIR / "EXPERIMENTS_COMPLETE_LISTING.md"
ALL_EXPERIMENTS_LIST = MD_DIR / "ALL_EXPERIMENTS_LIST.md"
NOVEL_EXPERIMENTS_COMPLETE = MD_DIR / "NOVEL_EXPERIMENTS_COMPLETE.md"
NOVEL_EXPERIMENTS_PROPOSAL = MD_DIR / "NOVEL_EXPERIMENTS_PROPOSAL.md"

EXPERIMENT_STATUS_JSON = ROOT / "experiment_status.json"
FIXED_EXPERIMENTS_STATUS_JSON = ROOT / "fixed_experiments_status.json"
STOPPED_EXPERIMENTS_JSON = ROOT / "stopped_experiments_resume_info.json"

OUTPUT_PATH = ROOT / "ALL_EXPERIMENTS_DETAILED_REPORT.md"


@dataclass
class ExperimentRecord:
    name: str
    status: str = "unknown"
    novelty_level: Optional[str] = None
    is_novel_idea: Optional[bool] = None
    ndcg10: Optional[str] = None
    description: Optional[str] = None
    sources: List[str] = field(default_factory=list)

    def merge(self, other: "ExperimentRecord") -> None:
        if self.status == "unknown" and other.status != "unknown":
            self.status = other.status
        if self.novelty_level is None and other.novelty_level is not None:
            self.novelty_level = other.novelty_level
        if self.is_novel_idea is None and other.is_novel_idea is not None:
            self.is_novel_idea = other.is_novel_idea
        if self.ndcg10 is None and other.ndcg10 is not None:
            self.ndcg10 = other.ndcg10
        if self.description is None and other.description is not None:
            self.description = other.description
        for src in other.sources:
            if src not in self.sources:
                self.sources.append(src)


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_markdown_table(md: str, header_pattern: str) -> List[List[str]]:
    lines = md.splitlines()
    rows: List[List[str]] = []
    for i, line in enumerate(lines):
        if re.search(header_pattern, line):
            # Expect separator next line, then rows until non-table
            for j in range(i + 2, len(lines)):
                row = lines[j].strip()
                if not row.startswith("|"):
                    break
                cols = [c.strip() for c in row.strip("|").split("|")]
                rows.append(cols)
            break
    return rows


def parse_experiments_complete_listing() -> Dict[str, ExperimentRecord]:
    records: Dict[str, ExperimentRecord] = {}
    md = read_text(EXPERIMENTS_COMPLETE_LISTING)
    if not md:
        return records

    completed_rows = parse_markdown_table(md, r"\|\s*Experiment Name\s*\|\s*Novelty\s*\|\s*nDCG@10")
    for cols in completed_rows:
        if len(cols) < 4:
            continue
        name = cols[0].strip("` ")
        novelty = cols[1]
        ndcg10 = cols[2]
        desc = cols[3]
        rec = ExperimentRecord(
            name=name,
            status="completed",
            novelty_level=novelty if novelty else None,
            ndcg10=ndcg10 if ndcg10 else None,
            description=desc if desc else None,
            sources=["EXPERIMENTS_COMPLETE_LISTING.md#completed"],
        )
        records[name] = rec

    running_rows = parse_markdown_table(md, r"\|\s*Experiment Name\s*\|\s*Novelty\s*\|\s*Status\s*\|")
    for cols in running_rows:
        if len(cols) < 4:
            continue
        name = cols[0].strip("` ")
        novelty = cols[1]
        status = cols[2]
        desc = cols[3]
        rec = ExperimentRecord(
            name=name,
            status=status.lower() if status else "running",
            novelty_level=novelty if novelty else None,
            description=desc if desc else None,
            sources=["EXPERIMENTS_COMPLETE_LISTING.md#running"],
        )
        records[name] = records.get(name, ExperimentRecord(name=name))
        records[name].merge(rec)

    pending_rows = parse_markdown_table(md, r"\|\s*Experiment Name\s*\|\s*Novelty\s*\|\s*Priority\s*\|")
    for cols in pending_rows:
        if len(cols) < 4:
            continue
        name = cols[0].strip("` ")
        novelty = cols[1]
        desc = cols[3]
        rec = ExperimentRecord(
            name=name,
            status="pending",
            novelty_level=novelty if novelty else None,
            description=desc if desc else None,
            sources=["EXPERIMENTS_COMPLETE_LISTING.md#pending"],
        )
        records[name] = records.get(name, ExperimentRecord(name=name))
        records[name].merge(rec)

    failed_rows = parse_markdown_table(md, r"\|\s*Experiment Name\s*\|\s*Novelty\s*\|\s*Error\s*\|")
    for cols in failed_rows:
        if len(cols) < 3:
            continue
        name = cols[0].strip("` ")
        novelty = cols[1]
        desc = cols[2]
        rec = ExperimentRecord(
            name=name,
            status="failed",
            novelty_level=novelty if novelty else None,
            description=desc if desc else None,
            sources=["EXPERIMENTS_COMPLETE_LISTING.md#failed"],
        )
        records[name] = records.get(name, ExperimentRecord(name=name))
        records[name].merge(rec)

    return records


def parse_all_experiments_list() -> Dict[str, ExperimentRecord]:
    records: Dict[str, ExperimentRecord] = {}
    md = read_text(ALL_EXPERIMENTS_LIST)
    if not md:
        return records
    for line in md.splitlines():
        m = re.match(r"\s*\d+\.\s+\*\*(.+?)\*\*\s+-\s+(.*)$", line)
        if not m:
            continue
        name = m.group(1).strip()
        desc = m.group(2).strip()
        rec = ExperimentRecord(
            name=name,
            status="run-unknown",
            description=desc,
            sources=["ALL_EXPERIMENTS_LIST.md"],
        )
        records[name] = rec
    return records


def parse_experiment_status_json() -> Dict[str, ExperimentRecord]:
    records: Dict[str, ExperimentRecord] = {}
    if not EXPERIMENT_STATUS_JSON.exists():
        return records
    data = json.loads(read_text(EXPERIMENT_STATUS_JSON) or "{}")
    for name, info in data.items():
        status = info.get("status") or "unknown"
        rec = ExperimentRecord(
            name=name,
            status=status.lower(),
            sources=["experiment_status.json"],
        )
        records[name] = rec
    return records


def parse_fixed_experiments_status_json() -> Dict[str, ExperimentRecord]:
    records: Dict[str, ExperimentRecord] = {}
    if not FIXED_EXPERIMENTS_STATUS_JSON.exists():
        return records
    data = json.loads(read_text(FIXED_EXPERIMENTS_STATUS_JSON) or "{}")
    experiments = data.get("experiments", {})
    for name, info in experiments.items():
        status = "unknown"
        if info.get("is_complete"):
            status = "completed"
        elif info.get("is_running"):
            status = "running"
        rec = ExperimentRecord(
            name=name,
            status=status,
            sources=["fixed_experiments_status.json"],
        )
        records[name] = rec
    return records


def parse_stopped_experiments_json() -> Dict[str, ExperimentRecord]:
    records: Dict[str, ExperimentRecord] = {}
    if not STOPPED_EXPERIMENTS_JSON.exists():
        return records
    data = json.loads(read_text(STOPPED_EXPERIMENTS_JSON) or "{}")
    for item in data.get("stopped_experiments", []):
        name = item.get("experiment_name")
        if not name:
            continue
        rec = ExperimentRecord(
            name=name,
            status="stopped",
            sources=["stopped_experiments_resume_info.json"],
        )
        records[name] = rec
    return records


def parse_novel_experiments_sets() -> Tuple[set, set]:
    novel_names = set()
    proposal_names = set()

    for path in [NOVEL_EXPERIMENTS_COMPLETE, NOVEL_EXPERIMENTS_PROPOSAL]:
        md = read_text(path)
        if not md:
            continue
        for match in re.findall(r"`(novel_[a-z0-9_]+)`", md):
            novel_names.add(match)
        for match in re.findall(r"\bnovel_[a-z0-9_]+\b", md):
            novel_names.add(match)

        for match in re.findall(r"\bCATR\b|\bUQ-Ret\b|\bDiff-Ret\b|\bMeta-Ret\b|\bCausal-Ret\b|\bGEP-Ret\b|\bX-Ret\b|\bCLCF-Ret\b", md):
            proposal_names.add(match)

    return novel_names, proposal_names


def decide_novel_flag(name: str, novelty_level: Optional[str], novel_name_set: set) -> bool:
    if novelty_level:
        if novelty_level.strip().lower() in {"high", "medium"}:
            return True
        if novelty_level.strip().lower() == "low":
            return False
    if name.startswith("novel_"):
        return True
    if name in novel_name_set:
        return True
    return False


def normalize_status(raw_status: str) -> str:
    s = (raw_status or "").strip().lower()
    if s in {"completed", "done", "success"}:
        return "completed"
    if s in {"running", "in_progress", "in progress"}:
        return "running"
    if s in {"failed", "error"}:
        return "failed"
    if s in {"pending", "not started", "not_started"}:
        return "pending"
    if s in {"stopped", "paused"}:
        return "stopped"
    if s in {"run-unknown"}:
        return "run-unknown"
    return s or "unknown"


def write_report(records: Dict[str, ExperimentRecord]) -> None:
    # Finalize status + novel flag
    novel_name_set, _ = parse_novel_experiments_sets()

    for rec in records.values():
        rec.status = normalize_status(rec.status)
        if rec.is_novel_idea is None:
            rec.is_novel_idea = decide_novel_flag(rec.name, rec.novelty_level, novel_name_set)
        if rec.description is None:
            rec.description = "No description found in source files."

    # Sort by status then name
    status_order = {"completed": 0, "running": 1, "pending": 2, "stopped": 3, "failed": 4, "run-unknown": 5, "unknown": 6}
    sorted_records = sorted(
        records.values(),
        key=lambda r: (status_order.get(r.status, 99), r.name),
    )

    # Summary counts
    counts: Dict[str, int] = {}
    for rec in sorted_records:
        counts[rec.status] = counts.get(rec.status, 0) + 1

    lines: List[str] = []
    lines.append("# All Experiments: Status, Novelty, and Explanations")
    lines.append("")
    lines.append(f"**Generated:** {Path(__file__).name}")
    lines.append("")
    lines.append("This report consolidates experiment status, novelty, and descriptions into a single list.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for status in sorted(counts.keys(), key=lambda k: status_order.get(k, 99)):
        lines.append(f"- **{status.title()}**: {counts[status]}")
    lines.append("")
    lines.append("## Experiment List (All)")
    lines.append("")
    lines.append("| Experiment | Status | Novel Idea | Novelty | nDCG@10 | Description |")
    lines.append("|-----------|--------|------------|---------|--------|-------------|")

    for rec in sorted_records:
        novel_flag = "Yes" if rec.is_novel_idea else "No"
        novelty = rec.novelty_level or ""
        ndcg10 = rec.ndcg10 or ""
        desc = rec.description.replace("\n", " ").strip()
        lines.append(f"| `{rec.name}` | {rec.status} | {novel_flag} | {novelty} | {ndcg10} | {desc} |")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- **Novel Idea** is marked `Yes` when novelty is High/Medium, the name starts with `novel_`, or it appears in novel proposal/implementation files.")
    lines.append("- **run-unknown** indicates the experiment appears in historical run lists without a clear completion status in the status files.")

    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    records: Dict[str, ExperimentRecord] = {}

    # Primary listing with status/novelty/description
    for name, rec in parse_experiments_complete_listing().items():
        records[name] = rec

    # Historical run list
    for name, rec in parse_all_experiments_list().items():
        records.setdefault(name, ExperimentRecord(name=name)).merge(rec)

    # Status JSON sources
    for name, rec in parse_experiment_status_json().items():
        records.setdefault(name, ExperimentRecord(name=name)).merge(rec)

    for name, rec in parse_fixed_experiments_status_json().items():
        records.setdefault(name, ExperimentRecord(name=name)).merge(rec)

    for name, rec in parse_stopped_experiments_json().items():
        records.setdefault(name, ExperimentRecord(name=name)).merge(rec)

    if not records:
        raise SystemExit("No experiments found from source files.")

    write_report(records)


if __name__ == "__main__":
    main()
