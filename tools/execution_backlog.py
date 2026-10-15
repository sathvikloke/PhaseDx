#!/usr/bin/env python3
"""Read/validate/render the PhaseDx task queue. Never executes study work."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "execution"
QUEUE = DOCS / "tasks.json"
STATUSES = {"ready", "pending", "doing", "blocked", "done", "optional",
            "decision_needed", "awaiting_scientific_review", "dropped"}


def load():
    return json.loads(QUEUE.read_text())["tasks"]


def completed(task):
    return task["status"] == "done" or (
        task["kind"] == "execution"
        and task["status"] == "awaiting_scientific_review"
        and bool(task.get("evidence"))
    )


def validate(tasks):
    ids = [t["id"] for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task IDs")
    by_id = {t["id"]: t for t in tasks}
    for t in tasks:
        for field in ("title", "owner", "kind", "status", "inputs", "outputs",
                      "steps", "acceptance", "resource", "card"):
            if not t.get(field):
                raise ValueError(f"{t['id']}: missing {field}")
        if t["status"] not in STATUSES:
            raise ValueError(f"{t['id']}: invalid status")
        if t["kind"] not in {"execution", "scientific_author"}:
            raise ValueError(f"{t['id']}: invalid kind")
        if not isinstance(t["steps"], list) or not t["steps"]:
            raise ValueError(f"{t['id']}: steps must be a nonempty list")
        if not isinstance(t.get("evidence"), list):
            raise ValueError(f"{t['id']}: evidence must be a list")
        if t["status"] in {"done", "awaiting_scientific_review"} and not t["evidence"]:
            raise ValueError(f"{t['id']}: completed status needs evidence")
        for dep in t["dependencies"]:
            if dep not in by_id or dep == t["id"]:
                raise ValueError(f"{t['id']}: invalid dependency {dep}")
        card = (DOCS / t["card"]).resolve()
        if not card.is_relative_to(DOCS.resolve()) or not card.is_file():
            raise ValueError(f"{t['id']}: card missing or outside docs")
    visited, visiting = set(), set()

    def visit(i):
        if i in visiting:
            raise ValueError(f"Dependency cycle at {i}")
        if i in visited:
            return
        visiting.add(i)
        for dep in by_id[i]["dependencies"]:
            visit(dep)
        visiting.remove(i)
        visited.add(i)

    for i in ids:
        visit(i)
    return by_id


def ready(tasks, by_id):
    return [t for t in tasks
            if t["kind"] == "execution" and not t["optional"]
            and t["status"] in {"ready", "pending"}
            and not t.get("gate")
            and all(completed(by_id[d]) for d in t["dependencies"])]


def render(tasks):
    lines = ["# Task index", "", "Generated from `tasks.json`. Task counts are not completion percentages or measured worker concurrency.", "",
             "All execution is owned by the user / Claude. Scientific and author decisions stay with the user/Codex or a qualified human. No collaborator compute allocation.", ""]
    for kind, heading in (("execution", "Execution tasks"), ("scientific_author", "Scientific and author tasks")):
        lines += [f"## {heading}", "", "| ID | Deliverable | Lane | Status | Dependencies |", "|---|---|---|---|---|"]
        for t in tasks:
            if t["kind"] == kind:
                deps = ", ".join(t["dependencies"]) or "none"
                lines.append(f"| [{t['id']}]({t['card']}) | {t['title']} | {t['lane']} | {t['status']} | {deps} |")
        lines += [""]
    (DOCS / "TASK_INDEX.md").write_text("\n".join(lines))
    fields = ["id", "title", "owner", "kind", "lane", "priority", "status", "dependencies", "resource", "card"]
    with (DOCS / "TASK_INDEX.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for t in tasks:
            row = {k: t[k] for k in fields}
            row["dependencies"] = ";".join(t["dependencies"])
            writer.writerow(row)
    for t in tasks:
        deps = ", ".join(f"[{i}]({i}.md)" for i in t["dependencies"]) or "none; start independently when inputs are available"
        parts = [f"# {t['id']} — {t['title']}", "",
                 f"Owner: **{t['owner']}**. Lane: {t['lane']}. Priority: {t['priority']}. Status: {t['status']}.", "",
                 f"Dependencies: {deps}.", "", "## Inputs", "", t["inputs"], "",
                 "Resolve batch locators through [source map](../source_map.json) and [input configuration](../INPUTS.md). Historical inputs are read-only.", "",
                 "## Deliverables and owned paths", "", t["outputs"], "", "## Execution", ""]
        parts += [f"{i+1}. {s}" for i, s in enumerate(t["steps"])]
        parts += ["", "## Acceptance", "", t["acceptance"], "",
                  f"Resource profile: {t['resource']}. Actual runtime is unknown until measured; no invented nine-hour compute estimate."]
        if t.get("gate"):
            parts += ["", "Decision boundary: " + t["gate"]]
        if t["optional"]:
            parts += ["", "**Conditional scope:** do not schedule automatically. Record the scientific/operational reason and activate only this task when that condition actually exists."]
        parts += ["", "Return: exact command and environment, input/config identities, output locations, exit status, failures and measured timing. Mark execution complete only with evidence; scientific acceptance is separate. If one input is missing, name it and backfill another ready task.", ""]
        if t["evidence"]:
            parts += ["## Completion evidence", ""] + [f"- {x}" for x in t["evidence"]] + [""]
        (DOCS / t["card"]).write_text("\n".join(parts))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["validate", "ready", "show", "render"])
    p.add_argument("task", nargs="?")
    a = p.parse_args()
    tasks = load()
    try:
        by_id = validate(tasks)
    except (ValueError, KeyError, TypeError) as e:
        p.exit(1, f"Invalid backlog: {e}\n")
    if a.command == "validate":
        counts = Counter(t["kind"] for t in tasks)
        print(f"Valid: {len(tasks)} cards, {counts['execution']} execution / {counts['scientific_author']} scientific-author; unique IDs, complete fields, existing cards, acyclic dependencies.")
        print(f"Dependency-ready execution tasks: {len(ready(tasks, by_id))}. Private-input availability remains task-specific.")
    elif a.command == "ready":
        for t in ready(tasks, by_id):
            print(f"{t['id']} | {t['lane']} | {t['title']}")
    elif a.command == "show":
        if a.task not in by_id:
            p.error("show requires an existing task ID")
        print((DOCS / by_id[a.task]["card"]).read_text())
    elif a.command == "render":
        render(tasks)
        print("Rendered task index, CSV and cards from tasks.json.")


if __name__ == "__main__":
    main()
