"""Score the rule-based label extractor against real radiologist prose.

The synthetic corpus cannot tell us whether the extractor works, because its
reports were written by the same hand as the phrase bank. This script runs the
extractor over the Open-i / Indiana University collection and scores it against
that corpus's manually assigned MeSH terms.

Only findings the MeSH vocabulary actually distinguishes are scored. Reporting a
macro average over labels the ground truth never asserts would flatter the result
by padding it with easy true negatives.

Usage:  python scripts/evaluate_openi.py --corpus realdata/ecgen-radiology
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cxr_harmony.adapters.openi import load_corpus  # noqa: E402
from cxr_harmony.qc.agreement import agreement_by_label, pooled_kappa  # noqa: E402
from cxr_harmony.reports.labels import legacy_patterns, positive_findings  # noqa: E402
from cxr_harmony.reports.parser import clinical_text, parse_sections  # noqa: E402
from cxr_harmony.schema.vocab import Finding  # noqa: E402

#: Findings the MeSH annotations distinguish often enough to score.
SCORED = (
    Finding.CARDIOMEGALY,
    Finding.PLEURAL_EFFUSION,
    Finding.CONSOLIDATION,
    Finding.PNEUMOTHORAX,
    Finding.PULMONARY_EDEMA,
    Finding.ATELECTASIS,
    Finding.NODULE,
    Finding.FRACTURE,
    Finding.TUBERCULOSIS,
)


@dataclass
class Score:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def support(self) -> int:
        return self.tp + self.fn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def split_corpus(reports: list, fold: str) -> list:
    """Deterministic halves for tuning and for reporting.

    The phrase bank was refined by reading failures on this corpus, so a score on
    the reports used for that refinement is optimistic by construction. Odd/even
    record ids give a stable split that needs no stored index, and the held-out
    half is the number that should be quoted.
    """
    if fold == "all":
        return reports
    want_even = fold == "dev"
    out = []
    for report in reports:
        digits = "".join(ch for ch in report.uid if ch.isdigit())
        index = int(digits) if digits else 0
        if (index % 2 == 0) == want_even:
            out.append(report)
    return out


def evaluate(
    corpus_dir: Path,
    limit: int | None = None,
    fold: str = "all",
    baseline: bool = False,
) -> dict:
    """Score the extractor. With ``baseline``, scores the pre-evaluation pattern set."""
    reports = split_corpus(load_corpus(corpus_dir, limit=limit), fold)

    if baseline:
        with legacy_patterns():
            return _score(reports, baseline=True)
    return _score(reports, baseline=False)


def _score(reports: list, *, baseline: bool) -> dict:

    per_finding: dict[Finding, Score] = {f: Score() for f in SCORED}
    normal_tp = normal_fp = normal_fn = 0
    exact = 0
    examples_missed: dict[str, list[str]] = {}
    examples_spurious: dict[str, list[str]] = {}
    predicted_sets: list[set[str]] = []
    reference_sets: list[set[str]] = []

    for report in reports:
        # Route through the same parser the pipeline uses, so the section
        # restriction is exercised rather than bypassed.
        sections = parse_sections(report.as_report_text())
        predicted = set(positive_findings(clinical_text(sections)))

        truth = set(report.findings)
        truth_pos = {f for f in truth if f in per_finding}
        pred_pos = {f for f in predicted if f in per_finding}
        predicted_sets.append({f.value for f in pred_pos})
        reference_sets.append({f.value for f in truth_pos})

        for finding in SCORED:
            score = per_finding[finding]
            in_truth, in_pred = finding in truth_pos, finding in pred_pos
            if in_truth and in_pred:
                score.tp += 1
            elif in_pred:
                score.fp += 1
                examples_spurious.setdefault(finding.value, [])
                if len(examples_spurious[finding.value]) < 3:
                    examples_spurious[finding.value].append(
                        report.clinical_text[:200].replace("\n", " ")
                    )
            elif in_truth:
                score.fn += 1
                examples_missed.setdefault(finding.value, [])
                if len(examples_missed[finding.value]) < 3:
                    examples_missed[finding.value].append(
                        report.clinical_text[:200].replace("\n", " ")
                    )

        truth_normal = Finding.NO_FINDING in truth
        pred_normal = Finding.NO_FINDING in predicted
        if truth_normal and pred_normal:
            normal_tp += 1
        elif pred_normal:
            normal_fp += 1
        elif truth_normal:
            normal_fn += 1

        if truth_pos == pred_pos:
            exact += 1

    micro = Score()
    for score in per_finding.values():
        micro.tp += score.tp
        micro.fp += score.fp
        micro.fn += score.fn

    scored_with_support = [s for s in per_finding.values() if s.support > 0]
    macro_f1 = (
        sum(s.f1 for s in scored_with_support) / len(scored_with_support)
        if scored_with_support
        else 0.0
    )

    normal = Score(tp=normal_tp, fp=normal_fp, fn=normal_fn)

    scored_names = [f.value for f in SCORED]
    agreements = agreement_by_label(predicted_sets, reference_sets, scored_names)
    pooled = pooled_kappa(agreements)

    return {
        "baseline": baseline,
        "n_reports": len(reports),
        "micro": {
            "precision": round(micro.precision, 4),
            "recall": round(micro.recall, 4),
            "f1": round(micro.f1, 4),
            "tp": micro.tp,
            "fp": micro.fp,
            "fn": micro.fn,
        },
        "macro_f1": round(macro_f1, 4),
        "exact_match_rate": round(exact / len(reports), 4) if reports else 0.0,
        "normal_detection": {
            "precision": round(normal.precision, 4),
            "recall": round(normal.recall, 4),
            "f1": round(normal.f1, 4),
            "support": normal.support,
        },
        "per_finding": {
            finding.value: {
                "precision": round(score.precision, 4),
                "recall": round(score.recall, 4),
                "f1": round(score.f1, 4),
                "support": score.support,
                "tp": score.tp,
                "fp": score.fp,
                "fn": score.fn,
            }
            for finding, score in per_finding.items()
        },
        "kappa": {
            "pooled": pooled.to_dict(),
            "per_finding": {k: v.to_dict() for k, v in sorted(agreements.items())},
        },
        "examples_missed": examples_missed,
        "examples_spurious": examples_spurious,
    }


def render(result: dict) -> str:
    lines = [
        f"Reports scored: {result['n_reports']}",
        "",
        f"{'finding':<20} {'prec':>7} {'recall':>7} {'F1':>7} {'support':>8} "
        f"{'TP':>5} {'FP':>5} {'FN':>5}",
        "-" * 74,
    ]
    for name, s in sorted(
        result["per_finding"].items(), key=lambda kv: -kv[1]["support"]
    ):
        lines.append(
            f"{name:<20} {s['precision']:>7.3f} {s['recall']:>7.3f} {s['f1']:>7.3f} "
            f"{s['support']:>8} {s['tp']:>5} {s['fp']:>5} {s['fn']:>5}"
        )
    m = result["micro"]
    lines += [
        "-" * 74,
        f"{'MICRO':<20} {m['precision']:>7.3f} {m['recall']:>7.3f} {m['f1']:>7.3f} "
        f"{m['tp'] + m['fn']:>8} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5}",
        "",
        f"macro F1            : {result['macro_f1']:.3f}",
        f"exact-match rate    : {result['exact_match_rate']:.3f}",
        f"normal detection F1 : {result['normal_detection']['f1']:.3f} "
        f"(support {result['normal_detection']['support']})",
    ]

    pooled = result["kappa"]["pooled"]
    lines += [
        "",
        f"Cohen's kappa (pooled): {pooled['kappa']:.3f}  ({pooled['interpretation']})"
        "   target > 0.80",
        "",
        f"{'finding':<20} {'kappa':>7} {'support':>8}  interpretation",
        "-" * 58,
    ]
    for name, k in sorted(
        result["kappa"]["per_finding"].items(), key=lambda kv: -kv[1]["support"]
    ):
        lines.append(
            f"{name:<20} {k['kappa']:>7.3f} {k['support']:>8}  {k['interpretation']}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("realdata/ecgen-radiology"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--show-misses", action="store_true")
    parser.add_argument("--fold", choices=["all", "dev", "heldout"], default="all")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="score the pre-evaluation pattern set, to regenerate the before/after comparison",
    )
    args = parser.parse_args()

    result = evaluate(args.corpus, args.limit, args.fold, args.baseline)
    print(f"fold: {args.fold}{'  [BASELINE pattern set]' if args.baseline else ''}")
    print(render(result))

    if args.show_misses:
        print("\n--- missed (false negatives) ---")
        for finding, examples in sorted(result["examples_missed"].items()):
            print(f"\n[{finding}]")
            for text in examples:
                print(f"  {text}")
        print("\n--- spurious (false positives) ---")
        for finding, examples in sorted(result["examples_spurious"].items()):
            print(f"\n[{finding}]")
            for text in examples:
                print(f"  {text}")

    if args.json_out:
        # Example sentences are printed to stdout for debugging but never written
        # to a file that might be committed: they are verbatim text from a corpus
        # licensed CC BY-NC-ND, which forbids derivatives. Excluding them here
        # rather than stripping afterwards means a future run cannot reintroduce
        # them by being invoked slightly differently.
        publishable = {
            k: v
            for k, v in result.items()
            if k not in ("examples_missed", "examples_spurious")
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(publishable, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
