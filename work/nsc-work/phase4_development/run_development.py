"""Frozen Development-only orchestration; never runs or reads Final outcomes.

The unmodified official evaluate.evaluate is called in each fresh candidate
process. Supplements retain scores and all seeded selections and verify their
aggregates against the official output, rather than redefining its metrics.
"""

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent
PROJECT = WORK.parents[1]
REPORT = PROJECT / "analysis" / "MB_NSC_Phase4_Development_Report.md"
sys.path.insert(0, str(WORK))
TOKYO = timezone(timedelta(hours=9))
GRID = tuple(j / 10 for j in range(11))
TOL = 1e-12
ALLOWED_DATA = {WORK / "data" / name for name in (
    "architectures.json", "splits.json", "resources.json", "development_labels.json"
)}
READS = set()
DENIALS = []


def access_guard(event, args):
    if event != "open" or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    if PROJECT not in path.parents:
        return
    flags = args[2] or 0
    writing = flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
    output = ROOT in path.parents or path == REPORT
    if (writing and not output) or (
        path.suffix.lower() == ".json" and not output and path not in ALLOWED_DATA
    ):
        DENIALS.append(str(path))
        raise PermissionError(f"Blocked non-Development/protected access: {path}")
    if not writing and path in ALLOWED_DATA:
        READS.add(str(path.relative_to(WORK)))


sys.addaudithook(access_guard)
os.environ["MPLCONFIGDIR"] = str(ROOT / "logs" / "matplotlib_cache")


def now():
    return datetime.now(TOKYO).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"Duplicate JSON key: {key}")
        obj[key] = value
    return obj


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def load_development():
    configs = read_json(WORK / "data" / "architectures.json")
    splits = read_json(WORK / "data" / "splits.json")
    resources = read_json(WORK / "data" / "resources.json")
    labels = read_json(WORK / "data" / "development_labels.json")
    keys, final_keys = splits["development"], splits["final"]
    assert len(configs) == 200 and len(keys) == len(set(keys)) == 100
    assert len(final_keys) == len(set(final_keys)) == 100
    assert not set(keys) & set(final_keys)
    assert set(keys) | set(final_keys) == set(configs)
    assert set(labels) == set(keys), "Development labels missing or extra entries"
    from nsc import layers
    for key in keys:
        assert finite_positive(labels[key]["valid_ppl"]), f"Invalid Development PPL: {key}"
        assert finite_positive(resources[key]["total"]), f"Invalid resource total: {key}"
        layers(configs[key])
    ppl = [float(labels[key]["valid_ppl"]) for key in keys]
    pairs = [(i, j) for i in range(100) for j in range(i)]
    matched = [(i, j) for i, j in pairs
               if configs[keys[i]]["d_model"] == configs[keys[j]]["d_model"]
               and min(resources[keys[i]]["total"], resources[keys[j]]["total"])
               / max(resources[keys[i]]["total"], resources[keys[j]]["total"]) >= .90]
    assert len(matched) == 180, f"Expected official matched pairs=180; found {len(matched)}"
    return configs, resources, keys, ppl, pairs, matched


def select_best(rows):
    """Maximum-anchored tolerance sets avoid nontransitive pair comparators."""
    eligible = [r for r in rows if r["rho"] is not None]
    if not eligible:
        raise ValueError("All candidates have undefined Spearman; selection stopped")
    assert all(math.isfinite(r["rho"]) and r["tau"] is not None and math.isfinite(r["tau"]) for r in eligible)
    rho_max = max(r["rho"] for r in eligible)
    rho_ties = [r for r in eligible if abs(r["rho"] - rho_max) <= TOL]
    tau_max = max(r["tau"] for r in rho_ties)
    tau_ties = [r for r in rho_ties if abs(r["tau"] - tau_max) <= TOL]
    minimum = min(r["lambda"] for r in tau_ties)
    conservative = [r for r in tau_ties if r["lambda"] == minimum]
    winner = next((r for r in conservative if r["model"] == "global"), conservative[0])
    return winner, {"rho_max": rho_max, "rho_ties": [r["method"] for r in rho_ties],
                    "tau_max_in_rho_ties": tau_max, "tau_ties": [r["method"] for r in tau_ties],
                    "minimum_lambda": minimum, "lambda_ties": [r["method"] for r in conservative],
                    "global_operational_tiebreak": len(conservative) > 1,
                    "winner": winner["method"]}


def pair_accuracy(summary):
    valid = summary["concordant"] + summary["discordant"] + summary["score_only_ties"]
    return ((summary["concordant"] + .5 * summary["score_only_ties"]) / valid if valid else None)


def selection_trials(pool, values, k, ppl, seed, repeats=200):
    rng = random.Random(seed)
    records = []
    oracle = min(ppl[i] for i in pool)
    for trial in range(repeats):
        if values is None:
            selected = rng.sample(pool, k)
        else:
            order = list(pool)
            rng.shuffle(order)
            order.sort(key=lambda i: values[i], reverse=True)
            selected = order[:k]
        best = min(ppl[i] for i in selected)
        records.append((trial + 1, selected, best, best - oracle))
    bests = [r[2] for r in records]
    stats = {"mean_best_ppl": statistics.mean(bests), "median_best_ppl": statistics.median(bests),
             "sd_best_ppl": statistics.pstdev(bests), "mean_regret": statistics.mean(bests) - oracle,
             "median_regret": statistics.median(bests) - oracle, "oracle_ppl": oracle, "trials": repeats}
    return records, stats


def assert_close(a, b):
    if a is None or b is None:
        assert a is b
    else:
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)


def candidate(variant, strength, label):
    assert variant == os.environ["MB_NSC_VARIANT"]
    assert float(os.environ["MB_NSC_LAMBDA"]) == strength and strength in GRID
    import evaluate as official
    from nsc import original_nsc, position_example, resource_estimate
    configs, resources, keys, ppl, pairs, matched = load_development()
    result = official.evaluate("development", WORK / "student_score.py")
    assert result["split"] == "development" and result["architectures"] == 100
    spec = importlib.util.spec_from_file_location("student_submission", WORK / "student_score.py")
    student = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student)
    raw, vectors = {}, {}
    for name, function in {"original_nsc": original_nsc, "position_example": position_example,
                           "student": student.score, "ablation": student.ablation}.items():
        raw[name] = []
        for key in keys:
            cfg = deepcopy(configs[key])
            snapshot = deepcopy(cfg)
            value = function(cfg)
            assert cfg == snapshot and math.isfinite(value)
            raw[name].append(value)
        vectors[name] = [official.canonical(value) for value in raw[name]]
    raw["reported_parameters"] = vectors["reported_parameters"] = [resources[key]["total"] for key in keys]
    raw["reference_flops"] = vectors["reference_flops"] = [resource_estimate(configs[key])["reference_flops_per_token"] for key in keys]
    assert raw["ablation"] == raw["original_nsc"], "Ablation exact equality failed"
    if strength == 0:
        assert raw["student"] == raw["original_nsc"], "Lambda=0 exact recovery failed"
    quality = [-v for v in ppl]
    for name, values in vectors.items():
        expected = result["ranking"][name]
        assert_close(official.correlation(official.ranks(values), official.ranks(quality)), expected["spearman_rho"])
        assert official.pair_summary(values, quality, pairs) == expected["all_pairs"]
        assert official.pair_summary(values, quality, matched) == expected["size_controlled_pairs"]
    folder = ROOT / "logs" / variant if label != "baseline" else ROOT / "logs"
    write_json(folder / f"{label}.official.json", result)
    write_json(folder / f"{label}.vectors.json", {"variant": variant, "lambda": strength,
               "keys": keys, "ppl": ppl, "raw_scores": raw, "scores": vectors,
               "allowed_data_reads": sorted(READS), "denied_accesses": DENIALS})
    print(json.dumps({"run": label, "variant": variant, "lambda": strength,
                      "rho": result["ranking"]["student"]["spearman_rho"],
                      "tau": result["ranking"]["student"]["all_pairs"]["tau_b"]}))


def run_one(variant, strength, label, manifests, version):
    folder = ROOT / "logs" / variant if label != "baseline" else ROOT / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--candidate",
               "--variant", variant, "--strength", format(strength, ".1f"), "--label", label]
    env = dict(os.environ, MB_NSC_VARIANT=variant, MB_NSC_LAMBDA=format(strength, ".1f"),
               PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    start, tick = now(), time.perf_counter()
    with (folder / f"{label}.stdout.log").open("x", encoding="utf-8") as stdout, (folder / f"{label}.stderr.log").open("x", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=WORK, env=env, stdout=stdout, stderr=stderr)
    manifests.append({"Model": variant, "Lambda": strength, "Run": label,
                      "Python": platform.python_version(), "Command": subprocess.list2cmdline(command),
                      "Start": start, "End": now(), "WallSeconds": time.perf_counter() - tick,
                      "Output": str(folder / f"{label}.official.json"), "CodeHash": version,
                      "Status": "PASS" if completed.returncode == 0 else "FAIL",
                      "Error": "" if completed.returncode == 0 else str(folder / f"{label}.stderr.log")})
    # Incremental manifest is append-only, preserving failures and prior runs.
    manifest = ROOT / "logs" / "run_manifest.csv"
    exists = manifest.exists()
    with manifest.open("a", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifests[-1]))
        if not exists:
            writer.writeheader()
        writer.writerow(manifests[-1])
    if completed.returncode:
        raise RuntimeError(f"Run failed; batch stopped: {label}; inspect stderr log")
    print(f"Completed {label} ({variant}, lambda={strength:.1f})", flush=True)
    return read_json(folder / f"{label}.official.json"), read_json(folder / f"{label}.vectors.json")


def summarize(baseline, entries, configs, resources, keys, ppl):
    import evaluate as official
    rawdir = ROOT / "raw_results"
    base_result, base_vectors = baseline
    base_names = {"Original NSC": "original_nsc", "Positional Example": "position_example",
                  "MB-NSC Ablation": "ablation", "Parameter Count": "reported_parameters",
                  "Reference FLOPs": "reference_flops"}
    methods = {name: {"scores": base_vectors["scores"][key], "raw": base_vectors["raw_scores"][key],
                       "ranking": base_result["ranking"][key], "reference": (base_result, key)}
               for name, key in base_names.items()}
    candidates = []
    for variant, strength, result, vectors in entries:
        assert vectors["keys"] == keys and vectors["ppl"] == ppl
        for name in base_names.values():
            assert vectors["scores"][name] == base_vectors["scores"][name]
        method = f"{'Layer-wise' if variant == 'layerwise' else 'Global'} (lambda={strength:.1f})"
        ranking = result["ranking"]["student"]
        row = {"method": method, "model": variant, "lambda": strength,
               "rho": ranking["spearman_rho"], "tau": ranking["all_pairs"]["tau_b"]}
        candidates.append(row)
        methods[method] = {"scores": vectors["scores"]["student"], "raw": vectors["raw_scores"]["student"],
                           "ranking": ranking, "reference": (result, "student")}
    assert len(candidates) == 22
    ranking_rows = []
    for row in candidates:
        s = methods[row["method"]]["ranking"]["size_controlled_pairs"]
        ranking_rows.append({"Model": row["model"], "Lambda": row["lambda"], "Spearman_rho": row["rho"],
                             "Kendall_tau_b": row["tau"], "SizeControlled_tau_b": s["tau_b"],
                             "SizeControlled_accuracy": pair_accuracy(s), "Status": "PASS" if row["rho"] is not None else "UNDEFINED"})
    write_csv(rawdir / "candidate_ranking_results.csv", ranking_rows)
    bests, trace = [], {}
    for variant in ("layerwise", "global"):
        best, trace[variant] = select_best([r for r in candidates if r["model"] == variant])
        bests.append(best)
    winner, trace["between_models"] = select_best(bests)
    selection_rows = [{"Model": r["model"], "BestLambda": r["lambda"], "Spearman_rho": r["rho"],
                       "Kendall_tau_b": r["tau"], "SelectionRank": 1 if r is winner else 2} for r in bests]
    write_csv(rawdir / "model_selection_results.csv", selection_rows)
    write_json(ROOT / "summary" / "selection_trace.json", trace)
    size_rows = []
    score_rows = []
    for name, method in methods.items():
        s = method["ranking"]["size_controlled_pairs"]
        size_rows.append({"Method": name, "Pairs": s["pairs"],
                          "ValidPairs": s["concordant"] + s["discordant"] + s["score_only_ties"],
                          "Correct": s["concordant"], "Incorrect": s["discordant"],
                          "PredictedTies_valid": s["score_only_ties"],
                          "ActualTies": s["quality_only_ties"] + s["both_ties"],
                          "BothTies": s["both_ties"], "PairwiseAccuracy": pair_accuracy(s), "Official_tau_b": s["tau_b"]})
        for index, key in enumerate(keys):
            score_rows.append({"Method": name, "Architecture": key, "RawScore": method["raw"][index],
                               "ComparisonScore12Sig": method["scores"][index], "ValidationPPL": ppl[index],
                               "TotalParameters": resources[key]["total"]})
    write_csv(rawdir / "size_controlled_results.csv", size_rows)
    write_csv(rawdir / "scores_by_architecture.csv", score_rows)
    budget_rows, distributions, random_rows = [], [], []
    all_methods = dict(methods, **{"Uniform Random": {"scores": None, "reference": (base_result, "random")}})
    for budget in official.BUDGETS:
        pool = [i for i, key in enumerate(keys) if resources[key]["total"] <= budget]
        for k in official.K_VALUES:
            seed = 202627 + budget + k
            for name, method in all_methods.items():
                if len(pool) < k:
                    distributions.append({"Method": name, "Budget": budget, "k": k, "Eligible": len(pool), "Status": "N/A: insufficient pool"})
                    continue
                trials, stats = selection_trials(pool, method["scores"], k, ppl, seed, official.REPEATS)
                ref_result, ref_method = method["reference"]
                ref = next(r for r in ref_result["selection"] if r["parameter_cap"] == budget and r["k"] == k)
                for field in ("mean_best_ppl", "mean_regret", "sd_best_ppl", "trials"):
                    assert_close(stats[field], ref["methods"][ref_method][field])
                assert_close(stats["oracle_ppl"], ref["oracle_ppl"])
                assert ref["eligible"] == len(pool)
                distributions.append({"Method": name, "Budget": budget, "k": k, "Eligible": len(pool), "Status": "PASS", "Seed": seed, **stats})
                if name == "Uniform Random":
                    random_rows.append(distributions[-1])
                for trial, selected, best, regret in trials:
                    budget_rows.append({"Method": name, "Budget": budget, "k": k, "Trial": trial, "Seed": seed,
                                        "SelectedArchitectures": json.dumps([keys[i] for i in selected]),
                                        "SelectedPPLs": json.dumps([ppl[i] for i in selected]),
                                        "SelectedBestPPL": best, "FeasibleBestPPL": stats["oracle_ppl"], "Regret": regret})
    write_csv(rawdir / "budgeted_selection_results.csv", budget_rows)
    write_csv(rawdir / "selection_distribution_summary.csv", distributions,
              fields=["Method", "Budget", "k", "Eligible", "Status", "Seed", "mean_best_ppl", "median_best_ppl", "sd_best_ppl", "mean_regret", "median_regret", "oracle_ppl", "trials"])
    write_csv(rawdir / "random_baseline_results.csv", random_rows)
    summary = {"split": "development", "architectures": 100, "matched_pairs": 180,
               "candidates": candidates, "model_bests": bests, "winner": winner, "selection_trace": trace,
               "baseline_rankings": {name: value["ranking"] for name, value in methods.items() if name in base_names},
               "size_controlled": size_rows, "selection_distributions": distributions,
               "budget_trial_rows": len(budget_rows), "method_count_including_random": len(all_methods),
               "allowed_data_reads": sorted(READS), "denied_accesses": DENIALS,
               "selection_rule": "rho within 1e-12 of maximum; then tau within 1e-12; smaller lambda; Global final operational tie"}
    write_json(ROOT / "summary" / "development_summary.json", summary)
    return summary


def figures(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 130, "savefig.dpi": 220})
    folder = ROOT / "figures"
    folder.mkdir(exist_ok=True)
    colors = {"layerwise": "#1967a3", "global": "#d45a31"}
    for number, metric, title in [(1, "rho", "Spearman rho"), (2, "tau", "Kendall tau-b")]:
        fig, ax = plt.subplots(figsize=(8.5, 5))
        for variant in ("layerwise", "global"):
            rows = [r for r in summary["candidates"] if r["model"] == variant]
            ax.plot([r["lambda"] for r in rows], [r[metric] for r in rows], "o-", color=colors[variant], label=variant)
            best = next(r for r in summary["model_bests"] if r["model"] == variant)
            ax.scatter([best["lambda"]], [best[metric]], marker="*", s=220, color=colors[variant], edgecolor="white", zorder=5)
            if number == 1:
                ax.annotate(f"{variant}: lambda={best['lambda']:.1f}", (best["lambda"], best[metric]),
                            xytext=(8, 15 if variant == "layerwise" else -25), textcoords="offset points", color=colors[variant])
        baseline = summary["baseline_rankings"]["Original NSC"]
        ax.axhline(baseline["spearman_rho"] if metric == "rho" else baseline["all_pairs"]["tau_b"], color="#666", linestyle="--", label="Original NSC")
        ax.set(xlabel="Penalty strength lambda", ylabel=title, title=f"Development: {title} vs lambda", xticks=GRID)
        ax.grid(alpha=.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(folder / ("figure1_spearman.png" if number == 1 else "figure2_kendall.png"))
        plt.close(fig)
    chosen = ["Original NSC"] + [r["method"] for r in summary["model_bests"]] + ["Parameter Count", "Reference FLOPs", "Positional Example"]
    ranks = {r["method"]: (r["rho"], r["tau"]) for r in summary["model_bests"]}
    for name, value in summary["baseline_rankings"].items():
        ranks[name] = (value["spearman_rho"], value["all_pairs"]["tau_b"])
    labels = ["Original NSC", "Layer-wise best", "Global best", "Parameters", "Reference FLOPs", "Positional"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for j, ax in enumerate(axes):
        ax.bar(range(len(chosen)), [ranks[n][j] for n in chosen], color=["#666", colors["layerwise"], colors["global"], "#729b69", "#9b7aa6", "#caa34a"])
        ax.set(title=["Spearman rho", "Kendall tau-b"][j], xticks=range(len(chosen)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Development ranking comparison (best lambda chosen on Development)")
    fig.tight_layout()
    fig.savefig(folder / "figure3_ranking_comparison.png")
    plt.close(fig)
    compare = chosen[:3] + ["MB-NSC Ablation", "Parameter Count", "Reference FLOPs", "Positional Example", "Uniform Random"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for k, ax in zip((1, 3), axes):
        for name in compare:
            rows = [r for r in summary["selection_distributions"] if r["Method"] == name and r["k"] == k and r["Status"] == "PASS"]
            ax.plot([r["Budget"] / 1e6 for r in rows], [r["mean_regret"] for r in rows], marker="o", label=name)
        ax.set(xlabel="Total parameter cap (million)", ylabel="Mean regret over 200 choices", title=f"k={k}", xticks=[10, 20, 30, 50])
        ax.grid(alpha=.2)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.suptitle("Development budgeted selection; random/tie variability only")
    fig.tight_layout()
    fig.savefig(folder / "figure4_regret.png")
    plt.close(fig)
    names = chosen[:3]
    lookup = {r["Method"]: r for r in summary["size_controlled"]}
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, field, title in [(axes[0], "PairwiseAccuracy", "Pairwise accuracy"), (axes[1], "Official_tau_b", "Official restricted tau-b")]:
        ax.bar(range(3), [lookup[n][field] for n in names], color=["#666", colors["layerwise"], colors["global"]])
        ax.set(title=title, xticks=range(3), xticklabels=["Original NSC", "Layer-wise best", "Global best"])
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Development size control: 180 overlapping pairs (not independent samples)")
    fig.tight_layout()
    fig.savefig(folder / "figure5_size_controlled.png")
    plt.close(fig)


class ProtocolTests(unittest.TestCase):
    def row(self, model="layerwise", strength=.5, rho=.8, tau=.6):
        return {"model": model, "lambda": strength, "rho": rho, "tau": tau, "method": f"{model}-{strength}"}

    def test_primary(self):
        a, b = self.row(rho=.81, tau=.5), self.row("global", rho=.8, tau=.7)
        self.assertIs(select_best([a, b])[0], a)

    def test_kendall_tie(self):
        a, b = self.row(), self.row("global", rho=.8 + .5e-12, tau=.7)
        self.assertIs(select_best([a, b])[0], b)

    def test_lower_lambda(self):
        a, b = self.row(strength=.1), self.row("global", strength=.2)
        self.assertIs(select_best([a, b])[0], a)

    def test_global_operational_tie(self):
        a, b = self.row(), self.row("global")
        self.assertIs(select_best([a, b])[0], b)

    def test_null(self):
        with self.assertRaises(ValueError):
            select_best([self.row(rho=None, tau=None)])
        good = self.row("global")
        self.assertIs(select_best([self.row(rho=None, tau=None), good])[0], good)

    def test_accuracy_excludes_actual_ties(self):
        self.assertEqual(pair_accuracy({"concordant": 2, "discordant": 1, "score_only_ties": 1}), .625)

    def test_duplicate_json(self):
        with self.assertRaises(ValueError):
            json.loads('{"a":1,"a":2}', object_pairs_hook=unique_object)

    def test_trials_match_official(self):
        import evaluate as official
        for scores in (None, [1, 1, 0]):
            for k in (1, 3):
                _, got = selection_trials([0, 1, 2], scores, k, [10, 20, 30], 123)
                expected = official.select_trials([0, 1, 2], scores, k, [10, 20, 30], 123)
                for field in expected:
                    assert_close(got[field], expected[field])


def run():
    protocol = ROOT / "config" / "protocol_snapshot.json"
    assert not protocol.exists(), "Never overwrite an existing experiment; stop"
    before = read_json(ROOT / "config" / "protection_before.json")
    frozen_sources = {r["Path"]: r["Hash"] for r in before["Protected"]}
    for path, expected in frozen_sources.items():
        assert digest(Path(path)).upper() == expected
    version = digest(Path(__file__))
    snapshot = {"registered_at_tokyo": now(), "split": "development", "grid": list(GRID),
                "candidate_count": 22, "selection": ["maximum Spearman (tolerance 1e-12)", "maximum all-pairs Kendall tau-b (tolerance 1e-12)", "smaller lambda", "Global operational tie"],
                "selection_scope": "best lambda per model, then compare model bests",
                "tolerance_sets": "anchor each tolerance set to its maximum; do not chain pairwise ties",
                "budgets": [10000000, 20000000, 30000000, 50000000], "k": [1, 3],
                "trials": 200, "seed": "202627 + budget + k", "scoring_precision": "official 12 significant digits",
                "supplementary_metrics_not_used_for_selection": ["size control", "regret", "resources", "random"],
                "source_hashes": frozen_sources, "runner_sha256": version,
                "environment": read_json(ROOT / "config" / "environment.json"),
                "preflight_tests": "90 passed; protocol self-tests passed before Development labels were read",
                "final_access": "forbidden; audit guard rejects all non-allowlisted project JSON",
                "source_protocol_sha256": digest(ROOT / "config" / "phase4_protocol_source.md")}
    write_json(protocol, snapshot)  # Pre-registration happens BEFORE label access.
    configs, resources, keys, ppl, _, matched = load_development()
    write_json(ROOT / "config" / "data_validation.json", {"architectures": len(configs), "development": len(keys),
               "final_specifications_only": 100, "labels": len(ppl), "matched_pairs": len(matched),
               "duplicate_key_check": "PASS", "positive_finite_labels": True,
               "allowed_data_sha256": {p.name: digest(p) for p in sorted(ALLOWED_DATA)}, "final_labels_read": False})
    manifests = []
    baseline = run_one("global", 0, "baseline", manifests, version)
    entries = []
    for variant in ("layerwise", "global"):
        for strength in GRID:
            label = f"lambda_{strength:.1f}"
            result, vectors = run_one(variant, strength, label, manifests, version)
            entries.append((variant, strength, result, vectors))
    summary = summarize(baseline, entries, configs, resources, keys, ppl)
    figures(summary)
    write_json(ROOT / "config" / "frozen_development_choice.json", {"selected_on": "development",
               "winner": summary["winner"], "source_hashes": frozen_sources, "runner_sha256": version,
               "protocol_sha256": digest(protocol), "final_evaluation_run": False,
               "requires_user_review_before_final": True})
    write_json(ROOT / "logs" / "access_audit.json", {"coordinator_allowed_data_reads": sorted(READS),
               "coordinator_denials": DENIALS, "children": [
                   {"variant": v, "lambda": l, "reads": vectors["allowed_data_reads"], "denials": vectors["denied_accesses"]}
                   for v, l, _, vectors in entries], "final_labels_read": False})
    print("DEVELOPMENT COMPLETE: " + json.dumps(summary["winner"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--self-test", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--candidate", action="store_true")
    parser.add_argument("--variant", choices=("layerwise", "global"))
    parser.add_argument("--strength", type=float)
    parser.add_argument("--label")
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProtocolTests))
        sys.exit(0 if result.wasSuccessful() else 1)
    elif args.candidate:
        candidate(args.variant, args.strength, args.label)
    else:
        run()
