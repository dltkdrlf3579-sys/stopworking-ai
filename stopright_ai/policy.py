from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any


def load_policy(config: Any) -> str:
    path = Path(config.get("policy", "current_policy_path", fallback="policies/current_policy.md"))
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path.resolve()}")
    return path.read_text(encoding="utf-8")


def save_policy(config: Any, policy_text: str) -> None:
    path = Path(config.get("policy", "current_policy_path", fallback="policies/current_policy.md"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(policy_text, encoding="utf-8")


def apply_policy_operations(current_policy: str, operations: list[dict], config: Any) -> tuple[str, str]:
    max_operations = config.getint("policy", "candidate_max_operations", fallback=3)
    if not operations:
        raise ValueError("candidate patch has no operations")
    if len(operations) > max_operations:
        raise ValueError(f"candidate patch has too many operations: {len(operations)} > {max_operations}")

    lines = current_policy.splitlines()
    applied = []

    for idx, operation in enumerate(operations, start=1):
        if not isinstance(operation, dict):
            raise ValueError(f"operation #{idx} is not an object")

        op = str(operation.get("op", "")).strip()
        if op == "insert_after":
            anchor = str(operation.get("anchor", "")).rstrip()
            text = str(operation.get("text", "")).rstrip()
            if not anchor or not text:
                raise ValueError(f"operation #{idx} insert_after requires anchor and text")
            line_index = find_exact_line(lines, anchor)
            lines.insert(line_index + 1, text)
            applied.append(f"insert_after: {anchor} -> {text}")
        elif op == "replace_line":
            target = str(operation.get("target", "")).rstrip()
            replacement = str(operation.get("replacement", "")).rstrip()
            if not target or not replacement:
                raise ValueError(f"operation #{idx} replace_line requires target and replacement")
            ensure_safe_to_modify(target)
            line_index = find_exact_line(lines, target)
            lines[line_index] = replacement
            applied.append(f"replace_line: {target} -> {replacement}")
        elif op == "delete_line":
            target = str(operation.get("target", "")).rstrip()
            if not target:
                raise ValueError(f"operation #{idx} delete_line requires target")
            ensure_safe_to_modify(target)
            line_index = find_exact_line(lines, target)
            del lines[line_index]
            applied.append(f"delete_line: {target}")
        else:
            raise ValueError(f"operation #{idx} has unsupported op: {op}")

    return "\n".join(lines) + ("\n" if current_policy.endswith("\n") else ""), "\n".join(applied)


def should_promote(base_metrics: dict, candidate_metrics: dict, config: Any) -> tuple[bool, str]:
    min_gain = config.getfloat("policy", "promotion_min_score_gain", fallback=0.01)
    max_fn_regression = config.getint("policy", "promotion_max_fn_regression", fallback=0)
    max_fp_regression = config.getint("policy", "promotion_max_fp_regression", fallback=5)

    score_gain = candidate_metrics.get("score", 0) - base_metrics.get("score", 0)
    fn_regression = candidate_metrics.get("fn_true_as_false", 0) - base_metrics.get("fn_true_as_false", 0)
    fp_regression = candidate_metrics.get("fp_false_as_true", 0) - base_metrics.get("fp_false_as_true", 0)

    if score_gain < min_gain:
        return False, f"score_gain {score_gain:.4f} < {min_gain:.4f}"
    if fn_regression > max_fn_regression:
        return False, f"FN regression {fn_regression} > {max_fn_regression}"
    if fp_regression > max_fp_regression:
        return False, f"FP regression {fp_regression} > {max_fp_regression}"
    return True, f"promoted: score_gain={score_gain:.4f}, fn_regression={fn_regression}, fp_regression={fp_regression}"


def validate_candidate_policy(current_policy: str, candidate_policy: str, config: Any) -> tuple[bool, str]:
    min_ratio = config.getfloat("policy", "candidate_min_policy_length_ratio", fallback=0.90)
    min_length = config.getint("policy", "candidate_min_policy_chars", fallback=1500)
    max_changed_lines = config.getint("policy", "candidate_max_changed_lines", fallback=12)
    max_deleted_lines = config.getint("policy", "candidate_max_deleted_lines", fallback=3)
    current_length = len(current_policy.strip())
    candidate_length = len(candidate_policy.strip())

    required_terms = [
        "진성",
        "가성",
        "판정",
        "판단근거",
        "확신도",
        "review_needed",
        "applied_step",
        "decisive_evidence",
    ]
    missing = [term for term in required_terms if term not in candidate_policy]
    if missing:
        return False, "candidate policy missing required terms: " + ", ".join(missing)

    threshold = max(min_length, int(current_length * min_ratio))
    if candidate_length < threshold:
        return False, f"candidate policy too short: {candidate_length} < {threshold}"

    changed = count_changed_lines(current_policy, candidate_policy)
    if changed["deleted"] > max_deleted_lines:
        return False, f"candidate policy deleted too many lines: {changed['deleted']} > {max_deleted_lines}"
    if changed["changed"] > max_changed_lines:
        return False, f"candidate policy changed too many lines: {changed['changed']} > {max_changed_lines}"

    return True, "candidate policy valid"


def find_exact_line(lines: list[str], target: str) -> int:
    matches = [idx for idx, line in enumerate(lines) if line.rstrip() == target.rstrip()]
    if not matches:
        raise ValueError(f"target line not found in current policy: {target}")
    if len(matches) > 1:
        raise ValueError(f"target line is ambiguous in current policy: {target}")
    return matches[0]


def ensure_safe_to_modify(line: str) -> None:
    protected_terms = [
        "판정",
        "판단근거",
        "확신도",
        "review_needed",
        "applied_step",
        "decisive_evidence",
        "[출력 형식]",
        "[입력 데이터]",
    ]
    if any(term in line for term in protected_terms):
        raise ValueError(f"protected policy line cannot be modified: {line}")


def count_changed_lines(current_policy: str, candidate_policy: str) -> dict[str, int]:
    current_lines = normalize_policy_lines(current_policy)
    candidate_lines = normalize_policy_lines(candidate_policy)
    diff = list(difflib.ndiff(current_lines, candidate_lines))
    added = sum(1 for line in diff if line.startswith("+ ") and line[2:].strip())
    deleted = sum(1 for line in diff if line.startswith("- ") and line[2:].strip())
    return {"added": added, "deleted": deleted, "changed": added + deleted}


def normalize_policy_lines(policy: str) -> list[str]:
    return [line.rstrip() for line in policy.splitlines() if line.strip()]


def make_policy_diff(current_policy: str, candidate_policy: str) -> str:
    diff = difflib.unified_diff(
        current_policy.splitlines(),
        candidate_policy.splitlines(),
        fromfile="current_policy.md",
        tofile="candidate_policy.md",
        lineterm="",
    )
    return "\n".join(diff)
