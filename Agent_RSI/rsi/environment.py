"""Deterministic task environment and versioned, declarative updates."""
from dataclasses import asdict, dataclass
from statistics import mean

ACTIONS = {
    "A": ("none", "none", "none"),
    "B": ("prompt", "ask_missing", "missing"),
    "C": ("memory", "refresh_cache", "stale"),
    "D": ("skill", "verify_total", "verify"),
}
FAULTS = ("missing", "stale", "verify", "none")


@dataclass(frozen=True)
class Harness:
    version: int = 0
    rules: tuple[str, ...] = ()


@dataclass(frozen=True)
class Episode:
    task_id: str
    task: dict
    trace: list[dict]
    feedback: dict
    labels: dict


def execute(case_id: int, fault: str, harness: Harness) -> Episode:
    if fault not in FAULTS:
        raise ValueError(f"Unknown scenario: {fault}")
    units = 2 + case_id % 13
    price = 10 + case_id % 23
    missing = fault == "missing"
    stale = fault == "stale"
    needs_check = fault == "verify"
    task = {
        "request": "Calculate the basket total using the current catalog. Verify the sum if requested.",
        "units": None if missing else units,
        "item": f"item-{case_id}",
        "verification_required": needs_check,
    }
    observed_units = task["units"]
    observed_price = price + 3 if stale else price
    trace = []
    if observed_units is None and "ask_missing" in harness.rules:
        observed_units = units
        trace.append({"action": "ask_quantity", "value": units})
    trace.append({"action": "read_cached_price", "value": observed_price})
    if "refresh_cache" in harness.rules:
        observed_price = price
        trace.append({"action": "read_current_price", "value": price})
    total = observed_price * (1 if observed_units is None else observed_units)
    checked = needs_check and "verify_total" in harness.rules
    if checked:
        trace.append({"action": "verify_arithmetic", "value": total})
    trace.append({"action": "submit", "total": total})
    # Independent verifier derives labels from actual state and observed actions.
    if observed_units is None:
        error = "missing"
        message = "A required input was unresolved when the agent submitted."
    elif observed_price != price:
        error = "stale"
        message = "The submitted calculation used an obsolete catalog quote."
    elif needs_check and not checked:
        error = "verify"
        message = "The required arithmetic verification was absent from the trace."
    else:
        error = "none"
        message = "The submission satisfies the task and its verification requirements."
    correct_total = total == units * price
    success = error == "none" and correct_total
    label = next(code for code, (_, _, kind) in ACTIONS.items() if kind == error)
    return Episode(str(case_id), task, trace, {
        "source": "independent_basket_verifier",
        "message": message, "success": success, "submitted_total_correct": correct_total,
    }, {
        "outcome": "success" if success else "failure",
        "error_location": {"missing": "input", "stale": "catalog", "verify": "verification", "none": "none"}[error],
        "root_cause": error,
        "recommended_action": label,
    })


def proposal(code: str) -> dict:
    component, operation, scope = ACTIONS[code]
    return {"choice": code, "component": component, "operation": operation, "scope": scope}


def evaluate(harness: Harness, case_ids: range) -> dict[str, float]:
    return {
        fault: mean(execute(case, fault, harness).feedback["success"] for case in case_ids)
        for fault in FAULTS
    }


def assess_update(harness: Harness, code: str, calibration: range) -> tuple[Harness, dict]:
    update = proposal(code)
    rule = update["operation"]
    prior = evaluate(harness, calibration)
    candidate = Harness(harness.version + 1, (*harness.rules, rule))
    trial = prior if code == "A" or rule in harness.rules else evaluate(candidate, calibration)
    accepted = mean(trial.values()) > mean(prior.values()) and all(
        trial[task] >= value for task, value in prior.items()
    )
    new = candidate if accepted else harness
    return new, {
        "before": asdict(harness), "proposal": update, "after": asdict(new),
        "candidate_diff": [] if code == "A" or rule in harness.rules else [{"add_rule": rule}],
        "applied_diff": [{"add_rule": rule}] if accepted else [],
        "accepted": accepted, "calibration_before": prior, "calibration_candidate": trial,
    }
