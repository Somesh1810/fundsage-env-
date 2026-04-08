def grade_task_2(action, state):
    available = {f["name"]: f for f in state.get("available_funds", [])}
    selected  = action.get("selected_funds", [])
    alloc     = action.get("allocation", [])
    if not selected:
        return 0.0
    if not alloc or len(alloc) != len(selected):
        alloc = [1.0 / len(selected)] * len(selected)
    funds = [(available[n], w) for n, w in zip(selected, alloc) if n in available]
    if not funds:
        return 0.0
    sum_ok       = abs(sum(w for _, w in funds) - 1.0) < 0.05
    risk_buckets = len({f["risk"] for f, _ in funds})
    diversity    = min(1.0, risk_buckets / 2)
    w_return     = sum(f["expected_return"] * w for f, w in funds)
    return_score = 1.0 if 0.08 <= w_return <= 0.11 else max(0.0, 1.0 - abs(w_return - 0.095) * 10)
    medium_wt    = sum(w for f, w in funds if f["risk"] == "medium")
    medium_score = min(1.0, medium_wt * 2)
    score = (0.35 * diversity + 0.35 * return_score + 0.30 * medium_score) * (1.0 if sum_ok else 0.8)
    return round(min(1.0, score), 4)
