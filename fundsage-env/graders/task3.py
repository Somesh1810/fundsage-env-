def grade_task_3(action, state):
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
    w_return     = sum(f["expected_return"] * w for f, w in funds)
    return_score = min(1.0, max(0.0, (w_return - 0.10) / (0.15 - 0.10)))
    w_expense    = sum(f["expense_ratio"] * w for f, w in funds)
    expense_score= 1.0 if w_expense <= 0.015 else max(0.0, 1.0 - (w_expense - 0.015) * 20)
    w_vol        = sum(f["volatility"] * w for f, w in funds)
    vol_score    = 1.0 if w_vol <= 0.18 else max(0.0, 1.0 - (w_vol - 0.18) * 10)
    tax_score    = 1.0 if any("ELSS" in f["name"] or "Index" in f["name"] for f, _ in funds) else 0.4
    score = (0.40 * return_score + 0.25 * expense_score + 0.20 * vol_score + 0.15 * tax_score) * (1.0 if sum_ok else 0.8)
    return round(min(1.0, score), 4)
