def grade_task_1(action, state):
    available = {f["name"]: f for f in state.get("available_funds", [])}
    selected  = action.get("selected_funds", [])
    alloc     = action.get("allocation", [])
    if not selected:
        return 0.0
    if not alloc or len(alloc) != len(selected):
        alloc = [1.0 / len(selected)] * len(selected)
    sum_penalty = 1.0 if abs(sum(alloc) - 1.0) < 0.05 else 0.7
    low_weight = sum(w for n, w in zip(selected, alloc) if available.get(n, {}).get("risk") == "low")
    return round(min(1.0, low_weight * sum_penalty), 4)
