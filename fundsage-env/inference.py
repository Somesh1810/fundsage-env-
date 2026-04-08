import os, json
from openai import OpenAI
from app.env import FundSageEnv

API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN if HF_TOKEN else "hf_placeholder")
env    = FundSageEnv()

SYSTEM_PROMPT = """You are a mutual fund advisor AI.
Given a user financial profile and available funds, recommend 1-3 funds with allocation.
Respond ONLY with valid JSON:
{"selected_funds": ["Fund Name A", "Fund Name B"], "allocation": [0.6, 0.4]}
Rules: allocation must sum to 1.0, only use fund names from the list."""

def llm_agent(state):
    user  = state["user_profile"]
    funds = state["available_funds"]
    fund_list = "\n".join(
        f'  - {f["name"]} | risk={f["risk"]} | return={f["expected_return"]:.0%} | expense={f["expense_ratio"]:.1%}'
        for f in funds
    )
    user_msg = f"""User Profile:
  Age: {user['age']}, Income: {user['income']}, Risk: {user['risk_tolerance']}
  Goal: {user['investment_goal']}, Horizon: {user['horizon_years']} years
Available Funds:\n{fund_list}
Recommend best portfolio allocation."""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":user_msg}],
            max_tokens=300, temperature=0.2,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"): raw = raw[4:]
        action = json.loads(raw.strip())
        sel   = action.get("selected_funds", [])
        alloc = action.get("allocation", [1.0/len(sel)]*len(sel))
        s     = sum(alloc)
        return {"selected_funds": sel, "allocation": [round(a/s,4) for a in alloc]}
    except:
        risk_map = {"low":["Debt Fund","Liquid Fund"],"medium":["Hybrid Balanced Fund","Index Fund Nifty 50"],"high":["Equity Growth Fund","ELSS Tax Saver"]}
        sel = risk_map.get(state["user_profile"]["risk_tolerance"], ["Hybrid Balanced Fund"])
        return {"selected_funds": sel, "allocation": [round(1.0/len(sel),4)]*len(sel)}

TASKS = [
    {"task_id":"easy_risk_match",       "difficulty":"easy",   "user_override":{"risk_tolerance":"low",    "investment_goal":"capital_preservation"}},
    {"task_id":"balanced_portfolio",    "difficulty":"medium", "user_override":{"risk_tolerance":"medium", "investment_goal":"wealth_creation"}},
    {"task_id":"high_return_optimized", "difficulty":"hard",   "user_override":{"risk_tolerance":"high",   "investment_goal":"wealth_creation"}},
]

print("[START]")
print(json.dumps({"event":"start","tasks":len(TASKS),"model":MODEL_NAME}))

total = 0.0
for task in TASKS:
    state = env.reset()
    env.user.update(task["user_override"])
    state  = env.state()
    action = llm_agent(state)
    result = env.step(action)
    total += result["reward"]
    print(f"[STEP] task_id={task['task_id']} difficulty={task['difficulty']} reward={result['reward']} done={result['done']} action={json.dumps(action)} info={json.dumps(result['info'])}")

print("[END]")
print(json.dumps({"event":"end","total_tasks":len(TASKS),"avg_reward":round(total/len(TASKS),4),"total_reward":round(total,4)}))
