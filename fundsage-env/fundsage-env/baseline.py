from env import FundSageEnv

print("🚀 Starting FundSage Baseline...\n")

env = FundSageEnv()
state = env.reset()

print("Initial State:", state)

selected = []
for fund in state["funds"]:
    if fund["risk"] == state["risk_profile"]:
        selected.append(fund["name"])

# fallback
if not selected:
    selected = [state["funds"][0]["name"]]

action = {
    "selected_funds": selected
}

print("Action Taken:", action)

total_reward = 0
done = False

while not done:
    state, reward, done, _ = env.step(action)
    total_reward += reward

print("\n✅ Final Reward:", round(total_reward, 4))
