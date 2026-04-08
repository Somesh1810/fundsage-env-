from env import FundSageEnv

env = FundSageEnv()

print("Running validation...\n")

# Test reset
state = env.reset()
assert isinstance(state, dict), "Reset must return dict"

# Test step
action = {"selected_funds": [f["name"] for f in state["funds"]]}
next_state, reward, done, info = env.step(action)

assert isinstance(next_state, dict), "Step must return state dict"
assert isinstance(reward, (int, float)), "Reward must be numeric"
assert isinstance(done, bool), "Done must be boolean"
assert isinstance(info, dict), "Info must be dict"

# Test state()
current_state = env.state()
assert isinstance(current_state, dict), "State must return dict"

print("✅ VALIDATION PASSED")
