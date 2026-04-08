from typing import Dict, Any
import random

class FundSageEnv:

    def __init__(self):
        self.done = False
        self.current_step = 0
        self.max_steps = 3

    def reset(self) -> Dict[str, Any]:
        self.done = False
        self.current_step = 0

        self.state_data = {
            "risk_profile": random.choice(["low", "medium", "high"]),
            "investment_horizon": random.choice(["short", "medium", "long"]),
            "funds": [
                {"name": "Large Cap Fund", "risk": "high", "expected_return": 0.13, "expense_ratio": 0.018, "volatility": 0.17},
                {"name": "Debt Fund", "risk": "low", "expected_return": 0.07, "expense_ratio": 0.006, "volatility": 0.05},
                {"name": "Balanced Fund", "risk": "medium", "expected_return": 0.10, "expense_ratio": 0.012, "volatility": 0.10}
            ]
        }

        return self.state_data

    def step(self, action: Dict[str, Any]):
        self.current_step += 1

        selected_funds = action.get("selected_funds", [])
        allocation = action.get("allocation", [1/len(selected_funds)] * len(selected_funds)) if selected_funds else []

        reward = 0

        for i, fund in enumerate(self.state_data["funds"]):
            if fund["name"] in selected_funds:

                weight = allocation[selected_funds.index(fund["name"])]

                # ✅ Risk alignment
                if fund["risk"] == self.state_data["risk_profile"]:
                    reward += 0.4 * weight
                else:
                    reward -= 0.2 * weight

                # ✅ Return contribution
                reward += fund["expected_return"] * weight

                # ✅ Expense penalty
                reward -= fund["expense_ratio"] * weight

                # ✅ Volatility penalty (important for low risk)
                if self.state_data["risk_profile"] == "low":
                    reward -= fund["volatility"] * weight

                # ✅ Horizon logic
                if self.state_data["investment_horizon"] == "long":
                    reward += fund["expected_return"] * 0.2 * weight

        # ✅ Diversification bonus
        if len(selected_funds) > 1:
            reward += 0.2

        # ✅ Penalty for no action
        if len(selected_funds) == 0:
            reward -= 0.5

        # ✅ Termination
        if self.current_step >= self.max_steps:
            self.done = True

        return self.state(), round(reward, 4), self.done, {}

    def state(self):
        return self.state_data