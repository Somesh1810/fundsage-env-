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
        reward = 0

        for fund in self.state_data["funds"]:
            if fund["name"] in selected_funds:
                if fund["risk"] == self.state_data["risk_profile"]:
                    reward += 0.4
                reward += fund["expected_return"]
                reward -= fund["expense_ratio"]

        if self.current_step >= self.max_steps:
            self.done = True

        return self.state(), reward, self.done, {}

    def state(self):
        return self.state_data
