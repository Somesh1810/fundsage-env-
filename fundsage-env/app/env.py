import random
from typing import Any

AVAILABLE_FUNDS = [
    {"name": "Equity Growth Fund",    "risk": "high",   "expected_return": 0.15, "expense_ratio": 0.02, "volatility": 0.20},
    {"name": "Large Cap Fund",        "risk": "high",   "expected_return": 0.13, "expense_ratio": 0.018,"volatility": 0.17},
    {"name": "Hybrid Balanced Fund",  "risk": "medium", "expected_return": 0.10, "expense_ratio": 0.015,"volatility": 0.12},
    {"name": "Conservative Hybrid",   "risk": "medium", "expected_return": 0.09, "expense_ratio": 0.013,"volatility": 0.10},
    {"name": "Debt Fund",             "risk": "low",    "expected_return": 0.07, "expense_ratio": 0.01, "volatility": 0.05},
    {"name": "Liquid Fund",           "risk": "low",    "expected_return": 0.06, "expense_ratio": 0.008,"volatility": 0.02},
    {"name": "ELSS Tax Saver",        "risk": "high",   "expected_return": 0.14, "expense_ratio": 0.02, "volatility": 0.18},
    {"name": "Index Fund Nifty 50",   "risk": "medium", "expected_return": 0.11, "expense_ratio": 0.005,"volatility": 0.14},
]

RISK_LEVELS    = ["low", "medium", "high"]
GOALS          = ["capital_preservation", "wealth_creation", "tax_saving", "retirement"]
MARKET_TRENDS  = ["bearish", "neutral", "bullish"]


class FundSageEnv:
    """
    OpenEnv-compliant environment for mutual fund recommendation.

    Observation (state):
        user_profile  – age, income, risk_tolerance, investment_goal, horizon_years
        market_state  – trend, inflation_rate
        available_funds – list of fund dicts

    Action:
        {
          "selected_funds": ["Fund Name A", "Fund Name B"],
          "allocation":     [0.6, 0.4]          # must sum to 1.0
        }

    Reward  (0.0 – 1.0):
        0.40 × risk_alignment
      + 0.30 × return_score
      + 0.20 × diversification_score
      - 0.10 × expense_penalty
    """

    # ------------------------------------------------------------------ #
    def __init__(self):
        self.available_funds = AVAILABLE_FUNDS
        self.user: dict  = {}
        self.market: dict = {}
        self._step_count  = 0

    # ------------------------------------------------------------------ #
    def reset(self) -> dict:
        self.user = {
            "age":               random.randint(22, 58),
            "income":            random.choice([300000, 500000, 800000, 1200000, 2000000]),
            "risk_tolerance":    random.choice(RISK_LEVELS),
            "investment_goal":   random.choice(GOALS),
            "horizon_years":     random.randint(1, 15),
        }
        self.market = {
            "trend":         random.choice(MARKET_TRENDS),
            "inflation_rate": round(random.uniform(4.0, 8.0), 2),
        }
        self._step_count = 0
        return self.state()

    # ------------------------------------------------------------------ #
    def state(self) -> dict:
        return {
            "user_profile":    self.user,
            "market_state":    self.market,
            "available_funds": self.available_funds,
            "step_count":      self._step_count,
        }

    # ------------------------------------------------------------------ #
    def step(self, action: dict) -> dict:
        reward = self._calculate_reward(action)
        self._step_count += 1
        done   = True  # single-step recommendation task

        return {
            "state":  self.state(),
            "reward": round(reward, 4),
            "done":   done,
            "info": {
                "risk_alignment":      self._risk_alignment(action),
                "return_score":        self._return_score(action),
                "diversification":     self._diversification_score(action),
                "expense_penalty":     self._expense_penalty(action),
            },
        }

    # ------------------------------------------------------------------ #
    # Reward sub-components
    # ------------------------------------------------------------------ #
    def _resolve_funds(self, action: dict) -> list[dict]:
        names = action.get("selected_funds", [])
        return [f for f in self.available_funds if f["name"] in names]

    def _risk_alignment(self, action: dict) -> float:
        funds = self._resolve_funds(action)
        if not funds:
            return 0.0
        alloc     = action.get("allocation", [1.0 / len(funds)] * len(funds))
        user_risk = self.user.get("risk_tolerance", "medium")
        score = 0.0
        for fund, w in zip(funds, alloc):
            if fund["risk"] == user_risk:
                score += 1.0 * w
            elif abs(RISK_LEVELS.index(fund["risk"]) - RISK_LEVELS.index(user_risk)) == 1:
                score += 0.5 * w
        return min(1.0, score)

    def _return_score(self, action: dict) -> float:
        funds = self._resolve_funds(action)
        if not funds:
            return 0.0
        alloc   = action.get("allocation", [1.0 / len(funds)] * len(funds))
        avg_ret = sum(f["expected_return"] * w for f, w in zip(funds, alloc))
        # Normalize: 0.06 (min) to 0.15 (max)
        return min(1.0, max(0.0, (avg_ret - 0.06) / (0.15 - 0.06)))

    def _diversification_score(self, action: dict) -> float:
        funds = self._resolve_funds(action)
        n     = len(funds)
        if n == 0:
            return 0.0
        # More funds + different risk buckets = better diversification
        risk_buckets = len({f["risk"] for f in funds})
        return min(1.0, (n / 4) * 0.5 + (risk_buckets / 3) * 0.5)

    def _expense_penalty(self, action: dict) -> float:
        funds = self._resolve_funds(action)
        if not funds:
            return 1.0
        alloc   = action.get("allocation", [1.0 / len(funds)] * len(funds))
        avg_exp = sum(f["expense_ratio"] * w for f, w in zip(funds, alloc))
        # Normalize: 0.005 → penalty 0, 0.02 → penalty 1
        return min(1.0, max(0.0, (avg_exp - 0.005) / (0.02 - 0.005)))

    def _calculate_reward(self, action: dict) -> float:
        ra   = self._risk_alignment(action)
        ret  = self._return_score(action)
        div  = self._diversification_score(action)
        exp  = self._expense_penalty(action)

        reward = (
            0.40 * ra +
            0.30 * ret +
            0.20 * div -
            0.10 * exp
        )
        return max(0.0, min(1.0, reward))
