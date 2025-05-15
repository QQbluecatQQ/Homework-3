"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=365, gamma=0):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """

        import torch
        import torch.nn as nn
        from sklearn.preprocessing import StandardScaler
        from gurobipy import GRB

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        lookback_window = self.lookback  # e.g. 365
        feature_window = 30  # ML 預測期望報酬用的歷史天數
        hidden_size = 64
        epochs = 50
        learning_rate = 0.001

        returns = self.returns[assets].copy()

        # 特徵：過去 feature_window 天的日報酬率
        X, y = [], []
        for i in range(lookback_window, len(returns) - 1):
            past_window = returns.iloc[i - feature_window:i].values
            future_return = returns.iloc[i + 1].values  # 預測隔日報酬
            X.append(past_window)
            y.append(future_return)

        X = np.array(X)  # (samples, time, features)
        y = np.array(y)  # (samples, features)

        # normalize
        scaler_x = StandardScaler()
        X = X.reshape(X.shape[0], -1)
        X = scaler_x.fit_transform(X)
        X = X.reshape(-1, feature_window, len(assets))

        scaler_y = StandardScaler()
        y = scaler_y.fit_transform(y)

        X_train = torch.tensor(X, dtype=torch.float32).to(device)
        y_train = torch.tensor(y, dtype=torch.float32).to(device)

        # 定義簡單的 LSTM 模型
        class LSTMModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
                self.fc = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                _, (hn, _) = self.lstm(x)
                out = self.fc(hn.squeeze(0))
                return out

        model = LSTMModel(len(assets), hidden_size, len(assets)).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # 訓練模型
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")

        # 開始預測測試區段的期望報酬，並使用 Gurobi 求權重
        self.portfolio_weights.iloc[:, :] = 0
        model.eval()
        for i in range(lookback_window + feature_window, len(returns) - 1):
            test_window = returns.iloc[i - feature_window:i].values
            test_input = scaler_x.transform(test_window.reshape(1, -1)).reshape(1, feature_window, len(assets))
            test_input = torch.tensor(test_input, dtype=torch.float32).to(device)
            with torch.no_grad():
                pred_return = model(test_input).cpu().numpy().flatten()
            pred_return = scaler_y.inverse_transform(pred_return.reshape(1, -1)).flatten()

            # covariance matrix from past window
            cov = returns.iloc[i - feature_window:i].cov().values

            try:
                m = gp.Model()
                w = m.addMVar(len(assets), lb=0, ub=1, name="w")
                m.setObjective(pred_return @ w - self.gamma * w @ cov @ w, GRB.MAXIMIZE)
                m.addConstr(w.sum() == 1)
                m.setParam("OutputFlag", 0)
                m.optimize()

                weight_vec = pd.Series(0, index=self.price.columns)
                weight_vec[assets] = w.X
                self.portfolio_weights.iloc[i] = weight_vec
            except:
                continue            
            
        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


"""
Assignment Judge

The following functions will help check your solution.
"""


class AssignmentJudge:
    def __init__(self):
        self.mp = MyPortfolio(df, "SPY").get_results()
        self.Bmp = MyPortfolio(Bdf, "SPY").get_results()

    def plot_performance(self, price, strategy):
        # Plot cumulative returns
        _, ax = plt.subplots()
        returns = price.pct_change().fillna(0)
        (1 + returns["SPY"]).cumprod().plot(ax=ax, label="SPY")
        (1 + strategy[1]["Portfolio"]).cumprod().plot(ax=ax, label=f"MyPortfolio")

        ax.set_title("Cumulative Returns")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Returns")
        ax.legend()
        plt.show()
        return None

    def plot_allocation(self, df_weights):
        df_weights = df_weights.fillna(0).ffill()

        # long only
        df_weights[df_weights < 0] = 0

        # Plotting
        _, ax = plt.subplots()
        df_weights.plot.area(ax=ax)
        ax.set_xlabel("Date")
        ax.set_ylabel("Allocation")
        ax.set_title("Asset Allocation Over Time")
        plt.show()
        return None

    def report_metrics(self, price, strategy, show=False):
        df_bl = pd.DataFrame()
        returns = price.pct_change().fillna(0)
        df_bl["SPY"] = returns["SPY"]
        df_bl["MP"] = pd.to_numeric(strategy[1]["Portfolio"], errors="coerce")
        sharpe_ratio = qs.stats.sharpe(df_bl)

        if show == True:
            qs.reports.metrics(df_bl, mode="full", display=show)

        return sharpe_ratio

    def cumulative_product(self, dataframe):
        (1 + dataframe.pct_change().fillna(0)).cumprod().plot()

    def check_sharp_ratio_greater_than_one(self):
        if not self.check_portfolio_position(self.mp[0]):
            return 0
        if self.report_metrics(df, self.mp)[1] > 1:
            print("Problem 4.1 Success - Get 15 points")
            return 15
        else:
            print("Problem 4.1 Fail")
        return 0

    def check_sharp_ratio_greater_than_spy(self):
        if not self.check_portfolio_position(self.mp[0]):
            return 0
        if (
            self.report_metrics(Bdf, self.Bmp)[1]
            > self.report_metrics(Bdf, self.Bmp)[0]
        ):
            print("Problem 4.2 Success - Get 15 points")
            return 15
        else:
            print("Problem 4.2 Fail")
        return 0

    def check_portfolio_position(self, portfolio_weights):
        if (portfolio_weights.sum(axis=1) <= 1.01).all():
            return True
        print("Portfolio Position Exceeds 1. No Leverage.")
        return False

    def check_all_answer(self):
        score = 0
        score += self.check_sharp_ratio_greater_than_one()
        score += self.check_sharp_ratio_greater_than_spy()
        return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()

    if args.score:
        if ("one" in args.score) or ("spy" in args.score):
            if "one" in args.score:
                judge.check_sharp_ratio_greater_than_one()
            if "spy" in args.score:
                judge.check_sharp_ratio_greater_than_spy()
        elif "all" in args.score:
            print(f"==> total Score = {judge.check_all_answer()} <==")

    if args.allocation:
        if "mp" in args.allocation:
            judge.plot_allocation(judge.mp[0])
        if "bmp" in args.allocation:
            judge.plot_allocation(judge.Bmp[0])

    if args.performance:
        if "mp" in args.performance:
            judge.plot_performance(df, judge.mp)
        if "bmp" in args.performance:
            judge.plot_performance(Bdf, judge.Bmp)

    if args.report:
        if "mp" in args.report:
            judge.report_metrics(df, judge.mp, show=True)
        if "bmp" in args.report:
            judge.report_metrics(Bdf, judge.Bmp, show=True)

    if args.cumulative:
        if "mp" in args.cumulative:
            judge.cumulative_product(df)
        if "bmp" in args.cumulative:
            judge.cumulative_product(Bdf)
