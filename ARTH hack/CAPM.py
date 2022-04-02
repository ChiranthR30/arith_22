import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from scipy import stats
import plotly.figure_factory as ff
import plotly.graph_objects as go

RISK_FREE_RATE = 0.05

MONTHS_IN_YEAR = 12

ASIANPAINT = pd.read_csv("D:\VScode\Python\ARTH hack\\asianpaints.csv")
HDFC_bank = pd.read_csv("D:\VScode\Python\ARTH hack\\HDFC.csv")
prakash = pd.read_csv("D:\VScode\Python\ARTH hack\\prakash.csv")
nifty = pd.read_csv("D:\VScode\Python\ARTH hack\\^NSEI.csv")
ASIANPAINT = ASIANPAINT[["Date", "Adj Close"]]
HDFC_bank = HDFC_bank[["Date", "Adj Close"]]
prakash = prakash[["Date", "Adj Close"]]
nifty = nifty[["Date", "Adj Close"]]
ASIANPAINT.rename(columns={"Adj Close": "ASIANPAINT"}, inplace = True)
HDFC_bank.rename(columns={"Adj Close": "HDFC_bank"}, inplace = True)
prakash.rename(columns={"Adj Close": "prakash"}, inplace = True)
nifty.rename(columns={"Adj Close": "nifty"}, inplace= True)
stocks_df = pd.concat([ASIANPAINT, HDFC_bank.drop(columns=["Date"]), prakash.drop(columns=["Date"]), nifty.drop(columns=["Date"])], axis = 1)
stocks_df = stocks_df.sort_values(by = ['Date'])
stocks_df.head(5)


def normalize_stocks(df):
    df_ = df.copy() 
    for stock in df_.columns[1:]:
        df_[stock] = df_[stock] / df_.loc[0, stock]
    return df_
norm_stocks_df = normalize_stocks(stocks_df)
norm_stocks_df.head(5)

fig = px.line(title = "Normalized stock prices")
 

for stock in norm_stocks_df.columns[1:]:
    fig.add_scatter(x = norm_stocks_df["Date"], y = norm_stocks_df[stock], name = stock)
fig.show()


def daily_return_estimator(df):
    df_daily_return = df.copy()
    for i in df.columns[1:]:
        for j in range(1, len(df)):
            df_daily_return[i][j] = ((df[i][j]- df[i][j-1])/df[i][j-1]) * 100
        df_daily_return.loc[0, i] = 0
 
    return df_daily_return
stocks_daily_return = daily_return_estimator(stocks_df)
stocks_daily_return.head(5)

stocks_daily_return.boxplot(figsize=(12, 10), grid=False)
plt.title("Daily returns of the stocks")
plt.show()

class CAPM:

    def __init__(self, stocks, start_date, end_date):
        self.data = None
        self.stocks = stocks
        self.start_date = start_date
        self.end_date = end_date

    def download_data(self):
        data = {}

        for stock in self.stocks:
            ticker = yf.download(stock, self.start_date, self.end_date)
            data[stock] = ticker['Adj Close']

        return pd.DataFrame(data)

    def initialize(self):
        stock_data = self.download_data()

        stock_data = stock_data.resample('M').last()

        self.data = pd.DataFrame({'s_adjclose': stock_data[self.stocks[0]],
                                  'm_adjclose': stock_data[self.stocks[1]]})


        self.data[['s_returns', 'm_returns']] = np.log(self.data[['s_adjclose', 'm_adjclose']] /
                                                       self.data[['s_adjclose', 'm_adjclose']].shift(1))


        self.data = self.data[1:]

    def calculate_beta(self):

        covariance_matrix = np.cov(self.data["s_returns"], self.data["m_returns"])

        beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
        print("Beta from formula: ", beta)

    def regression(self):

        beta, alpha = np.polyfit(self.data['m_returns'], self.data['s_returns'], deg=1)
        print("Beta from regression: ", beta)

        expected_return = RISK_FREE_RATE + beta * (self.data['m_returns'].mean()*MONTHS_IN_YEAR
                                                   - RISK_FREE_RATE)
        print("Expected return: ", expected_return)
        self.plot_regression(alpha, beta)

    def plot_regression(self, alpha, beta):
        fig, axis = plt.subplots(1, figsize=(20, 10))
        axis.scatter(self.data["m_returns"], self.data['s_returns'],
                     label="Data Points")
        axis.plot(self.data["m_returns"], beta * self.data["m_returns"] + alpha,
                  color='red', label="CAPM Line")
        plt.title('Capital Asset Pricing Model, finding alpha and beta')
        plt.xlabel('Market return $R_m$', fontsize=18)
        plt.ylabel('Stock return $R_a$')
        plt.text(0.08, 0.05, r'$R_a = \beta * R_m + \alpha$', fontsize=18)
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    a = 'ASIANPAINT.NS'
    capm = CAPM([a, '^NSEI'], '2010-01-01', '2020-01-01')
    capm.initialize()
    capm.calculate_beta()
    capm.regression()


