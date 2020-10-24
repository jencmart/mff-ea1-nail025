import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_apple_stock.csv')
    df.head()
    sns.lineplot(df,x = 'AAPL_x', y = 'AAPL_y')
    # fig = px.line(df, x='AAPL_x', y='AAPL_y', title='Apple Share Prices over time (2014)')