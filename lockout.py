"""
Make money off of lockup expiry dates.

A lockup period prevents insiders from trading for some time after IPO.
When this lockup expires, a massive amount of shares become available.
If the company is one of those super hype, sexy, overpriced tech companies
with low float. It's likely that it makes stupid runups and has a huge PE
after IPO. Insiders will naturally seek to sell their stocks at such outrageous
prices. The anticipation of this sell off causes the stock price to drop
well in advance, a few weeks. There may or may not be a strong price drop
on the day after expiration.

We do a backtest anslysis of what happens when you take a short position
some X days bfore the lockup expiration date, and exit the day after.

Example Usage:
	python3 lockout.py meme_lockouts.csv --days-before 7

Play around with the days-before value to optimize entry position.

The data strongly supports the thesis, when we consider unicorn companies like
the ones listed in meme_lockouts.csv.
"""

import argparse
import csv
from collections import namedtuple
import datetime
import matplotlib.pyplot as plt
import statistics as stats
import pandas as pd
import yfinance as yf


AnalysisResults = namedtuple(
	"AnalysisResults",
	"period_delta_pcnt"
)


# I don't think it really matter what we use.
# We could use some mid point between LOW and HIGH
DAILY_POINT = "Close"


def format_dt_for_dfq(dt):
	return dt.strftime('%Y%m%d')


def format_dt_for_yf(dt):
	return dt.strftime("%Y-%m-%d")


# TODO: visualize shit.
def normalize_signal(col):
	pass


class LockoutExpAnalyzer():
	def __init__(self, stocks_df, days_before, print_shit=False):
		self.stocks_df = stocks_df
		self.days_before = days_before
		self.print_shit = print_shit

		results = []
		for indx, row in self.stocks_df.iterrows():
			res = self.analyze_stock(row)
			if res:
				results.append(res)

		period_delta_pcnt_avg = stats.mean(r.period_delta_pcnt for r in results)
		drop_happens = map(lambda x: 1 if x < 0 else 0, [r.period_delta_pcnt for r in results])
		correctness = stats.mean(drop_happens)

		print(f"=========================")
		print(f"Aggregate Summary")
		print(f"Average Period Delta: {period_delta_pcnt_avg:.1f}%")
		print(f"How often are we correct about the drop: {100 * correctness:.1f}%")
		print(f"Number of data points: {len(results)}")


	def analyze_stock(self, row):
		"""
		Inputs:
			row: Ticker,Exp,Initial
		Returns:
			AnalysisResults
		"""
		ticker = row["Ticker"]
		lockout_exp_date = row["Exp"]
		if "Initial" in row:
			ipo_price = row["Initial"]
		else:
			ipo_price = 0

		# Parse input date into datetime object.
		lockout_exp_dt = datetime.datetime.strptime(lockout_exp_date, "%m/%d/%Y")

		# Determine start and end points analysis interval.
		start_dt = lockout_exp_dt - datetime.timedelta(days=self.days_before)
		end_dt = lockout_exp_dt + datetime.timedelta(days=5)
		
		# Format date to yfinance format: yyyy-mm-dd.
		lockout_exp_yf = format_dt_for_yf(lockout_exp_dt)
		start_yf = format_dt_for_yf(start_dt)
		end_yf = format_dt_for_yf(end_dt)
		
		# Download data from yfinance.
		# Data format: Open,High,Low,Close,Adj,Close,Volume.
		data = yf.download(ticker, start_yf, end_yf)

		# Format datetime objects for DF query.
		lockout_exp_dfq = format_dt_for_dfq(lockout_exp_dt)
		start_period_dfq = format_dt_for_dfq(start_dt)

		# Calculate the mean, and range in the period 1 week pre exp.
		pre_exp_df = data.query(f"Date <= {lockout_exp_dfq}")
		pre_exp_highest = pre_exp_df["High"].max()
		pre_exp_lowest = pre_exp_df["Low"].min()
		pre_exp_mean = pre_exp_df["Close"].mean()

		post_exp_df = data.query(f"Date > {lockout_exp_dfq}").iloc[0]
		post_exp_price_close = post_exp_df["Close"]
		post_exp_price_high = post_exp_df["High"]
		post_exp_price_low = post_exp_df["Low"]
		
		# Find the change in closing prices between the start date, and the day after exp.
		pre_exp_period_start_price = data["Close"][0]
		period_post_exp_price_delta = post_exp_price_close - pre_exp_period_start_price
		period_post_exp_price_delta_pcnt = 100 * period_post_exp_price_delta / pre_exp_period_start_price
		
		if self.print_shit:
			print(f"=============================================")
			print(f"Ticker: {ticker}")
			print(f"Day After Exp")
			print(f"\tClose: {post_exp_price_close:.1f}$")
			print(f"\tHigh: {post_exp_price_high:.1f}$")
			print(f"\tLow: {post_exp_price_low:.1f}$")
			
			print(f"Entry {self.days_before} Days Pre Exp Date")
			print(f"\tClose: {pre_exp_period_start_price:.1f}$")
			print(f"\tDelta Amount: {period_post_exp_price_delta:.1f}$")
			print(f"\tDelta Percent: {period_post_exp_price_delta_pcnt:.1f}%")

		res = AnalysisResults(
			period_delta_pcnt = period_post_exp_price_delta_pcnt,
		)
		return res

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"csv_file",
		type=str,
		help="Path to csv file containing tickers and lockup exp dates."
	)
	parser.add_argument(
		"--days-before",
		type=int,
		default=7,
		help="Number of (non trading) days before exp date."
	)
	parser.add_argument(
		"--print-shit",
		dest="print_shit",
		action="store_true",
		help="Print individual stock info."
	)
	args = parser.parse_args()
	stocks_df = pd.read_csv(args.csv_file)
	
	LockoutExpAnalyzer(stocks_df, args.days_before, args.print_shit)


if __name__ == "__main__":
	main()
