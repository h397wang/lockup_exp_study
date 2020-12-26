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
	"period_delta_pcnt signal_norm"
)


# I don't think it really matter what we use.
# We could use some mid point between LOW and HIGH
DAILY_POINT = "Close"
DROP_PCNT_THRESHOLD = -5


def format_dt_for_dfq(dt):
	return dt.strftime('%Y%m%d')


def format_dt_for_yf(dt):
	return dt.strftime("%Y-%m-%d")


def normalize_signal(df):
	v0 = df["Close"][0]
	df["Norm"] = df["Close"].apply(lambda x: 100 * (x - v0) / v0) 


class LockoutExpAnalyzer():
	def __init__(self, stocks_df, days_before, days_after=1, print_shit=False):
		self.stocks_df = stocks_df
		self.days_before = days_before
		self.days_after = days_after
		self.print_shit = print_shit
		self.results = []

		for indx, row in self.stocks_df.iterrows():
			res = self.analyze_stock(row)
			self.results.append(res)

		# Remove outliers when reporting average.
		deltas = sorted([r.period_delta_pcnt for r in self.results])
		cutoff = int(len(deltas) / 10)
		deltas_cutoff = deltas[cutoff:-cutoff]
		assert(len(deltas_cutoff) < len(self.results))
		period_delta_pcnt_avg = stats.mean(deltas_cutoff)
		outliers_removed = len(self.results) - len(deltas_cutoff)

		drop_happens = map(lambda r: 1 if r.period_delta_pcnt < 0 else 0, self.results)
		correctness = stats.mean(drop_happens)

		significant_drop_happens = map(
			lambda r: 1 if r.period_delta_pcnt < DROP_PCNT_THRESHOLD else 0,
			self.results
		)
		significant_correctness = stats.mean(significant_drop_happens)

		print(f"=======================================================")
		print(f"Report Summary")
		print(f"Average Period Delta: {period_delta_pcnt_avg:.1f}%")
		print(f"Drop Correctness: {100*correctness:.1f}%")
		print(f"Significant Drop Correctness ({DROP_PCNT_THRESHOLD}%): {100*significant_correctness:.1f}%")
		print(f"Number of data points: {len(self.results)}")
		print(f"Number of outliers removed: {outliers_removed}")


	def show_graph(self):
		for r in self.results:			
			plt.plot(r.signal_norm)
		plt.xlabel("Trading Days")
		plt.ylabel("Percentage Change")
		plt.title(f"Change From Days [-{self.days_before}, {self.days_after}] of Lockup Expiration")
		plt.grid(linestyle="--")
		plt.show()


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
		end_dt = lockout_exp_dt + datetime.timedelta(days=3 + self.days_after)
		
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
		days_after_dfq = format_dt_for_dfq(lockout_exp_dt + datetime.timedelta(days=self.days_after))

		"""
		pre_exp_period_df = data.query(f"Date <= {lockout_exp_dfq}")
		pre_exp_highest = pre_exp_period_df["High"].max()
		pre_exp_lowest = pre_exp_period_df["Low"].min()
		pre_exp_mean = pre_exp_period_df["Close"].mean()
		"""

		post_exp_df = data.query(f"Date <= {days_after_dfq}").iloc[-1]
		pre_exp_df = data.iloc[0]
		
		# Find the change in closing prices between the start date, and the day after exp.
		price_delta = post_exp_df["Close"] - pre_exp_df["Close"]
		price_delta_pcnt = 100 * price_delta / pre_exp_df["Close"]
		
		if self.print_shit:
			print(f"=============================================")
			print(f"Ticker: {ticker}")
			
			print(f"{self.days_before} Days Before Exp")
			print(f"\tClose: {pre_exp_df['Close']:.1f}$")
			print(f"\tHigh: {pre_exp_df['High']:.1f}$")
			print(f"\tLow: {pre_exp_df['Low']:.1f}$")

			print(f"Day After Exp")
			print(f"\tClose: {post_exp_df['Close']:.1f}$")
			print(f"\tHigh: {post_exp_df['High']:.1f}$")
			print(f"\tLow: {post_exp_df['Low']:.1f}$")

			print(f"Delta")
			print(f"\tDelta Amount: {price_delta:.1f}$")
			print(f"\tDelta Percent: {price_delta_pcnt:.1f}%")

		normalize_signal(data)

		res = AnalysisResults(
			period_delta_pcnt = price_delta_pcnt,
			signal_norm = list(data["Norm"]),
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
		"--days-after",
		type=int,
		default=1,
		help="Number of (non trading) days after exp date. Must be > 1."
	)
	parser.add_argument(
		"--print-shit",
		dest="print_shit",
		action="store_true",
		help="Print individual stock info."
	)
	parser.add_argument(
		"--show-graph",
		dest="show_graph",
		action="store_true",
		help="Show graph of aggregate normalized signals."
	)
	args = parser.parse_args()
	stocks_df = pd.read_csv(args.csv_file)
	
	lea = LockoutExpAnalyzer(
		stocks_df,
		args.days_before,
		args.days_after,
		args.print_shit
	)

	if args.show_graph:
		lea.show_graph()


if __name__ == "__main__":
	main()
