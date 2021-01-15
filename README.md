Make money off of lockup expiry dates.

A lockup period restricts insiders from trading for some time after IPO.
When this lockup expires, a significant amount of shares become available.
The volume varies case by case of course.

If the company is one of those super hype, sexy, overpriced tech companies
with low float. It's likely that it makes stupid runups post IPO and and
ends up with a huge PE. Insiders will naturally seek to sell their stocks
at such outrageous prices. The anticipation of this sell off causes the stock price to drop
well in advance, typically a week. There may or may not be a strong price drop
on the day after expiration because it depends on the insiders creating that sell pressure.
Sometimes, insiders end up buying more of the stock.

We do a backtest anslysis of what happens when you take a short position
some X days bfore the lockup expiration date, and exit the day after.
We play around with the days-before value to optimize entry position, but it looks like
8 or 9 days is ideal, and an exit can happen before the exp date.

The data strongly supports the thesis, when we consider unicorn companies like
the ones listed in meme_lockouts.csv. These companies are what I deem as sexy,
overpriced, and with huge hype. I also see them mentioned a lot of wsb and RH. This list
does not show any bias because I did not know what kind of historical behavior 
a particular stock had during their lockup exp period.

Example Usage:
	python3 lockout.py meme_lockouts.csv --days-before 7 --show-graph --print-shit

The results show an average drop of -11%, with a correct drop prediction over 80% of the time.

![alt text](https://github.com/h397wang/lockup_exp_study/blob/masterb/meme_lockups.png?raw=true)
