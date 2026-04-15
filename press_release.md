# Can yesterday's trading data actually predict tomorrow's stock returns?

## Hook
Every weekday, billions of shares change hands across U.S. stock exchanges.
Most Americans have money tied to those movements through retirement accounts
or savings plans, even if they never actively trade a single share. Whether
it is possible to look at today's market data and make a reliable guess about
tomorrow has been debated for decades.

## Problem Statement
Financial markets are highly noisy, and prediction models frequently overstate
accuracy due to methodological flaws like look-ahead bias, overlapping training
data, and survivorship bias. Determining if future price movements can be
mathematically anticipated from historical data remains a central challenge.

## Solution Description
Ten years of daily trading records (2015–2025) for five major U.S. companies
spanning four sectors were stored in MongoDB Atlas and used to train a Random
Forest model evaluated exclusively on the most recent 20% of trading days —
data the model had never seen. Performance is reported using RMSE, MAE, and R²
against a naive baseline that predicts zero change every day.

## Chart
[KDE plot: Distribution of Daily Percentage Price Changes by Company, 2015–2025]