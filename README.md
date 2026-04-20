# DS 4320 Project 2: Forecasting Next-Day Stock Returns

**Executive Summary**
This project provides an end-to-end data pipeline and machine learning analysis designed to forecast short-term stock price movements. It automates the programmatic acquisition of historical market data, establishes a robust secondary data storage system using MongoDB Atlas, and utilizes a Random Forest Regressor to evaluate whether engineered technical indicators can predict next-day percentage returns across multiple large-cap equities.

| Project Identity | Resource Links |
| :--- | :--- |
| **Name:** Michael Carlson | [Press Release File](press_release.md) |
| **NetID:** mjy7nw | [UVA OneDrive Data Directory](#) |
| **DOI:** [Insert DOI Badge Here] | [Pipeline Notebook](pipeline.ipynb) |
| **License:** [MIT](LICENSE) | [Pipeline Markdown](pipeline.md) |


## Problem Definition
**General Problem:** 3. Forecasting stock prices.

**Specific Problem Statement:** Can we build a rigorous, data-driven pipeline to predict the next-day percentage return of five major U.S. equities (AAPL, MSFT, JPM, XOM, UNH) using a decade of historical daily trading data and engineered technical indicators?

**Rationale for Refinement:** Attempting to predict exact future stock prices across the entire market is susceptible to non-stationary data drift (prices naturally trend upward over time). Refining the problem to focus on *percentage returns* rather than absolute prices isolates the scope, makes the data stationary, and allows the model's performance to be evaluated without the illusion of simply predicting an upward trend.

**Motivation for the Project:** Millions of individuals and institutional investors rely on the stock market for wealth generation. Financial markets are incredibly noisy, and predicting short-term movements is notoriously difficult. Building a robust pipeline to evaluate whether historical trading patterns contain actionable predictive power provides a mathematical baseline for understanding market volatility and tests the limits of historical data utility.

**Press Release:** [Can yesterday's trading data actually predict tomorrow's stock returns?](press_release.md)

---
## Domain Exposition
**Terminology Table:**
| Term | Definition |
| :--- | :--- |
| **yfinance** | An open-source Python library used to download historical market data from Yahoo Finance. |
| **Daily Return** | The percentage change in a stock's closing price from one trading day to the next. |
| **SMA** | Simple Moving Average; the unweighted mean of the previous *n* data points (e.g., 20-day or 50-day closing prices). |
| **RSI** | Relative Strength Index; a momentum oscillator measuring the speed and change of price movements, scaled from 0 to 100. |
| **MACD** | Moving Average Convergence Divergence; a trend-following momentum indicator calculated by subtracting the 26-period EMA from the 12-period EMA. |
| **Look-ahead Bias** | A critical error in quantitative analysis where information not yet available during the period being analyzed is included in the model. |

**Domain Explanation:**
This project operates within the domain of quantitative finance and time-series forecasting. Financial data is characterized by high noise-to-signal ratios, non-stationarity, and rapid reactions to external, unquantifiable events (news, macroeconomics). In this domain, even a model that predicts directionality slightly better than a random coin toss is considered highly valuable. Rigorous temporal validation is critical, as standard cross-validation techniques will leak future data into the past.

**Background Reading Folder:** [OneDrive Literature Repository](#) 

**Background Reading Table:**
| # | Title | Summary | Link |
|---|---|---|---|
| 1 | Fama (1970) – "Efficient Capital Markets: A Review of Theory and Evidence" | The foundational academic paper arguing that stock prices already reflect all available public information, which is the core theoretical reason why consistent prediction from historical data is difficult. Sets the intellectual context for the problem. | https://myuva-my.sharepoint.com/:b:/g/personal/mjy7nw_virginia_edu/IQBiAOm6o_d5S6dMrWE18OhkAYnOnssib88JO3gTlqjzC9A?e=7wSJUV |
| 2 | Bookmap (2024) – "Survivorship Bias in Market Data: What Traders Need to Know" | Explains how restricting analysis to companies that are still active today systematically inflates apparent historical returns and masks real risk. Directly relevant to the bias identification and mitigation requirements of this project. | https://myuva-my.sharepoint.com/:b:/g/personal/mjy7nw_virginia_edu/IQCE8ICVL97bQYhYl3EfDQMZASpcL9QVp-qJYS4SriApt60?e=c2aehM |
| 3 | Thakar (2020) – "Forecasting Time Series Data: Stock Price Analysis" | Covers why raw price levels are a poor prediction target, how to test whether a data series is stationary, and why percentage returns are the appropriate modeling target for this type of problem. Provides the statistical rationale for a key design decision in this project. | https://myuva-my.sharepoint.com/:b:/g/personal/mjy7nw_virginia_edu/IQAgmYymznd7RLIIUPVHQY-ZAaC7Wss5mUJWchnmoQTfieE?e=q76lDr |
| 4 | Yao (2025) – "Research on Machine Learning Based Stock Price Prediction Model" | A peer-reviewed comparison of Ridge Regression, Random Forest, and Gradient Boosting on real stock price data. Provides empirical benchmarks and a methodological template directly comparable to this project's pipeline. | https://myuva-my.sharepoint.com/:b:/g/personal/mjy7nw_virginia_edu/IQDCVphhUDiuSLuDY9kVY25OAbdrvVkEpE739-QPY1keU1s?e=JVnjVk |
| 5 | Bland (2020) – "yfinance Library: A Complete Guide" | Technical documentation of the data acquisition tool used in this project. Covers what data types are available, retrieval methods, known limitations, and reliability considerations that inform the provenance discussion. | https://myuva-my.sharepoint.com/:b:/g/personal/mjy7nw_virginia_edu/IQAVa0epoiYnT6WJhz2eDx00AWxH0aTNFzmtdFgTGe3fc_c?e=7QwYb7 |

## Data Creation

> ### 🛠 Data Provenance & Reproducibility
> Raw trading data is programmatically acquired via the `yfinance` API, pulling historical Open, High, Low, Close, and Volume (OHLCV) data directly from Yahoo Finance for the period 2015-01-01 to 2025-01-01. The pipeline is designed to be **reproducible**; the data is processed in-memory to calculate rolling technical indicators before being ingested into a secondary **MongoDB Atlas** database as fully established documents, guarded by idempotency checks to prevent duplication.

**Code Provenance Table:**
| Script | Description | Link |
|--------|-------------|------|
| `pipeline.ipynb` | Jupyter notebook executing data acquisition, feature engineering, MongoDB ingestion, and Random Forest modeling. | [GitHub](pipeline.ipynb) |

**Bias Identification:**
The primary biases in this dataset include Survivorship Bias (all five tickers are currently active, highly successful large-cap companies; failed companies are excluded, overstating general market stability) and the severe risk of Look-ahead Bias inherent in any time-series financial modeling.

> ### ⚖️ Analytic Rigor: Bias Mitigation
> To address **Look-ahead Bias**, we strictly enforce a Temporal Train/Test Split (80/20, no shuffling) to ensure the model never trains on future data. Furthermore, the target variable (`next_day_return`) is carefully engineered by shifting the daily returns backward by exactly one row, guaranteeing the model only uses data available at the close of day *t* to predict day *t+1*. **Survivorship Bias** is acknowledged as a scope limitation; results apply only to established large-cap equities.

## Metadata
**ER Diagram:**
![Document Model Architecture](./images/MongoDB_Architecture.png) *(Placeholder - you can add a screenshot of your Atlas cluster or a conceptual document diagram here)*

**Data Table List:**
The resulting structural collections inside the `stock_data` MongoDB database:
| Collection Name | Description |
| :--- | :--- |
| `AAPL` | ~2,515 documents of daily trading data and technical indicators. |
| `MSFT` | ~2,515 documents of daily trading data and technical indicators. |
| `JPM` | ~2,515 documents of daily trading data and technical indicators. |
| `XOM` | ~2,515 documents of daily trading data and technical indicators. |
| `UNH` | ~2,515 documents of daily trading data and technical indicators. |

**Data Dictionary:**
| Name | Data Type | Description | Example | Missingness | Uncertainty (Numerical Features) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Date` | Datetime | The trading date (market close). | `2015-02-18` | Complete (expected 0%). | N/A (non-numerical). |
| `Ticker` | String | The stock symbol. | `AAPL` | Complete (expected 0%). | N/A (non-numerical). |
| `Close` | Double | The closing price in USD. | `32.06` | Complete (expected 0%). | Historical prices are retroactively adjusted for stock splits/dividends; represents an adjusted calculation, not literal dollars exchanged that day. |
| `daily_return` | Double | Percentage change from previous day's close. | `1.24` | Complete (expected 0%). | Micro-variances due to float precision rounding in Python/pandas calculations. |
| `SMA_20` | Double | 20-day Simple Moving Average of closing price. | `30.15` | Null for first 19 days. | Micro-variances due to float precision rounding. |
| `SMA_50` | Double | 50-day Simple Moving Average of closing price. | `28.90` | Null for first 49 days. | Micro-variances due to float precision rounding. |
| `RSI_14` | Double | 14-day Relative Strength Index. | `65.4` | Null for first 14 days. | Micro-variances due to float precision rounding. |
| `MACD` | Double | Difference between 12-day and 26-day EMA. | `0.45` | Null for first 26 days. | Micro-variances due to float precision rounding. |
| `MACD_signal` | Double | 9-day EMA of the MACD line. | `0.41` | Null for first 34 days. | Micro-variances due to float precision rounding. |
| `volume_change` | Double | Percentage change in trading volume from previous day. | `-5.2` | Complete (expected 0%). | Trading volume is occasionally revised post-close by financial exchanges. |
| `next_day_return` | Double | **TARGET:** The `daily_return` of the following trading day. | `-0.85` | Null for the final day. | Same uncertainty profile as `daily_return`. |