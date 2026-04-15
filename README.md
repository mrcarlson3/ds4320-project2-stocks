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
| Source Title | Relevance | Source URL/Reference |
|--------------|-----------|----------------------|
| Efficient Capital Markets: A Review of Theory and Empirical Work (Fama, 1970) | Outlines the Efficient Market Hypothesis (EMH), arguing that stock prices fully reflect all available information. | [Local Copy in OneDrive](#) |
| Understanding the Relative Strength Index (RSI) | Guide detailing how RSI is calculated and traditionally interpreted to identify overbought/oversold conditions. | [Local Copy in OneDrive](#) |
| Moving Average Convergence Divergence (MACD) | Guide explaining the mechanics of MACD and its signal line for momentum tracking. | [Local Copy in OneDrive](#) |
| Time Series Split in Scikit-Learn | Scikit-Learn documentation explaining the strict necessity of temporal splitting for time-series data to avoid data leakage. | [Local Copy in OneDrive](#) |
| MongoDB Document Model Design | MongoDB documentation detailing best practices for structuring time-series data in BSON format. | [Local Copy in OneDrive](#) |

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