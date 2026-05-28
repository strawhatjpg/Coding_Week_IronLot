# Autonomous Bidding Agent & Car Auction Price Predictor

## Overview
This project contains a Machine Learning pipeline and an autonomous bidding agent designed to predict used car wholesale prices and bid against rival agents in a live simulation using a fixed $500,000 budget.

## 1. Diagnostic Exploratory Data Analysis (EDA)
Before altering the data, an in-depth EDA was conducted to justify the downstream pipeline. Key findings include:
* **Outlier Boundaries:** Vehicles priced below $500 and above **$120,000** were identified as data entry errors or extreme luxury outliers, justifying their removal.
* **Non-Linear Condition Drops:** The drop in a car's value is non-linear relative to its `condition` score. A drop from 2.0 to 1.5 destroys significantly more value than a drop from 4.0 to 3.5.
* **Interaction Premiums:** The value of a manual transmission depends entirely on the body type, justifying a combined `body_transmission` feature.
* **100k Odometer Threshold:** Identified a severe, non-linear collapse in `sellingprice` exactly as vehicles cross the 100,000-mile mark, establishing a critical psychological pricing threshold.
* **Wear-and-Tear:** Calculating `usage_intensity` (miles per year) provides a clearer signal of aggressive wear than raw odometer readings alone.

## 2. Data Cleaning & Feature Engineering
* **Strict Partitioning:** The Train/Validation split occurs *before* any imputation statistics are calculated to strictly prevent target leakage.
* **Imputation:** Missing values for `odometer` and `condition` are imputed using medians and lookups calculated exclusively on the training set. A safe fallback is also included for missing `year` data.
* **Feature Engineering:**
  * **`car_age`:** Calculated using a dynamic `max_year` derived from the training data to prevent crashes on unseen future test data.
  * **`usage_intensity`:** Created by dividing `odometer` by `car_age` to measure aggressive wear over time.
  * **`log_odometer`:** Applied a `log1p` transform to the raw odometer reading to tighten its heavily right-skewed distribution.
  * **`condition_bucket`:** Grouped the raw continuous condition scores into distinct ordinal tiers to capture the non-linear value drops identified in the EDA.
  * **Interactions:** Created concatenated categorical features like `make_model` and `body_transmission` to capture nuanced segment premiums.
* **Encoding:** Out-of-fold (OOF) target encoding is applied to high-cardinality features (like `trim`), while low-cardinality strings are left as standard categories for native LightGBM splitting.

## 3. Model Training
* **Algorithm:** LightGBM Regressor.
* **Objective (Standard MSE):** The model uses **Standard Mean Squared Error (MSE)** to act as an objective "oracle" predicting the *true average market value* of the vehicle. All financial risk management is delegated to the downstream Python agent to keep the ML architecture mathematically pure.
* **Export:** The model and all OOF lookups/encoders are exported as pure `.pkl` artifacts for fast, O(1) live inference without Pandas overhead.

## 4. The Bidding Strategy (Agent Script)
The `LiveAuctionAgent` (`agent.py`) calculates a deterministic bid sequence using the **Zeno's Half-Step Strategy** paired with strict budget management:

* **The Profit Margin Floor:** Because the model predicts the average price, the agent demands a strict 10% discount on the predicted value to protect against the Winner's Curse and guarantee profit.
* **The Exposure Cap:** The agent refuses to spend more than 10% of its *current* bankroll on any single car to prevent ruinous losses. 
* **End-Game Starvation Fix:** A hard $25,000 minimum exposure floor is implemented so the agent remains willing to bid on mid-range cars even if its bankroll drops to dangerous levels.
* **The Absolute Ceiling:** The strict walk-away limit is whichever is lower: the Profit Floor or the Exposure Cap.
* **Zeno's Sequence:** To outmaneuver rivals, the agent's bid jumps exactly **50% of the remaining gap** between the current highest bid and its absolute ceiling (enforcing a $50 minimum increment). This guarantees a massive, intimidating first bid that mathematically slows down and safely converges on the limit without overshooting.
