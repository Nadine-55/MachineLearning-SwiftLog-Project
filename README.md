# SwiftLog Logistics Analytics and AI Transformation


## What this project does

SwiftLog is a logistics company with 9 linked tables (customers, products, warehouses, orders, order items, shipments, inventory, delivery performance). This project prepares the data, explores it, builds and tests four prediction models, explains them, wraps them in a prototype app, and turns the findings into recommendations.

| Problem | Type | Test result (tuned) | Baseline | Shuffle-test p |
|---|---|---|---|---|
| Delivery delay | Classification | AUC 0.492, F1 0.791 | AUC 0.500, F1 0.792 | 0.64 |
| Demand forecasting | Regression | R2 -0.064, MAE 11.34 | R2 -0.064, MAE 11.32 | 0.36 |
| Customer value | Regression | R2 -0.023, MAE 4,671.7 | R2 -0.001, MAE 4,662.6 | 0.48 |
| Warehouse performance | Regression | R2 -0.077, MAE 0.061 | R2 -0.014, MAE 0.058 | 0.40 |

A p-value below 0.05 would indicate real signal. All four are well above it.

## What the data shows 

- 63.7% of 5,000 shipments were delayed by 3 or more days (mean 3.6 days), similar across all warehouses and cities.
- 32.5% of orders are Cancelled and 34.3% are Pending; 87.8% of Pending orders are older than 30 days.
- All 1,626 cancelled orders still have a shipment.
- 75.1% of shipments are cross-city; Ibadan has no warehouse.
- Stock cover is about 150 months overall, yet 142 of 2,455 warehouse-product pairs hold under 50 units.

## Scope decisions

- **Delay prediction replaces supplier reliability**, because no table links suppliers to shipments or products.
- **Product returns were not modelled**, because the Returns table was not provided.
- Leakage was checked: Delay Status is derived from DelayDays, and current-month warehouse delay (r = 0.85 with its target) was excluded.
- Time-based splits are used throughout; customer value uses Jan to May behaviour to predict Jun to Aug spend.

## Methods

- Pipelines with StandardScaler, OneHotEncoder and SelectKBest; 3 algorithms per problem (Logistic or Ridge, Random Forest, Gradient Boosting).
- RandomizedSearchCV and GridSearchCV with time-aware cross-validation.
- Baselines (majority class, mean), permutation importance, SHAP, error analysis and a 1,000-shuffle significance test.
- Prototype Streamlit app with a reliability banner on each tab.

## Repository structure

```
Week5_MachineLearning_SwiftLog_Nadine/
  Data/
    Original_Datasets/
    Processed_Datasets/
    Feature_Engineered_Dataset.csv
  Notebook/
    Week5_MachineLearning.ipynb   (master summary)
    01_Data_Prep.ipynb ... 06_Interpretation.ipynb
  Models/
    Best_Model.pkl
    Model_Comparison.xlsx
    Feature_Importance.png
  App/
    app.py
    requirements.txt
    assets/
  Reports/
    Technical_Report.pdf
    Executive_Summary.pdf
  Presentation/
    Executive_Presentation.pptx
  References/
  README.md
```

## How to run

```
pip install -r App/requirements.txt
streamlit run App/app.py
```

Run the command from the repository root. The notebooks are numbered 01 to 06 and are meant to run in order.

## Recommendations (summary)

12 recommendations are in the technical report. The main ones: track on-time delivery as a headline KPI; record carrier, route and timestamp data; clear the Pending backlog; stop creating shipments for cancelled orders; right-size and rebalance inventory; link suppliers to products; and do not let the current models drive decisions until better data lifts them above the baselines. The roadmap has three phases over 18 months.

## Limitations

- Small samples (105 warehouse-months, 1,000 customers) and few informative features.
- Findings are descriptive, not causal.
- Delay labels may reflect how the data was recorded rather than real-world causes.
