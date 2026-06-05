Steps:

1. Import data, do EDA, analyze patterns (do this part carefully, slowly, cleanly!)

2. Try models:
- Baseline model (mean)
- Linear Regression
- Ridge
- Lasso
- RandomTree (with and without polynomial) + feature importance
- RandomForest (with and without polynomial) + feature importance
- Hard threshold + 2 regressors idea 

```Criteria:```

* Do it cleanly: cross-validation, nested cross-validation, hyperparameter tuning using good methods (e.g. GridSearch), properly analyze and discuss any intresting pattern or occurance, write functions for repetitive tasks, residual analysis

* Grading goals:
| 15 | All code runs correctly from start to finish. Notebook is cleanly organized into logical sections. Variable names are clear, repeated logic is factored when appropriate, and results are reproducible (for example via fixed random seeds). |
| 10 | Data loading, inspection, and exploratory analysis are thorough. Relevant statistics and visualizations are included. Important patterns, anomalies, imbalance, skewness, outliers, or correlations are identified and interpreted. |
| 15 | A clear baseline is established. Multiple reasonable models are tested. Model comparison is rigorous, with appropriate validation and metrics. Hyperparameter tuning, residual analysis, error analysis, or confusion-matrix analysis are used where relevant. Choices are justified. |
| 10 | The notebook reads like a clear technical report. Markdown sections, explanations, comments, and conclusions guide the reader through the full workflow. Decisions are motivated, limitations are discussed, and conclusions are tied back to the project question. |

3. Properly compare the models in a nice visualization and conclude about our final choice

4. If we have time, try even more advanced models and report results

The code we can do individually and improve at the end to be consistent / fix small errors. The analysis or tricky part 

Memo: don't forget to add import warnings
warnings.filterwarnings("ignore") after you are done and have checked causes of warnings

Ask Lucas:
- is it okay to do the EDA on the whole train set before CV?

we wanna do EDA, but also nested cross-validation Heatmap, distribution of each variable, pairplots of features against features or target against target

- Should we maybe try PCA too?
- how do we explain `size_mismatch_reports` has little correlation? (we expected some)
- we did .describe(), make sure to actually explain ranges, minimums, etc. if relevant OR just remove it if visuals are good enough

* focus on the bimodal shape of the return_rate_pct against features and its own distribution
* how to integrate polynomial/unimodal relationshiop between price_eur vs return_rate_pct
* linear correlation between image and description quality -> how to take care of multicolinearity? 
* notice the gap in description quality at the lower end of the range (+ weird skewed distribution)
* consider given correlation coefficients if some multicolinearity between less obvious features might affect our model
* do not forget about discrete and categorical variables, try to explain for example why holiday has a asymmetric distribution, check why size_variants_code looks bimodal, etc. (do not forget discrete stuff!)
* check for outliers if any!
* polynomial degree debate or another transformation
* how to combine strength of random forest regressor with polynomial transformation
* don't forget random seed bs!