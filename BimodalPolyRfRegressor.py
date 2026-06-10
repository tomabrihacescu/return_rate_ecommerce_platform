import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin #Have to have this for the CV apparently
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


class BimodalPolyRfRegressor(BaseEstimator, RegressorMixin): #BaseEstimator & RegressorMixin for the CV's 
    def __init__(self, threshold=173.96, threshold_feature='price_eur', random_state=42):
        self.threshold = threshold
        self.threshold_feature = threshold_feature
        self.random_state = random_state
        
        self.grid_low_ = None
        self.grid_high_ = None

    def fit(self, X, y):
        #Models
        model_poly = Pipeline([
            ('poly', PolynomialFeatures()), 
            ('scaler', StandardScaler()),
            ('linear', LinearRegression())
        ])
        poly_params = {
            'poly__degree': list(range(0, 6))
        }

        model_rf = RandomForestRegressor(random_state=self.randomx_state)
        rf_params = {
            "n_estimators": [50, 100], 
            "max_depth": [None, 5, 10]
        }

        #Grid
        self.grid_low_ = GridSearchCV(model_poly, poly_params, cv=3)
        self.grid_high_ = GridSearchCV(model_rf, rf_params, cv=3)

        #Threshold
        low_mask = X[self.threshold_feature] <= self.threshold


        #Fitting
        if low_mask.any():
            self.grid_low_.fit(X[low_mask], y[low_mask])
            
        if (~low_mask).any():
            self.grid_high_.fit(X[~low_mask], y[~low_mask])

        return self 

    def predict(self, X):
        predictions = np.zeros(len(X))
        
        #Threshold
        low_mask = X[self.threshold_feature] <= self.threshold

        # Predict
        if low_mask.any():
            predictions[low_mask] = self.grid_low_.predict(X[low_mask])
            
        if (~low_mask).any():
            predictions[~low_mask] = self.grid_high_.predict(X[~low_mask])

        return predictions