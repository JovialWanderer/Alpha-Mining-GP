from arch import arch_model
from hyperparam import *

class SelectVolatilityModel:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy().reset_index(drop=True)
        # Calculate historical volatility 
        self.historical_volatility = self.calc_true_volatility(
            yz_window=config['volatility']['yz_window_length']
        )
        self.conditional_rmse=self.volatility_selector()
    def calc_true_volatility(self, yz_window: int = 30) -> pd.Series:
        """
        Calculates the Yang-Zhang historical volatility using pandas rolling operations.
        Returns a pandas Series aligned with the original DataFrame's index.
        """
        # --- 1. Data Preparation and Alignment---
        temp_df = pd.DataFrame(index=self.df.index)
        temp_df['o'] = pd.to_numeric(self.df['Open'], errors='coerce')
        temp_df['h'] = pd.to_numeric(self.df['High'], errors='coerce')
        temp_df['l'] = pd.to_numeric(self.df['Low'], errors='coerce')
        temp_df['c'] = pd.to_numeric(self.df['Close'], errors='coerce')
        
        # Log returns using the previous day's close
        prev_c = temp_df['c'].shift(1)
        temp_df['r_oc'] = np.log(temp_df['o'] / prev_c)  # Open-to-Previous-Close
        temp_df['r_cc'] = np.log(temp_df['c'] / prev_c)  # Close-to-Close
        
        # Rogers-Satchell term
        temp_df['r_s'] = (np.log(temp_df['h'] / temp_df['c']) * np.log(temp_df['h'] / temp_df['o']) +
                        np.log(temp_df['l'] / temp_df['c']) * np.log(temp_df['l'] / temp_df['o']))

        temp_df = temp_df.dropna()

        k = 0.34 / (1.34 + ((yz_window + 1) / (yz_window - 1)))
        
        sigma_o_sq = temp_df['r_oc'].rolling(window=yz_window).var()
        sigma_c_sq = temp_df['r_cc'].rolling(window=yz_window).var()
        sigma_rs = temp_df['r_s'].rolling(window=yz_window).mean()
        
        # Calculate variance and square root for volatility
        yz_var = (k * sigma_o_sq) + ((1 - k) * sigma_c_sq) + sigma_rs
        
        # Rogers-Satchell negative variance values to a small positive number before taking root.
        yz_var[yz_var < 0] = 1e-8
        yz_vol = np.sqrt(yz_var)
        return yz_vol

    def volatility_selector(self, method: str = 'bootstrap') -> float:
        """
        Selects the best GARCH model based on RMSE against historical volatility.
        """
        # --- Data Preparation ---
        returns = self.df['Close'].pct_change() * 100
        aligned_data = pd.concat([returns, self.historical_volatility], axis=1)
        aligned_data.columns = ['returns', 'historical_vol']
        aligned_data = aligned_data.dropna()

        historical_vol = aligned_data['historical_vol']
        returns = aligned_data['returns']
        
        # --- Train/Test Split ---
        train_end_idx = int(0.85 * len(aligned_data))
        train_returns = returns.iloc[:train_end_idx]
        test_returns = returns.iloc[train_end_idx:]
        train_historical = historical_vol.iloc[:train_end_idx]
        test_historical = historical_vol.iloc[train_end_idx:]

        # --- Conditional Volatility Model Fitting ---
        vol_params = config['volatility']['params']
        # Default 'o' and 'power' to 0 and 2 if not specified
        o_val = vol_params.get('o', 0)
        power_val = vol_params.get('power', 2.0)
        dist_val = vol_params.get('dist', 'normal')

        am = arch_model(train_returns, p=vol_params['p'], q=vol_params['q'], o=o_val, power=power_val, dist=dist_val)
        res = am.fit(disp='off')

        # --- Forecasting and Comparison ---
        forecasts = res.forecast(horizon=len(test_returns), method=method)
        
        predicted_variance = forecasts.variance.values[-1, :]
        predicted_vol = np.sqrt(predicted_variance)

        # --- Scaling and RMSE ---
        f1 = np.ptp(res.conditional_volatility) / np.ptp(train_historical) if np.ptp(train_historical) > 0 else 1
        f2 = np.ptp(predicted_vol) / np.ptp(test_historical) if np.ptp(test_historical) > 0 else 1
        
        scaled_train_vol = res.conditional_volatility / f1
        scaled_test_vol = predicted_vol / f2
        
        rmse = np.sqrt(np.mean((test_historical.values - scaled_test_vol) ** 2))

        return rmse