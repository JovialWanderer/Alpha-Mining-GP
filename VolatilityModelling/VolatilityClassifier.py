from arch import arch_model
from arch.univariate.base import ARCHModelResult
from hyperparam import *

def volatility_classifier(dataset: pd.DataFrame):
    """
    Fits a GARCH model and classifies the original data into contiguous high and low volatility periods.
    """
    #GARCH Model Fitting 
    df = dataset.copy()
    
    # Calculate absolute percent returns for GARCH modeling- also scale by 100 to increase numerical stability
    returns = np.abs(df['Close'].pct_change().dropna() * 100)

    # Fit the GARCH model
    vol_params = config['volatility']['params']# Extract GARCH parameters from config
    model = arch_model(
        returns,
        p=vol_params['p'],
        q=vol_params['q'],
        o=vol_params.get('o', 0),
        power=vol_params.get('power', 2.0),
        dist=vol_params.get('dist', 'StudentsT'),
    )
    garch_result = model.fit(disp="off")
    predict_vol = garch_result.conditional_volatility

    # Classify Volatility into Blocks 
    window = config['execution']['warmstart']['vanilla_window']
    block_means = [
        np.mean(predict_vol[i : i + window])
        for i in range(0, len(predict_vol), window)
    ]
    
    # Classify each block as high (1) or low (0) vol
    vol_threshold = np.mean(block_means)
    classified_blocks = [1 if mean > vol_threshold else 0 for mean in block_means]

    #  3. Identify Contiguous Regime Boundaries (Refactored Logic) 
    regimes = []
    if not classified_blocks:
        # Handle case with no data
        return garch_result, predict_vol, [], [], [], [], [], []

    # Start with the first regime
    current_regime_type = classified_blocks[0]
    regime_start_index = 0

    for i in range(1, len(classified_blocks)):
        if classified_blocks[i] != current_regime_type:
            # Regime has changed, finalize the previous one and start a new one
            regime_end_index = i * window
            regimes.append((regime_start_index, regime_end_index, current_regime_type))
            current_regime_type = classified_blocks[i]
            regime_start_index = i * window

    final_end_index = len(predict_vol)
    regimes.append((regime_start_index, final_end_index, current_regime_type))
    
    # Create Datasets from Identified Boundaries 
    aligned_df = df.iloc[1:].reset_index(drop=True)

    high_vol_datasets, low_vol_datasets = [], []
    start_high_ind, end_high_ind, start_low_ind, end_low_ind = [], [], [], []

    for start_idx, end_idx, regime_type in regimes:
        end_idx = min(end_idx, len(aligned_df))
        
        if regime_type == 1:  # High Volatility
            start_high_ind.append(start_idx)
            end_high_ind.append(end_idx)
            high_vol_datasets.append(aligned_df.iloc[start_idx:end_idx])
        else:  # Low Volatility
            start_low_ind.append(start_idx)
            end_low_ind.append(end_idx)
            low_vol_datasets.append(aligned_df.iloc[start_idx:end_idx])

    return (
        garch_result,
        predict_vol,
        high_vol_datasets,
        low_vol_datasets,
        start_high_ind,
        end_high_ind,
        start_low_ind,
        end_low_ind,
    )

def perform_rolling_garch_forecast(
    dataset: pd.DataFrame,
    garch_result: ARCHModelResult,
    dataset_test: pd.DataFrame
) -> np.ndarray:
    """
    Forecast volatility for the test dataset using a rolling GARCH model fitted on the training dataset.
    """
    train_returns = dataset['Close'].pct_change().dropna() * 100
    test_returns = dataset_test['Close'].pct_change().dropna() * 100
    vol_params = config['volatility']['params']
    garch_params = {
        'p': vol_params['p'],
        'q': vol_params['q'],
        'o': vol_params.get('o', 0),
        'power': vol_params.get('power', 2.0),
        'dist': vol_params.get('dist', 'StudentsT')
    }

    step_size = config['execution']['data_window']['forecast_horizon']
    all_forecasts = []

    for i in range(0, len(test_returns), step_size):

        current_train = pd.concat([train_returns, test_returns.iloc[:i]])

        model = arch_model(current_train, **garch_params)
        res = model.fit(disp='off', show_warning=False)

        horizon = min(step_size, len(test_returns) - i)
        if horizon <= 0:
            break

        forecast = res.forecast(horizon=horizon, reindex=False)
        variance_chunk = forecast.variance.values[-1]
        all_forecasts.append(np.sqrt(variance_chunk))

    if not all_forecasts:
        return np.array([])

    pred_volatility = np.concatenate(all_forecasts)

    return pred_volatility[:len(test_returns)]