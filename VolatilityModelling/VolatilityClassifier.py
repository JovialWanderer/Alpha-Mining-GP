from arch import arch_model
from hyperparam import *

def volatility_classifier(dataset: pd.DataFrame):
    """
    Fits a GARCH model and classifies the original data into contiguous high and low volatility periods.
    """
    # --- 1. GARCH Model Fitting ---
    df = dataset.copy()
    
    # Calculate absolute percent returns for GARCH modeling
    returns = np.abs(df['Close'].pct_change().dropna() * 100)

    # Fit the GARCH model
    vol_params = config['volatility']['params']
    model = arch_model(
        returns,
        p=vol_params['p'],
        q=vol_params['q'],
        o=vol_params.get('o', 0),
        power=vol_params.get('power', 2.0),
        dist=vol_params.get('dist', 'normal')
    )
    garch_result = model.fit(disp="off")
    predict_vol = garch_result.conditional_volatility

    # --- 2. Classify Volatility into Blocks ---
    window = config['execution']['vanilla_window']
    block_means = [
        np.mean(predict_vol[i : i + window])
        for i in range(0, len(predict_vol), window)
    ]
    
    # Classify each block as high (1) or low (0) vol
    vol_threshold = np.mean(block_means)
    classified_blocks = [1 if mean > vol_threshold else 0 for mean in block_means]

    # --- 3. Identify Contiguous Regime Boundaries (Refactored Logic) ---
    regimes = []
    if not classified_blocks:
        # Handle case with no data
        return garch_result, predict_vol, [], [], [], [], [], []

    # Start with the first regime
    current_regime_type = classified_blocks[0]
    regime_start_index = 0

    for i in range(1, len(classified_blocks)):
        if classified_blocks[i] != current_regime_type:
            # Regime has changed, finalize the previous one
            regime_end_index = i * window
            regimes.append((regime_start_index, regime_end_index, current_regime_type))
            
            # Start a new regime
            current_regime_type = classified_blocks[i]
            regime_start_index = i * window

    final_end_index = len(predict_vol)
    regimes.append((regime_start_index, final_end_index, current_regime_type))
    
    # --- 4. Create Datasets from Identified Boundaries ---
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