"""A GPlearn-based ablation/test module integrated with the project's core components."""

import logging
from typing import Any, Dict, List, Optional, Tuple
import vectorbt as vbt
import numpy as np
import pandas as pd
from gplearn.fitness import _Fitness
from gplearn.genetic import SymbolicRegressor

from BacktestFolder.backtest import VectorBacktest
from StrategyTree.TreeUtils import dataset_preprocess
from hyperparam import config

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s", force=True)
logger = logging.getLogger(__name__)

# --- Config helpers ---
DATA_PATH = config['basicfeed']['filepath']
START_COL = config['execution']['start_col']
FORECAST_HORIZON = config['ablation']['mlp']['horizon']
NUM_IND = config['indicators']['num_indicators']
SIGNAL_THRESHOLD = config['backtest']['signal_threshold']
TRAIN_LEN = config['execution']['data_window']['fixed_train_length']
TEST_LEN = config['execution']['data_window']['fixed_test_length']
STEP_DAYS = config['execution']['data_window']['sliding_window_days']


def add_forward_return(
    df: pd.DataFrame,
    horizon: int = 1,
    price_col: str = "Close",
    out_col: str = "fwd_return",
) -> pd.DataFrame:
    """Add a forward return column for supervised training.

    The target at index t is the percent return from t -> t+1.
    """

    df = df.copy()
    df[out_col] = df[price_col].pct_change().shift(-horizon)
    return df
def load_dataset(path: Optional[str] = None, start_ind: int = 0,end_ind: int = TRAIN_LEN+TEST_LEN) -> pd.DataFrame:
    """Load dataset and add forward return column."""
    path = path or DATA_PATH
    df = pd.read_csv(path)
    df = df.iloc[start_ind:end_ind].reset_index(drop=True)
    df = add_forward_return(df, horizon=FORECAST_HORIZON)
    df = df.dropna().reset_index(drop=True)
    return df

def pick_indicator_columns(
    df: pd.DataFrame,
    indicator_prefix: Optional[str] = None,
    start_col: Optional[int] = None,
    num_ind: int = 80,
) -> List[str]:
    """Pick indicator columns by prefix or by position."""

    if indicator_prefix:
        cols = [c for c in df.columns if c.startswith(indicator_prefix)]
        if len(cols) >= num_ind:
            return cols[:num_ind]
        if cols:
            return cols

    if start_col is not None:
        return list(df.columns[start_col:(start_col + num_ind)])

    exclude = {'Date', 'Close', 'High', 'Low', 'Open', 'Volume'}
    cols = [c for c in df.columns if c not in exclude]
    return cols[:num_ind]

# Main GPlearn comparison class
class CompGPlearn:
  def __init__(self,df:pd.DataFrame,st_day:int,train_len:int,test_len:int,sig_th:float,is_train:bool=True):
    self.df=df.copy()
    self.train_df=self.df.iloc[:train_len].copy().reset_index(drop=True)
    self.test_df=self.df.iloc[train_len:].copy().reset_index(drop=True)
    self.train_dataset=self.train_df[['Close']].reset_index(drop=True)
    self.test_dataset=self.test_df[['Close']].reset_index(drop=True)
    self.indicator_columns=pick_indicator_columns(self.df,start_col=START_COL,num_ind=NUM_IND)
    self.alpha_signals=self.df[self.indicator_columns].values

    #Training and test split
    self.X_train=np.nan_to_num(self.alpha_signals.T)[:train_len]
    self.X_test=np.nan_to_num(self.alpha_signals.T)[train_len:]
    self.y_train=np.nan_to_num(pd.DataFrame(self.df['fwd_return']).iloc[st_day:st_day+train_len].values)
    self.signal_threshold=sig_th
    # Initialize custom fitness function for GPlearn
    self.custom_fitness = _Fitness(function=self.custom_fitness_,greater_is_better=True)
    self.function_set = ['add', 'sub', 'mul','abs', 'max', 'min']
    self.population_size = config['ablation']['gplearn']['population_size']
    self.generations = config['ablation']['gplearn']['generations']
    self.random_state = config['basicfeed']['SEED']

    # Initialize the SymbolicRegressor with the custom fitness function
    self.est_gp=SymbolicRegressor(
        population_size=self.population_size,
        generations=self.generations,
        init_depth=(config['ablation']['gplearn']['init_depth'][0], config['ablation']['gplearn']['init_depth'][1]),
        tournament_size=config['ablation']['gplearn']['tournament_size'],
        stopping_criteria=1.,
        p_crossover=config['ablation']['gplearn']['p_crossover'],
        p_subtree_mutation=config['ablation']['gplearn']['p_subtree_mutation'],
        p_hoist_mutation=config['ablation']['gplearn']['p_hoist_mutation'],
        p_point_mutation=config['ablation']['gplearn']['p_point_mutation'],
        p_point_replace=config['ablation']['gplearn']['p_point_replace'],
        max_samples=config['ablation']['gplearn']['max_samples'],
        verbose=1,
        parsimony_coefficient=config['ablation']['gplearn']['parsimony_coefficient'],
        random_state=self.random_state,
        function_set=self.function_set,
        metric=self.custom_fitness,
        const_range=None,
        n_jobs=1)
    
    if is_train:
      self.train_gplearn()
      self.final_generation,self.test_scores=self.test_gplearn()
      self.test_scores.sort(key=lambda x: x[0], reverse=True)
      self.test_sharpe=self.finalised_result()

  def custom_fitness_(self,y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
    """
    Custom fitness function to evaluate chromosomes using the VectorBacktest.
    y_true, sample_weight are ignored since we're only using y_pred (chromosome signals).
    """
    try:
        #Ensure signals (chromosomes) are 1D
        if y_pred.ndim > 1:
            y_pred = y_pred.ravel()
        discrete_signals = y_pred
        #Check if discrete_signals is empty
        if discrete_signals.size == 0:
             return 0.0
        backtest = VectorBacktest(self.train_dataset, discrete_signals)
        sharpe_ratio = backtest.fitness("sharpe")
        if (np.isnan(sharpe_ratio)or np.isinf(sharpe_ratio)):
            return 0.0
        return sharpe_ratio
    except Exception as e:
        print(f"Error in fitness evaluation: {e}")
        return -1e6
  def train_gplearn(self):
    self.est_gp.fit(self.X_train, self.y_train.ravel())

  def test_gplearn(self):
    final_generation = self.est_gp._programs[-1]
    test_scores = []

    for prog in final_generation:
      preds = prog.execute(self.X_test)
      if preds.size == 0:
          new_signals = np.zeros(len(self.test_dataset))
      else:
          new_signals = np.where(preds >self.signal_threshold, 1,
                            np.where(preds < -self.signal_threshold, -1, 0)).astype(int)

      # unique_values, counts = np.unique(new_signals, return_counts=True)

      new_sharpe = VectorBacktest(self.test_dataset, new_signals).fitness("sharpe")
      if np.isfinite(new_sharpe):
        test_scores.append((round(float(new_sharpe),3), prog))
    return final_generation,test_scores

  def perform_oos(self,fin_gen,oos_test_df: pd.DataFrame):
    test_alpha_signals= oos_test_df[self.indicator_columns].values
    test_signals = pd.DataFrame(test_alpha_signals.T)
    X_test = np.nan_to_num(test_signals)
    test_dataset = oos_test_df[['Close']]
    test_scores = []
    gpportfolio_arr=[]
    for prog in fin_gen:
      preds = prog.execute(X_test)
      if preds.size == 0:
         new_signals = np.zeros(len(test_dataset))
      else:
        new_signals = np.where(preds >self.signal_threshold, 1,
                          np.where(preds < -self.signal_threshold, -1, 0)).astype(int)
        fin_obj=VectorBacktest(test_dataset, new_signals)

        fin_sharpe=fin_obj.fitness("sharpe")
        fin_multi=(fin_obj.fitness("max_drawdown"),fin_obj.port_ret().total_return())
        if np.isfinite(fin_sharpe):
          test_score=(round(float(fin_sharpe),3), fin_multi[0], fin_multi[1])
          test_scores.append(test_score)
          gpportfolio_arr.append(fin_obj.portfolio)

    return test_scores,gpportfolio_arr


  def finalised_result(self):
    avg_res=0.0
    top_n = config['ablation']['gplearn']['top_n']
    top_test_strategies_ = self.test_scores[:top_n]
    print(f"Top {top_n} strategies in test set:")
    for s, p in top_test_strategies_:
      avg_res+=s
    print(f"Sharpe {(avg_res/top_n):.4f}")
    return avg_res/top_n

if __name__ == "__main__":
    compdf=load_dataset(path=DATA_PATH)
    final_gen=None
    for sft_win in range(21):
        start_day,end_day=STEP_DAYS*sft_win,STEP_DAYS*sft_win+TRAIN_LEN
        compdf=load_dataset(path=DATA_PATH,start_ind=start_day,end_ind=end_day)
        gplearn_obj=CompGPlearn(df = compdf,st_day=start_day,train_len=TRAIN_LEN,test_len=TEST_LEN,signal_threshold=SIGNAL_THRESHOLD)
        if(sft_win==20):
            final_gen=gplearn_obj.final_generation
    logging.info(f"{sft_win} sliding window completed.")