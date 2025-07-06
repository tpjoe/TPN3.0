#!/usr/bin/env python
import pickle
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import pearsonr
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from typing import Dict, Tuple, List, Optional


class DataLoader:
    """Handles data loading and preprocessing"""
    
    def __init__(self, data_path: str, patient_splits_path: str, scaler_path: str):
        self.data_path = data_path
        self.patient_splits_path = patient_splits_path
        self.scaler_path = scaler_path
        
    def load_test_data(self, max_seq_length: Optional[int] = None) -> pd.DataFrame:
        """Load and filter test data with right censoring"""
        df = pd.read_csv(self.data_path, low_memory=False)
        df['max_chole_TPNEHR_date'] = pd.to_datetime(df['max_chole_TPNEHR_date'], errors='coerce')
        df['DateOrdered'] = pd.to_datetime(df['DateOrdered'], errors='coerce')
        
        with open(self.patient_splits_path, 'rb') as f:
            patient_splits = pickle.load(f)
        
        test_patient_ids = patient_splits['test']
        
        # Right censoring
        mask = df['max_chole_TPNEHR_date'].isna() | (df['DateOrdered'] <= df['max_chole_TPNEHR_date'])
        df_censored = df[mask].copy()
        
        # Filter for test patients
        df_censored = df_censored[df_censored['person_id'].isin(test_patient_ids)]
        print(f"Filtered to {len(df_censored)} timesteps from {df_censored['person_id'].nunique()} test patients")
        
        # Sort and apply max sequence length
        df_censored = df_censored.sort_values(['person_id', 'days_on_TPN'])
        if max_seq_length is not None:
            df_censored = df_censored.groupby('person_id').tail(max_seq_length)
            df_censored = df_censored.reset_index(drop=True)
        
        df_censored['timestep'] = df_censored.groupby('person_id').cumcount()
        return df_censored
    
    def scale_data(self, df: pd.DataFrame, column_config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """Scale data using loaded scaler"""
        with open(self.scaler_path, 'rb') as f:
            scaler_info = pickle.load(f)
        
        # Extract data arrays using config columns
        treatments = df[column_config['treatment_columns']].values
        vitals = df[column_config['vital_columns']].values
        outcomes = df[column_config['outcome_columns']].values
        static_features_df = df.groupby('person_id')[column_config['static_columns']].first()
        static_features = static_features_df.values
        
        # Scale treatments
        treatments_std = scaler_info['treatments_std'].values
        # Set to zero where std is zero (no variance means feature is constant)
        treatments_scaled = np.where(treatments_std != 0, 
                                   (treatments - scaler_info['treatments_mean'].values) / treatments_std,
                                   0.0)
        treatments = np.nan_to_num(treatments_scaled, nan=0.0)
        
        # Scale vitals
        vitals_std = scaler_info['vitals_std']
        # Set to zero where std is zero (no variance means feature is constant)
        vitals_scaled = np.where(vitals_std != 0,
                                (vitals - scaler_info['vitals_mean']) / vitals_std,
                                0.0)
        vitals = np.nan_to_num(vitals_scaled, nan=0.0)
        
        # Scale static features
        static_stds = scaler_info['static_stds']
        # Set to zero where std is zero (no variance means feature is constant)
        static_features_scaled = np.where(static_stds != 0,
                                         (static_features - scaler_info['static_means']) / static_stds,
                                         0.0)
        static_features = np.nan_to_num(static_features_scaled, nan=0.0)
        
        # Scale outcomes
        zscore_mean = scaler_info['output_means']['zscore']
        zscore_std = scaler_info['output_stds']['zscore']
        # Set to zero where std is zero (no variance means feature is constant)
        if zscore_std != 0:
            outcomes[:, 0] = (outcomes[:, 0] - zscore_mean) / zscore_std
        else:
            outcomes[:, 0] = 0.0
        
        return treatments, vitals, outcomes, static_features, scaler_info
    
    def prepare_batch(self, df: pd.DataFrame, treatments: np.ndarray, vitals: np.ndarray, 
                      outcomes: np.ndarray, static_features: np.ndarray) -> Dict[str, torch.Tensor]:
        """Prepare padded batch for model input"""
        patient_ids = df['person_id'].unique()
        sequence_lengths = df.groupby('person_id').size().values
        
        max_seq_len = sequence_lengths.max()
        n_patients = len(patient_ids)
        
        # Initialize arrays
        treatments_padded = np.zeros((n_patients, max_seq_len, treatments.shape[1]))
        vitals_padded = np.zeros((n_patients, max_seq_len, vitals.shape[1]))
        outcomes_padded = np.zeros((n_patients, max_seq_len, 2))
        active_entries = np.zeros((n_patients, max_seq_len, 1))
        
        # Fill arrays
        start_idx = 0
        for i, seq_len in enumerate(sequence_lengths):
            treatments_padded[i, :seq_len] = treatments[start_idx:start_idx+seq_len]
            vitals_padded[i, :seq_len] = vitals[start_idx:start_idx+seq_len]
            outcomes_padded[i, :seq_len] = outcomes[start_idx:start_idx+seq_len]
            active_entries[i, :seq_len] = 1
            start_idx += seq_len
        
        # Create previous treatments and outputs
        prev_treatments = np.zeros_like(treatments_padded)
        prev_treatments[:, 1:] = treatments_padded[:, :-1]
        
        prev_outputs = np.zeros_like(outcomes_padded)
        for i in range(n_patients):
            if sequence_lengths[i] > 0:
                prev_outputs[i, 0, :] = outcomes_padded[i, 0, :]
        prev_outputs[:, 1:] = outcomes_padded[:, :-1]
        
        return {
            'prev_treatments': torch.tensor(prev_treatments, dtype=torch.float32),
            'current_treatments': torch.tensor(treatments_padded, dtype=torch.float32),
            'vitals': torch.tensor(vitals_padded, dtype=torch.float32),
            'prev_outputs': torch.tensor(prev_outputs, dtype=torch.float32),
            'outputs': torch.tensor(outcomes_padded, dtype=torch.float32),
            'static_features': torch.tensor(static_features, dtype=torch.float32),
            'active_entries': torch.tensor(active_entries, dtype=torch.float32),
            'sequence_lengths': torch.tensor(sequence_lengths, dtype=torch.long)
        }


class ModelWrapper:
    """Handles model initialization and prediction"""
    
    def __init__(self, model_path: str, config_path: str = "config"):
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.cfg = None
        
    def initialize_model(self):
        """Initialize model from saved state and config"""
        state_dict = torch.load(self.model_path, map_location='cpu')
        
        OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)
        
        with initialize(config_path=self.config_path, version_base=None):
            self.cfg = compose(config_name="config.yaml", 
                               overrides=["+backbone=ct", "+dataset=synthetic_neonatal", 
                                        "+backbone/ct_hparams=synthetic_neonatal"])
            
            dataset_collection = instantiate(self.cfg.dataset, _recursive_=True)
            dataset_collection.process_data_multi()
            
            self.model = instantiate(self.cfg.model.multi, self.cfg, dataset_collection, _recursive_=False)
            self.model.load_state_dict(state_dict)
            self.model.eval()
    
    def get_column_config(self) -> Dict:
        """Extract column configuration from loaded config"""
        return {
            'treatment_columns': list(self.cfg.dataset.treatment_columns),
            'vital_columns': list(self.cfg.dataset.vital_columns),
            'static_columns': list(self.cfg.dataset.static_columns),
            'outcome_columns': list(self.cfg.dataset.outcome_columns)
        }
    
    def predict_teacher_forcing(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Make predictions with teacher forcing"""
        with torch.no_grad():
            _, outcome_pred_dict, _ = self.model(batch)
            outcome_pred = torch.cat([outcome_pred_dict['zscore'], 
                                     outcome_pred_dict['dated_max_chole_TPNEHR']], dim=-1)
        return outcome_pred
    
    def predict_autoregressive(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Make predictions in autoregressive mode"""
        with torch.no_grad():
            n_patients = batch['static_features'].shape[0]
            max_steps = batch['vitals'].shape[1]
            
            outcome_pred = torch.zeros((n_patients, max_steps, 2), dtype=torch.float32)
            
            for t in range(max_steps):
                _, step_outcome_pred_dict, _ = self.model(batch)
                
                step_outcome_pred = torch.cat([step_outcome_pred_dict['zscore'], 
                                              step_outcome_pred_dict['dated_max_chole_TPNEHR']], dim=-1)
                outcome_pred[:, t, :] = step_outcome_pred[:, t, :]
                
                if t < max_steps - 1:
                    zscore_pred = step_outcome_pred[:, t, 0:1]
                    disease_logit = step_outcome_pred[:, t, 1:2]
                    disease_prob = torch.sigmoid(disease_logit)
                    batch['prev_outputs'][:, t + 1, :] = torch.cat([zscore_pred, disease_prob], dim=-1)
            
        return outcome_pred


class MetricsCalculator:
    """Handles evaluation metrics calculation"""
    
    @staticmethod
    def extract_last_timestep_predictions(outcome_pred: torch.Tensor, outcomes_padded: np.ndarray, 
                                         sequence_lengths: np.ndarray) -> Tuple[List, List, List, List]:
        """Extract predictions and true values for last timestep"""
        last_zscore_predictions = []
        last_zscore_true = []
        last_disease_predictions = []
        last_disease_true = []
        
        for i, seq_len in enumerate(sequence_lengths):
            last_zscore_predictions.append(outcome_pred[i, seq_len-1, 0].item())
            last_zscore_true.append(outcomes_padded[i, seq_len-1, 0])
            
            last_disease_predictions.append(outcome_pred[i, seq_len-1, 1].item())
            last_disease_true.append(outcomes_padded[i, seq_len-1, 1])
        
        return last_zscore_predictions, last_zscore_true, last_disease_predictions, last_disease_true
    
    @staticmethod
    def extract_early_predictions(outcome_pred: torch.Tensor, outcomes_padded: np.ndarray, 
                                 sequence_lengths: np.ndarray, days_before: int) -> Tuple[List, List, List, List]:
        """Extract predictions made 'days_before' days early, compared to final true values"""
        zscore_predictions = []
        zscore_true = []
        disease_predictions = []
        disease_true = []
        
        for i, seq_len in enumerate(sequence_lengths):
            # Only include patients with enough timepoints
            if seq_len > days_before:
                # Get prediction from earlier timestep
                pred_idx = seq_len - 1 - days_before
                zscore_predictions.append(outcome_pred[i, pred_idx, 0].item())
                disease_predictions.append(outcome_pred[i, pred_idx, 1].item())
                
                # Get true values from last timestep
                zscore_true.append(outcomes_padded[i, seq_len-1, 0])
                disease_true.append(outcomes_padded[i, seq_len-1, 1])
        
        return zscore_predictions, zscore_true, disease_predictions, disease_true
    
    @staticmethod
    def calculate_metrics(zscore_predictions: List, zscore_true: List, disease_predictions: List, 
                         disease_true: List, scaler_info: Dict) -> Dict:
        """Calculate evaluation metrics"""
        # Convert disease logits to probabilities
        disease_probabilities = 1 / (1 + np.exp(-np.array(disease_predictions)))
        
        # Unscale zscore predictions
        zscore_mean = scaler_info['output_means']['zscore']
        zscore_std = scaler_info['output_stds']['zscore']
        zscore_predictions_unscaled = np.array(zscore_predictions) * zscore_std + zscore_mean
        zscore_true_unscaled = np.array(zscore_true) * zscore_std + zscore_mean
        
        # Calculate metrics
        zscore_rmse = np.sqrt(np.mean((zscore_predictions_unscaled - zscore_true_unscaled) ** 2))
        zscore_pearsonr, zscore_pvalue = pearsonr(zscore_predictions_unscaled, zscore_true_unscaled)
        
        disease_auc_roc = roc_auc_score(disease_true, disease_probabilities)
        disease_auc_pr = average_precision_score(disease_true, disease_probabilities)
        
        return {
            'zscore_rmse': zscore_rmse,
            'zscore_pearsonr': zscore_pearsonr,
            'zscore_pvalue': zscore_pvalue,
            'disease_auc_roc': disease_auc_roc,
            'disease_auc_pr': disease_auc_pr,
            'disease_probabilities': disease_probabilities,
            'zscore_predictions_unscaled': zscore_predictions_unscaled,
            'zscore_true_unscaled': zscore_true_unscaled
        }


def main(teacher_forcing: bool = True):
    """Main prediction pipeline"""
    # Initialize components
    data_loader = DataLoader(
        data_path='data/processed/autoregressive.csv',
        patient_splits_path='outputs/patient_splits.pkl',
        scaler_path='outputs/data_scaler.pkl'
    )
    
    model_wrapper = ModelWrapper(model_path='outputs/trained_model.pt')
    model_wrapper.initialize_model()
    
    # Get column configuration from model's config
    column_config = model_wrapper.get_column_config()
    
    # Load and preprocess data
    df_test = data_loader.load_test_data(max_seq_length=model_wrapper.cfg.dataset.max_seq_length)
    treatments, vitals, outcomes, static_features, scaler_info = data_loader.scale_data(df_test, column_config)
    
    # Prepare batch
    batch = data_loader.prepare_batch(df_test, treatments, vitals, outcomes, static_features)
    
    # Make predictions
    if teacher_forcing:
        outcome_pred = model_wrapper.predict_teacher_forcing(batch)
    else:
        outcome_pred = model_wrapper.predict_autoregressive(batch)
    
    # Calculate metrics
    outcomes_padded = batch['outputs'].numpy()
    sequence_lengths = batch['sequence_lengths'].numpy()
    
    # Evaluate at multiple horizons for both modes
    mode_name = "Teacher Forcing" if teacher_forcing else "Autoregressive"
    print(f"\n=== Multi-Horizon Evaluation ({mode_name}) ===")
    print(f"Total patients: {len(sequence_lengths)}")
    
    for days_before in range(0, 6):  # 0 to 5 days before
        if days_before == 0:
            # Last timestep prediction
            zscore_pred, zscore_true, disease_pred, disease_true = MetricsCalculator.extract_last_timestep_predictions(
                outcome_pred, outcomes_padded, sequence_lengths
            )
        else:
            # Early prediction
            zscore_pred, zscore_true, disease_pred, disease_true = MetricsCalculator.extract_early_predictions(
                outcome_pred, outcomes_padded, sequence_lengths, days_before
            )
        
        if len(zscore_pred) > 0:  # Only calculate if we have valid predictions
            metrics = MetricsCalculator.calculate_metrics(
                zscore_pred, zscore_true, disease_pred, disease_true, scaler_info
            )
            
            print(f"\n{days_before} days before end (n={len(zscore_pred)} patients):")
            print(f"  Zscore R: {metrics['zscore_pearsonr']:.4f}")
            print(f"  Disease AUC-ROC: {metrics['disease_auc_roc']:.4f}")
            print(f"  Disease AUC-PR: {metrics['disease_auc_pr']:.4f}")


if __name__ == "__main__":
    # Set to False for autoregressive mode
    TEACHER_FORCING = True
    main(teacher_forcing=TEACHER_FORCING)