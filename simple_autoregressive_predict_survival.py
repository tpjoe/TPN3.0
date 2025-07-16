#!/usr/bin/env python
"""Survival version of simple_autoregressive_predict.py"""
import torch
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import pearsonr
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from typing import Dict, Tuple, List
import yaml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelWrapper:
    """Handles model initialization and prediction"""
    
    def __init__(self, cfg, dataset_collection):
        self.cfg = cfg
        self.dataset_collection = dataset_collection
        self.model = None
        
    def initialize_model(self, model_path: str):
        """Initialize model from saved state"""
        # Create minimal dataset for model initialization
        class MinimalDatasetCollection:
            def __init__(self, cfg, real_dataset):
                self.outcome_columns = real_dataset.outcome_columns
                self.outcome_types = real_dataset.outcome_types if hasattr(real_dataset, 'outcome_types') else ['binary']
                self.has_vitals = real_dataset.has_vitals
                self.autoregressive = True
                self.projection_horizon = cfg.dataset.projection_horizon
        
        minimal_dataset = MinimalDatasetCollection(self.cfg, self.dataset_collection)
        
        # Load model
        self.model = instantiate(self.cfg.model.multi, self.cfg, minimal_dataset, _recursive_=False)
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
    
    def predict_teacher_forcing(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Make predictions with teacher forcing"""
        with torch.no_grad():
            _, outcome_pred_dict, _ = self.model(batch)
            
            # Get predictions for the target disease
            target_disease = self.cfg.dataset.target_disease
            if isinstance(outcome_pred_dict, dict):
                outcome_pred = outcome_pred_dict[target_disease]
            else:
                outcome_pred = outcome_pred_dict
                
        return outcome_pred
    
    def predict_autoregressive(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Make predictions in autoregressive mode"""
        # For simplicity, using teacher forcing for now
        # Full autoregressive would require step-by-step prediction
        return self.predict_teacher_forcing(batch)


class MetricsCalculator:
    """Handles evaluation metrics calculation"""
    
    @staticmethod
    def extract_predictions_at_horizon(outcome_pred: torch.Tensor, test_data: Dict,
                                     days_before: int) -> Tuple[np.ndarray, np.ndarray]:
        """Extract predictions and true values at specified horizon"""
        seq_lengths = test_data['sequence_lengths']
        active_entries = test_data['active_entries']
        true_outcomes = test_data['outputs']
        n_patients = len(seq_lengths)
        
        y_true = []
        y_pred = []
        
        for i in range(n_patients):
            active_mask = active_entries[i, :, 0].astype(bool)
            
            if active_mask.any():
                active_indices = np.where(active_mask)[0]
                last_active_idx = active_indices[-1]
                
                if days_before == 0:
                    # Use last active timestep
                    y_true.append(true_outcomes[i, last_active_idx, 0])
                    y_pred.append(outcome_pred[i, last_active_idx, 0])
                else:
                    # Use prediction from earlier timestep
                    early_idx = last_active_idx - days_before
                    if early_idx >= 0 and active_entries[i, early_idx, 0] > 0:
                        # Compare early prediction to final outcome
                        y_true.append(true_outcomes[i, last_active_idx, 0])
                        y_pred.append(outcome_pred[i, early_idx, 0])
        
        return np.array(y_true), np.array(y_pred)
    
    @staticmethod
    def calculate_binary_metrics(y_true: np.ndarray, y_pred_logits: np.ndarray) -> Dict:
        """Calculate binary classification metrics"""
        metrics = {}
        
        if len(y_true) == 0:
            return metrics
            
        # Convert logits to probabilities
        y_probs = 1 / (1 + np.exp(-y_pred_logits))
        
        # Count cases and controls
        n_cases = np.sum(y_true == 1)
        n_controls = np.sum(y_true == 0)
        metrics['n_patients'] = len(y_true)
        metrics['n_cases'] = n_cases
        metrics['n_controls'] = n_controls
        metrics['prevalence'] = n_cases / len(y_true) if len(y_true) > 0 else 0
        
        # Calculate AUC metrics if both classes present
        if len(np.unique(y_true)) > 1:
            metrics['auc_roc'] = roc_auc_score(y_true, y_probs)
            metrics['auc_pr'] = average_precision_score(y_true, y_probs)
        else:
            metrics['auc_roc'] = np.nan
            metrics['auc_pr'] = np.nan
            
        return metrics


def main(teacher_forcing: bool = True):
    """Main prediction pipeline"""
    
    # Load config
    config_path = Path(__file__).parent / 'config' / 'dataset' / 'survival.yaml'
    with open(config_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    target_disease = dataset_config.get('dataset', {}).get('target_disease', 'bpd')
    mute_decoder = dataset_config.get('dataset', {}).get('mute_decoder', False)
    
    # Load patient splits
    with open('outputs/patient_splits_survival.pkl', 'rb') as f:
        patient_splits = pickle.load(f)
    test_person_ids = patient_splits['test']
    print(f"Loaded {len(test_person_ids)} test patient IDs from saved splits")
    
    # Load raw data
    data_path = dataset_config.get('dataset', {}).get('path', 'data/processed/autoregressive.csv')
    df = pd.read_csv(data_path, low_memory=False)
    print(f"Initial data: {len(df)} records, {df['person_id'].nunique()} patients")
    
    # Sort by person_id and days_on_TPN
    df = df.sort_values(['person_id', 'days_on_TPN'])
    
    # Apply right censoring if enabled
    right_censor = dataset_config.get('dataset', {}).get('right_censor', True)
    if right_censor:
        disease_column = target_disease.replace('dated_', '') if target_disease.startswith('dated_') else target_disease
        disease_date_col = f'{disease_column}_date'
        
        print(f"Applying right censoring using DateOrdered <= {disease_date_col}")
        df['DateOrdered'] = pd.to_datetime(df['DateOrdered'], errors='coerce')
        df[disease_date_col] = pd.to_datetime(df[disease_date_col], errors='coerce')
        
        df = df.loc[(pd.to_datetime(df['DateOrdered']) <= pd.to_datetime(df[disease_date_col])) | 
                   (df[disease_column] == 0), :]
        print(f"After censoring: {len(df)} records from {df['person_id'].nunique()} patients")
    
    # Filter to only test patients
    df_test = df[df['person_id'].isin(test_person_ids)]
    print(f"Filtered to test set: {len(df_test)} records from {df_test['person_id'].nunique()} patients")
    
    # Filter by minimum sequence length
    min_seq_length = dataset_config.get('dataset', {}).get('min_seq_length', 2)
    patient_counts = df_test.groupby('person_id').size()
    valid_patients = patient_counts[patient_counts >= min_seq_length].index
    df_test = df_test[df_test['person_id'].isin(valid_patients)]
    
    # Apply max sequence length if specified
    max_seq_length = dataset_config.get('dataset', {}).get('max_seq_length', None)
    if max_seq_length is not None:
        df_test = df_test.groupby('person_id').tail(max_seq_length)
    
    # Create timestep column
    df_test = df_test.copy()
    df_test['timestep'] = df_test.groupby('person_id').cumcount()
    
    # Get columns from config
    treatment_columns = dataset_config['dataset']['treatment_columns']
    vital_columns = dataset_config['dataset'].get('vital_columns', [])
    static_columns = dataset_config['dataset']['static_columns']
    outcome_columns = [target_disease]
    
    # Load scaler
    with open('outputs/data_scaler_survival.pkl', 'rb') as f:
        scaler_info = pickle.load(f)
    
    # Prepare data arrays
    patients = df_test['person_id'].unique()
    n_patients = len(patients)
    max_len = df_test.groupby('person_id').size().max()
    
    # Initialize arrays
    treatments = np.zeros((n_patients, max_len, len(treatment_columns)))
    vitals = np.zeros((n_patients, max_len, len(vital_columns))) if vital_columns else np.zeros((n_patients, max_len, 0))
    outcomes = np.zeros((n_patients, max_len, 1))
    static_features = np.zeros((n_patients, len(static_columns)))
    active_entries = np.zeros((n_patients, max_len, 1))
    sequence_lengths = np.zeros(n_patients, dtype=int)
    
    # Fill arrays
    for i, pid in enumerate(patients):
        patient_data = df_test[df_test['person_id'] == pid]
        seq_len = len(patient_data)
        sequence_lengths[i] = seq_len
        
        # Extract and scale treatments
        treat_data = patient_data[treatment_columns].values
        treat_scaled = (treat_data - scaler_info['treatments_mean'].values) / scaler_info['treatments_std'].values
        treatments[i, :seq_len, :] = np.nan_to_num(treat_scaled)
        
        # Extract and scale vitals
        if vital_columns:
            vital_data = patient_data[vital_columns].values
            # Replace zero std with 1 to avoid division by zero
            vitals_std = scaler_info['vitals_std'].copy()
            vitals_std[vitals_std == 0] = 1.0
            vital_scaled = (vital_data - scaler_info['vitals_mean']) / vitals_std
            vitals[i, :seq_len, :] = np.nan_to_num(vital_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Extract outcomes (binary, no scaling needed)
        outcomes[i, :seq_len, 0] = patient_data[target_disease].values
        
        # Extract and scale static features
        static_data = patient_data[static_columns].iloc[0].values
        static_scaled = (static_data - scaler_info['static_means']) / scaler_info['static_stds']
        static_features[i, :] = np.nan_to_num(static_scaled)
        
        # Mark active entries
        active_entries[i, :seq_len, 0] = 1.0
    
    # Prepare data in the format expected by the model
    test_data = {
        'sequence_lengths': sequence_lengths - 1,
        'prev_treatments': treatments[:, :-1, :],
        'vitals': vitals[:, 1:, :],
        'next_vitals': vitals[:, 2:, :] if vitals.shape[2] > 0 else np.zeros((n_patients, max_len-2, 0)),
        'current_treatments': treatments[:, 1:, :],
        'static_features': static_features,
        'active_entries': active_entries[:, 1:, :],
        'outputs': outcomes[:, 1:, :],
        'unscaled_outputs': outcomes[:, 1:, :],  # Binary outcomes are not scaled
        'prev_outputs': outcomes[:, :-1, :].copy()  # Copy to avoid aliasing
    }
    
    # Apply muting if enabled
    if mute_decoder:
        print("Note: Decoder inputs (prev_outputs) are muted as per configuration")
        test_data['prev_outputs'] = np.zeros_like(test_data['prev_outputs'])
    
    # Initialize model configuration
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)
    
    with initialize(config_path="config", version_base=None):
        cfg = compose(config_name="config.yaml", 
                       overrides=["+backbone=ct_survival", "+dataset=survival", 
                                "+backbone/ct_hparams=survival"])
        
        # Set dimensions based on actual data
        cfg.model.dim_treatments = len(treatment_columns)
        cfg.model.dim_vitals = len(vital_columns) if vital_columns else 0
        cfg.model.dim_static_features = len(static_columns)
        cfg.model.dim_outcomes = 1
        
        # Create a simple namespace to hold the test data
        class TestDataset:
            def __init__(self, data):
                self.data = data
        
        # Create a mock dataset collection
        class DatasetCollection:
            def __init__(self, test_data):
                self.test_f = TestDataset(test_data)
                self.has_vitals = len(vital_columns) > 0
                self.outcome_columns = outcome_columns
                self.outcome_types = ['binary']  # For binary outcomes
                self.autoregressive = True
                self.projection_horizon = dataset_config.get('dataset', {}).get('projection_horizon', 0)
        
        dataset_collection = DatasetCollection(test_data)
        
        # Initialize model
        model_wrapper = ModelWrapper(cfg, dataset_collection)
        model_wrapper.initialize_model('outputs/trained_model_survival.pt')
    
    # Use the prepared test data
    n_patients = len(test_data['outputs'])
    seq_lengths = test_data['sequence_lengths']
    
    # Prepare batch
    batch = {k: torch.tensor(v, dtype=torch.float32 if k != 'sequence_lengths' else torch.long) 
             for k, v in test_data.items()}
    
    # Make predictions
    if teacher_forcing:
        outcome_pred = model_wrapper.predict_teacher_forcing(batch)
    else:
        outcome_pred = model_wrapper.predict_autoregressive(batch)
    
    outcome_pred = outcome_pred.numpy()
    
    # Print evaluation header
    mode_name = "Teacher Forcing" if teacher_forcing else "Autoregressive"
    print(f"\n=== Multi-Horizon Evaluation ({mode_name}) ===")
    print(f"Total patients: {n_patients}")
    
    # Overall statistics
    y_true_all, _ = MetricsCalculator.extract_predictions_at_horizon(outcome_pred, test_data, 0)
    if len(y_true_all) > 0:
        n_cases_total = np.sum(y_true_all == 1)
        n_controls_total = np.sum(y_true_all == 0)
        print(f"Overall dataset: {n_cases_total} cases, {n_controls_total} controls")
    
    print(f"Outcomes: [{target_disease}]")
    
    # Store results for CSV output
    results_list = []
    
    # Evaluate at multiple horizons (0, 2, 4, 6, 8, 10, 12, 14)
    for days_before in range(0, 16, 2):  # 0 to 14 days before, step by 2
        y_true, y_pred = MetricsCalculator.extract_predictions_at_horizon(
            outcome_pred, test_data, days_before)
        
        if len(y_true) > 0:
            metrics = MetricsCalculator.calculate_binary_metrics(y_true, y_pred)
            
            case_control_str = f" - {metrics['n_cases']} cases, {metrics['n_controls']} controls"
            print(f"\n{days_before} days before end (n={metrics['n_patients']} patients{case_control_str}):")
            
            if days_before == 0:
                print(f"  Minimum sequence length in test data: {seq_lengths.min()}")
            
            if not np.isnan(metrics['auc_roc']):
                print(f"  {target_disease.capitalize()} AUC-ROC: {metrics['auc_roc']:.4f}")
                print(f"  {target_disease.capitalize()} AUC-PR: {metrics['auc_pr']:.4f}")
                print(f"  Prevalence: {metrics['prevalence']:.4f}")
                
                # Store results for CSV
                results_list.append({
                    'disease': target_disease,
                    'days_before': days_before,
                    'auc_roc': metrics['auc_roc'],
                    'auc_pr': metrics['auc_pr'],
                    'prevalence': metrics['prevalence'],
                    'n_patients': metrics['n_patients'],
                    'n_cases': metrics['n_cases'],
                    'n_controls': metrics['n_controls']
                })
            else:
                print(f"  Cannot calculate AUC - only one class present")
    
    # Save results to CSV
    if results_list:
        import pandas as pd
        results_df = pd.DataFrame(results_list)
        csv_filename = f'outputs/autoregressive_results_survival_{target_disease}.csv'
        results_df.to_csv(csv_filename, index=False)
        print(f"\nResults saved to {csv_filename}")


if __name__ == "__main__":
    main(teacher_forcing=True)