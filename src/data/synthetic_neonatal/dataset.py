import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split

from src import ROOT_PATH
from src.data.mimic_iii.real_dataset import MIMIC3RealDataset, MIMIC3RealDatasetCollection

logger = logging.getLogger(__name__)


def load_synthetic_neonatal_data(data_path: str,
                                 vital_columns: list,
                                 static_columns: list,
                                 treatment_columns: list,
                                 outcome_column: str,
                                 timestamp_column: str,
                                 min_seq_length: int = None,
                                 max_seq_length: int = None,
                                 max_number: int = None,
                                 data_seed: int = 100,
                                 **kwargs):
    """
    Load and pre-process synthetic neonatal CSV data to match MIMIC-III format
    
    Returns: treatments, outcomes, vitals, static_features, outcomes_unscaled, scaling_params
    """
    logger.info(f"Loading CSV data from {data_path}")
    df = pd.read_csv(data_path)
    
    # First sort by person_id and timestamp to ensure proper time ordering
    df = df.sort_values(['person_id', timestamp_column])
    
    # Create timestep column for each patient
    # Group by person_id and create sequential timesteps
    df['timestep'] = df.groupby('person_id').cumcount()
    
    # Set multi-index (person_id as subject_id, timestep)
    df = df.set_index(['person_id', 'timestep'])
    df.index.names = ['subject_id', 'timestep']  # Rename to match MIMIC-III convention
    
    # Sort index to ensure proper ordering
    df = df.sort_index()
    
    # Extract different data types as DataFrames with multi-index
    treatments = df[treatment_columns]
    vitals = df[vital_columns] if vital_columns else pd.DataFrame()
    outcomes = df[[outcome_column]]
    
    # Extract static features (one per patient)
    # Get unique values per patient from the first timestep
    static_features = df[static_columns].groupby('subject_id').first()
    
    # Filter by sequence length
    user_sizes = df.groupby('subject_id').size()
    filtered_users = user_sizes.index[user_sizes >= min_seq_length] if min_seq_length is not None else user_sizes.index
    
    if max_number is not None:
        np.random.seed(data_seed)
        filtered_users = np.random.choice(filtered_users, size=min(max_number, len(filtered_users)), replace=False)
    
    # Apply filtering
    treatments = treatments.loc[filtered_users]
    outcomes = outcomes.loc[filtered_users]
    vitals = vitals.loc[filtered_users] if not vitals.empty else vitals
    
    # Crop to max sequence length
    if max_seq_length is not None:
        treatments = treatments.groupby('subject_id').head(max_seq_length)
        outcomes = outcomes.groupby('subject_id').head(max_seq_length)
        vitals = vitals.groupby('subject_id').head(max_seq_length) if not vitals.empty else vitals
    
    static_features = static_features[static_features.index.isin(filtered_users)]
    
    logger.info(f'Number of patients filtered: {len(filtered_users)}.')
    
    # Store unscaled outcomes
    outcomes_unscaled = outcomes.copy()
    
    # Global scaling (same as MIMIC-III)
    # Scale all continuous data
    if not vitals.empty:
        all_time_varying = pd.concat([vitals, outcomes], axis=1)
    else:
        all_time_varying = outcomes
        
    mean = np.mean(all_time_varying, axis=0)
    std = np.std(all_time_varying, axis=0)
    
    # Apply scaling
    if not vitals.empty:
        vitals = (vitals - mean[vitals.columns]) / std[vitals.columns]
    outcomes = (outcomes - mean[outcomes.columns]) / std[outcomes.columns]
    
    # Scale treatments separately
    treatments_mean = np.mean(treatments, axis=0)
    treatments_std = np.std(treatments, axis=0)
    treatments = (treatments - treatments_mean) / treatments_std
    
    # Process static features (standardize continuous, keep categorical as is)
    processed_static = []
    for col in static_features.columns:
        col_data = static_features[col]
        # Assume numeric columns are continuous
        if pd.api.types.is_numeric_dtype(col_data):
            mean_val = np.mean(col_data)
            std_val = np.std(col_data)
            processed_static.append((col_data - mean_val) / std_val)
        else:
            processed_static.append(col_data)
    
    static_features = pd.concat(processed_static, axis=1)
    
    scaling_params = {
        'output_means': mean[outcome_column].reshape(-1),
        'output_stds': std[outcome_column].reshape(-1),
    }
    
    return treatments, outcomes, vitals, static_features, outcomes_unscaled, scaling_params


class SyntheticNeonatalDatasetCollection(MIMIC3RealDatasetCollection):
    """
    Dataset collection for synthetic neonatal data - inherits from MIMIC3RealDatasetCollection
    """
    def __init__(self,
                 path: str,
                 vital_columns: list,
                 static_columns: list, 
                 treatment_columns: list,
                 outcome_column: str,
                 timestamp_column: str,
                 min_seq_length: int = 30,
                 max_seq_length: int = 80,
                 seed: int = 100,
                 max_number: int = None,
                 split: dict = {'val': 0.2, 'test': 0.2},
                 projection_horizon: int = 5,
                 autoregressive=True,
                 **kwargs):
        """
        Args:
            path: Path to CSV file (relative to ROOT_PATH)
            vital_columns: List of vital/time-varying columns
            static_columns: List of static feature columns
            treatment_columns: List of treatment columns
            outcome_column: Outcome column name
            min_seq_length: Min sequence length in cohort
            max_seq_length: Max sequence length in cohort
            seed: Seed for random cohort patient selection
            max_number: Maximum number of patients in cohort
            split: Ratio of train / val / test split
            projection_horizon: Range of tau-step-ahead prediction
            autoregressive: Whether to use autoregressive mode
        """
        # Load data directly here instead of relying on parent's init
        self.seed = seed
        
        treatments, outcomes, vitals, static_features, outcomes_unscaled, scaling_params = \
            load_synthetic_neonatal_data(
                ROOT_PATH + '/' + path,
                vital_columns=vital_columns,
                static_columns=static_columns,
                treatment_columns=treatment_columns,
                outcome_column=outcome_column,
                timestamp_column=timestamp_column,
                min_seq_length=min_seq_length,
                max_seq_length=max_seq_length,
                max_number=max_number,
                data_seed=seed
            )
        
        # Train/val/test random_split
        static_features, static_features_test = train_test_split(static_features, test_size=split['test'], random_state=seed)
        treatments, outcomes, vitals, outcomes_unscaled, treatments_test, outcomes_test, vitals_test, outcomes_unscaled_test = \
            treatments.loc[static_features.index], \
            outcomes.loc[static_features.index], \
            vitals.loc[static_features.index] if not vitals.empty else vitals, \
            outcomes_unscaled.loc[static_features.index], \
            treatments.loc[static_features_test.index], \
            outcomes.loc[static_features_test.index], \
            vitals.loc[static_features_test.index] if not vitals.empty else vitals, \
            outcomes_unscaled.loc[static_features_test.index]

        if split['val'] > 0.0:
            static_features_train, static_features_val = train_test_split(static_features,
                                                                          test_size=split['val'] / (1 - split['test']),
                                                                          random_state=2 * seed)
            treatments_train, outcomes_train, vitals_train, outcomes_unscaled_train, treatments_val, outcomes_val, vitals_val, \
                outcomes_unscaled_val = \
                treatments.loc[static_features_train.index], \
                outcomes.loc[static_features_train.index], \
                vitals.loc[static_features_train.index] if not vitals.empty else vitals, \
                outcomes_unscaled.loc[static_features_train.index], \
                treatments.loc[static_features_val.index], \
                outcomes.loc[static_features_val.index], \
                vitals.loc[static_features_val.index] if not vitals.empty else vitals, \
                outcomes_unscaled.loc[static_features_val.index]
        else:
            static_features_train = static_features
            treatments_train, outcomes_train, vitals_train, outcomes_unscaled_train = \
                treatments, outcomes, vitals, outcomes_unscaled

        self.train_f = MIMIC3RealDataset(treatments_train, outcomes_train, vitals_train, static_features_train,
                                         outcomes_unscaled_train, scaling_params, 'train')
        if split['val'] > 0.0:
            self.val_f = MIMIC3RealDataset(treatments_val, outcomes_val, vitals_val, static_features_val, outcomes_unscaled_val,
                                           scaling_params, 'val')
        self.test_f = MIMIC3RealDataset(treatments_test, outcomes_test, vitals_test, static_features_test, outcomes_unscaled_test,
                                        scaling_params, 'test')

        self.projection_horizon = projection_horizon
        self.has_vitals = len(vital_columns) > 0 if vital_columns else False
        self.autoregressive = autoregressive
        self.processed_data_encoder = True