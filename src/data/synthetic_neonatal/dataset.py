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
                                 outcome_columns: list,
                                 outcome_types: list,
                                 timestamp_column: str,
                                 min_seq_length: int = None,
                                 max_seq_length: int = None,
                                 max_number: int = None,
                                 data_seed: int = 100,
                                 right_censor: bool = False,
                                 **kwargs):
    """
    Load and pre-process synthetic neonatal CSV data to match MIMIC-III format
    
    Returns: treatments, outcomes, vitals, static_features, outcomes_unscaled, scaling_params
    """
    logger.info(f"Loading CSV data from {data_path}")
    # Use low_memory=False to avoid DtypeWarning about mixed types
    df = pd.read_csv(data_path, low_memory=False)
    
    
    # First sort by person_id and timestamp to ensure proper time ordering
    df = df.sort_values(['person_id', timestamp_column])
    
    # Shift outcomes by 1 position earlier if requested (for 1-day-ahead prediction)
    outcome_shift = kwargs.get('outcome_shift', 0)
    if outcome_shift > 0:
        logger.info(f"Shifting outcomes {outcome_shift} position(s) earlier for each patient")
        shifted_dfs = []
        for person_id, person_df in df.groupby('person_id'):
            person_df = person_df.copy()
            # Shift outcome columns up by outcome_shift positions
            for col in outcome_columns:
                if col in person_df.columns:
                    person_df[col] = person_df[col].shift(-outcome_shift)
            # Drop last outcome_shift rows (which now have NaN outcomes)
            person_df = person_df.iloc[:-outcome_shift]
            if len(person_df) > 0:  # Only keep if there are remaining rows
                shifted_dfs.append(person_df)
        
        df = pd.concat(shifted_dfs, ignore_index=True)
        logger.info(f"After outcome shifting: {len(df)} timesteps from {df['person_id'].nunique()} patients")
    
    # Apply right censoring if requested
    if right_censor:
        # Find binary outcome columns
        binary_dated_outcomes = []
        for i, (col, otype) in enumerate(zip(outcome_columns, outcome_types)):
            if otype == 'binary':
                binary_dated_outcomes.append(col)
        
        if binary_dated_outcomes:
            # Use the first binary outcome for censoring (typically there's only one)
            outcome_column = binary_dated_outcomes[0]
            # Remove 'dated_' prefix if it exists, otherwise use as is
            disease_column = outcome_column.replace('dated_', '') if outcome_column.startswith('dated_') else outcome_column
            disease_date_col = f'{disease_column}_date'
            
            # Check required columns exist
            if 'DateOrdered' not in df.columns:
                raise ValueError(f"DateOrdered column not found in data. Available columns: {df.columns.tolist()}")
            
            if disease_date_col not in df.columns:
                raise ValueError(f"{disease_date_col} column not found in data. Available columns: {df.columns.tolist()}")
            
            if disease_column not in df.columns:
                raise ValueError(f"{disease_column} column not found in data. Available columns: {df.columns.tolist()}")
            
            logger.info(f"Applying right censoring using DateOrdered <= {disease_date_col}")
            
            
            # Convert date columns to datetime
            df['DateOrdered'] = pd.to_datetime(df['DateOrdered'], errors='coerce')
            df[disease_date_col] = pd.to_datetime(df[disease_date_col], errors='coerce')
            
            # Apply censoring: keep only records where DateOrdered <= disease_date (or disease==0)
            df = df.loc[(pd.to_datetime(df['DateOrdered']) <= pd.to_datetime(df[disease_date_col])) | 
                       (df[disease_column] == 0), :]
            logger.info(f"After DateOrdered censoring: {len(df)} timesteps from {df['person_id'].nunique()} patients")
            
            
            # Special handling for max_chole_TPNEHR
            if 'max_chole_TPNEHR' in disease_column:
                logger.info("Applying special exclusion for max_chole_TPNEHR cases")
                
                # Get unique patients
                unique_patients = df.drop_duplicates('person_id')
                
                # Check for both possible date columns
                required_date_cols = ['chole_EHR_date', 'max_chole_TPN_date']
                missing_cols = [col for col in required_date_cols if col not in df.columns]
                if missing_cols:
                    raise ValueError(f"Required date columns {missing_cols} not found for max_chole_TPNEHR exclusion. Available columns: {df.columns.tolist()}")
                
                if 'max_chole_TPNEHR' not in df.columns:
                    raise ValueError(f"max_chole_TPNEHR column required but not found. Available columns: {df.columns.tolist()}")
                
                # Find patients with max_chole_TPNEHR==1 but all date columns are NaN
                exclude_mask = (unique_patients['max_chole_TPNEHR'] == 1) & \
                             (unique_patients[required_date_cols].isna().all(axis=1))
                exclude_person_ids = unique_patients.loc[exclude_mask, 'person_id']
                
                # Exclude these patients
                df = df.loc[~df['person_id'].isin(exclude_person_ids.values), :]
                logger.info(f"Excluded {len(exclude_person_ids)} patients with max_chole_TPNEHR==1 but no dates")
                logger.info(f"After exclusion: {len(df)} timesteps from {df['person_id'].nunique()} patients")
    
    # Create timestep column for each patient
    # Group by person_id and create sequential timesteps
    # Use copy() to avoid PerformanceWarning about DataFrame fragmentation
    df = df.copy()
    df['timestep'] = df.groupby('person_id').cumcount()
    
    # Set multi-index (person_id as subject_id, timestep)
    df = df.set_index(['person_id', 'timestep'])
    df.index.names = ['subject_id', 'timestep']  # Rename to match MIMIC-III convention
    
    # Sort index to ensure proper ordering
    df = df.sort_index()
    
    # Extract different data types as DataFrames with multi-index
    treatments = df[treatment_columns]
    vitals = df[vital_columns] if vital_columns else pd.DataFrame()
    outcomes = df[outcome_columns]
    
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
    
    # Crop to max sequence length (take last N timesteps)
    if max_seq_length is not None:
        treatments = treatments.groupby('subject_id').tail(max_seq_length)
        outcomes = outcomes.groupby('subject_id').tail(max_seq_length)
        vitals = vitals.groupby('subject_id').tail(max_seq_length) if not vitals.empty else vitals
        
        # Reset timestep indices to start from 0 for each patient after tail()
        # This ensures proper alignment for the reshape operation
        def reset_timesteps(df):
            df = df.reset_index(level=1, drop=True)
            df.index = pd.MultiIndex.from_arrays([
                df.index,
                df.groupby(level=0).cumcount()
            ], names=['subject_id', 'timestep'])
            return df
        
        treatments = reset_timesteps(treatments)
        outcomes = reset_timesteps(outcomes)
        vitals = reset_timesteps(vitals) if not vitals.empty else vitals
    
    static_features = static_features[static_features.index.isin(filtered_users)]
    
    logger.info(f'Number of patients filtered: {len(filtered_users)}.')
    
    # Store unscaled outcomes
    outcomes_unscaled = outcomes.copy()
    
    # Global scaling (same as MIMIC-III)
    # Scale vitals first
    if not vitals.empty:
        mean_vitals = np.mean(vitals, axis=0)
        std_vitals = np.std(vitals, axis=0)
        # Replace zero std with 1 to avoid division by zero
        std_vitals[std_vitals == 0] = 1.0
        vitals = (vitals - mean_vitals) / std_vitals
        # Handle NaN values from zero std
        vitals = vitals.fillna(0.0)
    
    # Scale outcomes based on their types
    mean_outcomes = {}
    std_outcomes = {}
    scaled_outcomes = []
    
    for col, otype in zip(outcome_columns, outcome_types):
        if otype == 'binary':
            # Keep binary outcomes as is (0 or 1)
            scaled_outcomes.append(outcomes[col])
            mean_outcomes[col] = 0.0  # Dummy values for compatibility
            std_outcomes[col] = 1.0
        else:
            # Scale continuous outcomes
            col_mean = np.mean(outcomes[col])
            col_std = np.std(outcomes[col])
            # Replace zero std with 1 to avoid division by zero
            if col_std == 0:
                col_std = 1.0
            scaled_col = (outcomes[col] - col_mean) / col_std
            scaled_col = scaled_col.fillna(0.0)
            scaled_outcomes.append(scaled_col)
            mean_outcomes[col] = col_mean
            std_outcomes[col] = col_std
    
    # Reconstruct outcomes DataFrame with scaled values
    outcomes = pd.concat(scaled_outcomes, axis=1)
    
    # Scale treatments separately
    treatments_mean = np.mean(treatments, axis=0)
    treatments_std = np.std(treatments, axis=0)
    # Replace zero std with 1 to avoid division by zero
    treatments_std[treatments_std == 0] = 1.0
    treatments = (treatments - treatments_mean) / treatments_std
    # Handle NaN values from zero std
    treatments = treatments.fillna(0.0)
    
    # Process static features (standardize continuous, keep categorical as is)
    # First, get the mean and std for static features before scaling
    static_features_numeric = static_features.select_dtypes(include=[np.number])
    static_mean = np.mean(static_features_numeric.values, axis=0)
    static_std = np.std(static_features_numeric.values, axis=0)
    # Replace zero std with 1 to avoid division by zero
    static_std[static_std == 0] = 1.0
    
    processed_static = []
    for col in static_features.columns:
        col_data = static_features[col]
        # Assume numeric columns are continuous
        if pd.api.types.is_numeric_dtype(col_data):
            mean_val = np.mean(col_data)
            std_val = np.std(col_data)
            # Replace zero std with 1 to avoid division by zero
            if std_val == 0:
                std_val = 1.0
            scaled_data = (col_data - mean_val) / std_val
            # Handle NaN values from zero std
            scaled_data = scaled_data.fillna(0.0)
            processed_static.append(scaled_data)
        else:
            processed_static.append(col_data)
    
    static_features = pd.concat(processed_static, axis=1)
    
    scaling_params = {
        'output_means': mean_outcomes,
        'output_stds': std_outcomes,
    }
    
    # Save scaling info directly here
    import pickle
    from pathlib import Path
    root_dir = Path(__file__).parent.parent.parent.parent  # Go up one more level to project root
    output_dir = root_dir / 'outputs'
    output_dir.mkdir(exist_ok=True)
    scaler_info = {
        'output_means': mean_outcomes,
        'output_stds': std_outcomes,
        'treatments_mean': treatments_mean,
        'treatments_std': treatments_std,
        'vitals_mean': mean_vitals.values if not vitals.empty else None,
        'vitals_std': std_vitals.values if not vitals.empty else None,
        'static_means': static_mean,
        'static_stds': static_std,
    }
    with open(output_dir / 'data_scaler.pkl', 'wb') as f:
        pickle.dump(scaler_info, f)
    logger.info(f"Saved scaler to {output_dir / 'data_scaler.pkl'}")
    
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
                 outcome_columns: list,
                 outcome_types: list,
                 timestamp_column: str,
                 min_seq_length: int = 30,
                 max_seq_length: int = 80,
                 seed: int = 100,
                 max_number: int = None,
                 split: dict = {'val': 0.2, 'test': 0.2},
                 projection_horizon: int = 5,
                 autoregressive=True,
                 right_censor: bool = False,
                 min_length_filter: bool = False,
                 one_seq_per_patient_eval: bool = False,
                 outcome_shift: int = 0,
                 **kwargs):
        """
        Args:
            path: Path to CSV file (relative to ROOT_PATH)
            vital_columns: List of vital/time-varying columns
            static_columns: List of static feature columns
            treatment_columns: List of treatment columns
            outcome_columns: List of outcome column names
            outcome_types: List of outcome types ('continuous' or 'binary')
            min_seq_length: Min sequence length in cohort
            max_seq_length: Max sequence length in cohort
            seed: Seed for random cohort patient selection
            max_number: Maximum number of patients in cohort
            split: Ratio of train / val / test split
            projection_horizon: Range of tau-step-ahead prediction
            autoregressive: Whether to use autoregressive mode
            outcome_shift: Number of days to shift outcomes earlier (0 = predict same day, 1 = predict 1 day ahead)
        """
        # Load data directly here instead of relying on parent's init
        self.seed = seed
        
        # Store outcome-related attributes
        self.outcome_columns = outcome_columns
        self.outcome_types = outcome_types
        self.min_length_filter = min_length_filter
        self.one_seq_per_patient_eval = one_seq_per_patient_eval
        
        treatments, outcomes, vitals, static_features, outcomes_unscaled, scaling_params = \
            load_synthetic_neonatal_data(
                ROOT_PATH + '/' + path,
                vital_columns=vital_columns,
                static_columns=static_columns,
                treatment_columns=treatment_columns,
                outcome_columns=outcome_columns,
                outcome_types=outcome_types,
                timestamp_column=timestamp_column,
                min_seq_length=min_seq_length,
                max_seq_length=max_seq_length,
                max_number=max_number,
                data_seed=seed,
                right_censor=right_censor,
                outcome_shift=outcome_shift
            )
        
        # Train/val/test random_split
        
        static_features, static_features_test = train_test_split(static_features, test_size=split['test'], random_state=seed)
        
        
        # Fix MultiIndex selection issue - properly select all timesteps for each patient
        train_val_mask = treatments.index.get_level_values('subject_id').isin(static_features.index)
        test_mask = treatments.index.get_level_values('subject_id').isin(static_features_test.index)
        
        treatments, outcomes, vitals, outcomes_unscaled, treatments_test, outcomes_test, vitals_test, outcomes_unscaled_test = \
            treatments.loc[train_val_mask], \
            outcomes.loc[train_val_mask], \
            vitals.loc[train_val_mask] if not vitals.empty else vitals, \
            outcomes_unscaled.loc[train_val_mask], \
            treatments.loc[test_mask], \
            outcomes.loc[test_mask], \
            vitals.loc[test_mask] if not vitals.empty else vitals, \
            outcomes_unscaled.loc[test_mask]
            

        if split['val'] > 0.0:
            static_features_train, static_features_val = train_test_split(static_features,
                                                                          test_size=split['val'] / (1 - split['test']),
                                                                          random_state=2 * seed)
            # Fix MultiIndex selection for train/val split too
            train_mask = treatments.index.get_level_values('subject_id').isin(static_features_train.index)
            val_mask = treatments.index.get_level_values('subject_id').isin(static_features_val.index)
            
            treatments_train, outcomes_train, vitals_train, outcomes_unscaled_train, treatments_val, outcomes_val, vitals_val, \
                outcomes_unscaled_val = \
                treatments.loc[train_mask], \
                outcomes.loc[train_mask], \
                vitals.loc[train_mask] if not vitals.empty else vitals, \
                outcomes_unscaled.loc[train_mask], \
                treatments.loc[val_mask], \
                outcomes.loc[val_mask], \
                vitals.loc[val_mask] if not vitals.empty else vitals, \
                outcomes_unscaled.loc[val_mask]
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
        

        # Save patient splits after all splits are done
        import pickle
        patient_splits = {
            'train': static_features_train.index.tolist(),
            'val': static_features_val.index.tolist() if split['val'] > 0.0 else [],
            'test': static_features_test.index.tolist()
        }
        with open(ROOT_PATH + '/outputs/patient_splits.pkl', 'wb') as f:
            pickle.dump(patient_splits, f)
        logger.info(f"Saved patient splits - Train: {len(patient_splits['train'])}, Val: {len(patient_splits['val'])}, Test: {len(patient_splits['test'])}")

        self.projection_horizon = projection_horizon
        self.has_vitals = len(vital_columns) > 0 if vital_columns else False
        self.autoregressive = autoregressive
        self.processed_data_encoder = True