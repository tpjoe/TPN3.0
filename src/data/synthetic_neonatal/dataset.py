import pandas as pd
import numpy as np
import logging
from dateutil.parser import parse
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
                                 horizons: list,
                                 landmarks: list,
                                 min_seq_length: int = None,
                                 max_seq_length: int = None,
                                 max_number: int = None,
                                 data_seed: int = 100,
                                 **kwargs):
    """
    Load and pre-process synthetic neonatal CSV data to match MIMIC-III format
    
    Returns: treatments, outcomes, vitals, static_features, scaling_params
    """
    logger.info(f"Loading CSV data from {data_path}")
    # Use low_memory=False to avoid DtypeWarning about mixed types
    df = pd.read_csv(data_path, low_memory=False)
    
    
    # First sort by person_id and timestamp to ensure proper time ordering
    df = df.sort_values(['person_id', timestamp_column])
    
    # Apply right censoring if requested
    disease_column = outcome_columns[0]
    disease_date_col = f'{disease_column}_date'
    
    # Convert date columns to datetime
    def parse_date_or_nat(s):
        try:
            return pd.to_datetime(parse(str(s)))
        except Exception:
            return pd.NaT
    
    df['DateOrdered'] = df['DateOrdered'].apply(parse_date_or_nat)
    df[disease_date_col] = df[disease_date_col].apply(parse_date_or_nat)
    
    # Apply censoring: keep only records where DateOrdered <= disease_date (or disease==0)
    _, abs_min_days, _ = horizons[0]
    df = df.loc[(pd.to_datetime(df['DateOrdered']) <= (pd.to_datetime(df[disease_date_col]) - pd.to_timedelta(abs_min_days, unit='d'))) | 
                (df[disease_column] == 0), :]
    
    # Special handling for max_chole_TPNEHR
    if 'max_chole_TPNEHR' in disease_column:
        unique_patients = df.drop_duplicates('person_id')
        required_date_cols = ['chole_EHR_date', 'max_chole_TPN_date']
        missing_cols = [col for col in required_date_cols if col not in df.columns]
        exclude_mask = (unique_patients['max_chole_TPNEHR'] == 1) & \
                        (unique_patients[required_date_cols].isna().all(axis=1))
        exclude_person_ids = unique_patients.loc[exclude_mask, 'person_id']
        
        # Exclude these patients
        df = df.loc[~df['person_id'].isin(exclude_person_ids.values), :]
    
    # IMPORTANT: Calculate outcomes_bucket BEFORE landmark filtering
    # This ensures we capture the true event time even if we truncate data later
    days_to_disease_full = (df[disease_date_col] - df['DateOrdered']).dt.days
    df["outcomes_bucket"] = len(horizons)  # Default to censored
    
    for bucket_idx, (_, min_days, max_days) in enumerate(horizons):
        mask = (days_to_disease_full >= min_days) & (days_to_disease_full <= max_days)
        df.loc[mask, "outcomes_bucket"] = bucket_idx

    # sliding windows and screen by landmarks
    if landmarks:  # If landmarks are specified
        # For each patient, only include them in landmarks they've reached
        dfs_by_landmark = []
        
        # Group by person_id to process each patient
        for person_id, patient_df in df.groupby('person_id'):
            max_day = patient_df['day_since_birth'].max()
            
            # For each landmark this patient has reached
            for lm in landmarks:
                if max_day >= lm:
                    # Include data up to landmark day
                    patient_lm_df = patient_df.loc[patient_df.day_since_birth <= lm].copy()
                    patient_lm_df['landmark'] = lm
                    dfs_by_landmark.append(patient_lm_df)
            
            # Add the full sequence only if it's different from existing landmarks
            if max_day not in landmarks:
                patient_full_df = patient_df.copy()
                patient_full_df['landmark'] = max(landmarks) + 1  # Use a value that won't conflict
                dfs_by_landmark.append(patient_full_df)
        
        df = pd.concat(dfs_by_landmark, ignore_index=True)
    else:  # No landmarks - just use full sequences
        df['landmark'] = 0  # Single landmark value for all patients

    # Create timestep column for each patient-landmark combination
    df = df.copy()
    
    # Create unique subject_id that combines person_id and landmark
    df['subject_id'] = df['person_id'].astype(str) + '_L' + df['landmark'].astype(str)
    
    # Now create timestep based on this new subject_id
    df['timestep'] = df.groupby('subject_id').cumcount()

    # Set multi-index using the new subject_id
    df = df.set_index(['subject_id', 'timestep'])
    
    # Sort index to ensure proper ordering
    df = df.sort_index()
    
    # Extract different data types as DataFrames with multi-index
    treatments = df[treatment_columns]
    vitals = df[vital_columns] if vital_columns else pd.DataFrame()
    outcomes = df[outcome_columns]
    
    # Add landmark as a categorical static feature
    # Convert landmark values to categorical indices (0, 1, 2, ...)
    unique_landmarks = sorted(df['landmark'].unique())
    landmark_to_idx = {lm: idx for idx, lm in enumerate(unique_landmarks)}
    # Use copy() to avoid fragmentation warning
    df = df.copy()
    df['landmark_cat'] = df['landmark'].map(landmark_to_idx)
    
    # Extract static features (one per patient)
    static_columns_with_landmark = static_columns + ['landmark_cat']
    static_features = df[static_columns_with_landmark].groupby('subject_id').first()
    
    # Filter by sequence length
    user_sizes = df.groupby('subject_id').size()
    filtered_users = user_sizes.index[user_sizes >= min_seq_length] if min_seq_length is not None else user_sizes.index
    
    # Apply filtering
    treatments = treatments.loc[filtered_users]
    outcomes = outcomes.loc[filtered_users]
    vitals = vitals.loc[filtered_users] if not vitals.empty else vitals
    filtered_df = df.loc[filtered_users].copy()
    
    # Crop to max sequence length (take last N timesteps)
    if max_seq_length is not None:
        treatments = treatments.groupby('subject_id').tail(max_seq_length)
        outcomes = outcomes.groupby('subject_id').tail(max_seq_length)
        vitals = vitals.groupby('subject_id').tail(max_seq_length) if not vitals.empty else vitals
        filtered_df = filtered_df.groupby('subject_id').tail(max_seq_length)
        
        # Reset timestep indices to start from 0 for each patient after tail()
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
        filtered_df = reset_timesteps(filtered_df)
    
    static_features = static_features[static_features.index.isin(filtered_users)]
    
    logger.info(f'Number of patients filtered: {len(filtered_users)}.')
        
    # Global scaling (same as MIMIC-III)
    if not vitals.empty:
        mean_vitals = np.mean(vitals, axis=0)
        std_vitals = np.std(vitals, axis=0)
        # Replace zero std with 1 to avoid division by zero
        std_vitals[std_vitals == 0] = 1.0
        vitals = (vitals - mean_vitals) / std_vitals
        # Handle NaN values from zero std
        vitals = vitals.fillna(0.0)
    
    # Scale treatments separately
    treatments_mean = np.mean(treatments, axis=0)
    treatments_std = np.std(treatments, axis=0)
    # Replace zero std with 1 to avoid division by zero
    treatments_std[treatments_std == 0] = 1.0
    treatments = (treatments - treatments_mean) / treatments_std
    # Handle NaN values from zero std
    treatments = treatments.fillna(0.0)
    
    # Process static features - only scale gest_age and bw
    continuous_cols = ['gest_age', 'bw']
    
    # Calculate mean and std only for continuous columns
    static_mean = []
    static_std = []
    for col in static_features.columns:
        if col in continuous_cols:
            static_mean.append(np.mean(static_features[col]))
            std_val = np.std(static_features[col])
            static_std.append(std_val if std_val != 0 else 1.0)
        else:
            static_mean.append(0.0)  # Placeholder for categorical
            static_std.append(1.0)   # Placeholder for categorical
    
    static_mean = np.array(static_mean)
    static_std = np.array(static_std)
    
    processed_static = []
    for col in static_features.columns:
        col_data = static_features[col]
        if col in continuous_cols:
            # Scale continuous columns
            mean_val = np.mean(col_data)
            std_val = np.std(col_data)
            if std_val == 0:
                std_val = 1.0
            scaled_data = (col_data - mean_val) / std_val
            scaled_data = scaled_data.fillna(0.0)
            processed_static.append(scaled_data)
        else:
            # Keep categorical columns as is (including landmark_cat)
            processed_static.append(col_data)
    
    static_features = pd.concat(processed_static, axis=1)
    
    # Save scaling info directly here
    import pickle
    from pathlib import Path
    root_dir = Path(__file__).parent.parent.parent.parent  # Go up one more level to project root
    output_dir = root_dir / 'outputs'
    output_dir.mkdir(exist_ok=True)
    scaler_info = {
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
    
    # Create scaling_params for compatibility
    scaling_params = scaler_info

    # outcomes_bucket was already calculated before landmark filtering
    # Just extract it from filtered_df with the same MultiIndex structure
    outcomes_bucket = filtered_df["outcomes_bucket"].to_frame()
    return treatments, outcomes, vitals, static_features, scaling_params, outcomes_bucket


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
                 horizons: list,
                 landmarks: list,
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
        """
        # Load data directly here instead of relying on parent's init
        self.seed = seed
        
        # Store outcome-related attributes
        self.outcome_columns = outcome_columns
        self.outcome_types = outcome_types
        self.min_length_filter = min_length_filter
        self.one_seq_per_patient_eval = one_seq_per_patient_eval
        self.horizons = horizons
        self.landmarks = landmarks
        self.vital_columns = vital_columns
        treatments, outcomes, vitals, static_features, scaling_params, outcomes_bucket = \
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
                horizons=horizons,
                landmarks=landmarks,
            )
        
        # Train/val/test random_split at person level (not subject_id level)
        # Extract unique person_ids from subject_ids
        static_features['person_id'] = static_features.index.str.replace(r'_L\d+', '', regex=True)
        unique_person_ids = static_features['person_id'].unique()
        
        # Split at person level
        train_val_persons, test_persons = train_test_split(unique_person_ids, test_size=split['test'], random_state=seed)
        
        # Create masks for static features based on person_id
        static_features_test = static_features[static_features['person_id'].isin(test_persons)]
        static_features = static_features[static_features['person_id'].isin(train_val_persons)]
        
        # Drop the temporary person_id column
        static_features = static_features.drop('person_id', axis=1)
        static_features_test = static_features_test.drop('person_id', axis=1)
        
        
        # Fix MultiIndex selection issue - properly select all timesteps for each patient
        train_val_mask = treatments.index.get_level_values('subject_id').isin(static_features.index)
        test_mask = treatments.index.get_level_values('subject_id').isin(static_features_test.index)
        
        treatments, outcomes, vitals, outcomes_bucket, treatments_test, outcomes_test, vitals_test, outcomes_bucket_test = \
            treatments.loc[train_val_mask], \
            outcomes.loc[train_val_mask], \
            vitals.loc[train_val_mask] if not vitals.empty else vitals, \
            outcomes_bucket.loc[train_val_mask], \
            treatments.loc[test_mask], \
            outcomes.loc[test_mask], \
            vitals.loc[test_mask] if not vitals.empty else vitals, \
            outcomes_bucket.loc[test_mask]            

        if split['val'] > 0.0:
            # Split at person level for train/val too
            static_features['person_id'] = static_features.index.str.replace(r'_L\d+', '', regex=True)
            unique_train_val_persons = static_features['person_id'].unique()
            
            train_persons, val_persons = train_test_split(unique_train_val_persons,
                                                          test_size=split['val'] / (1 - split['test']),
                                                          random_state=2 * seed)
            
            static_features_train = static_features[static_features['person_id'].isin(train_persons)]
            static_features_val = static_features[static_features['person_id'].isin(val_persons)]
            
            # Drop the temporary person_id column
            static_features_train = static_features_train.drop('person_id', axis=1)
            static_features_val = static_features_val.drop('person_id', axis=1)
            # Fix MultiIndex selection for train/val split too
            train_mask = treatments.index.get_level_values('subject_id').isin(static_features_train.index)
            val_mask = treatments.index.get_level_values('subject_id').isin(static_features_val.index)
            
            treatments_train, outcomes_train, vitals_train, outcomes_bucket_train, treatments_val, outcomes_val, vitals_val, outcomes_bucket_val = \
                treatments.loc[train_mask], \
                outcomes.loc[train_mask], \
                vitals.loc[train_mask] if not vitals.empty else vitals, \
                outcomes_bucket.loc[train_mask], \
                treatments.loc[val_mask], \
                outcomes.loc[val_mask], \
                vitals.loc[val_mask] if not vitals.empty else vitals, \
                outcomes_bucket.loc[val_mask]
        else:
            static_features_train = static_features
            treatments_train, outcomes_train, vitals_train, outcomes_bucket_train = \
                treatments, outcomes, vitals, outcomes_bucket
            # For no validation split case, define empty val variables
            treatments_val = outcomes_val = vitals_val = outcomes_bucket_val = static_features_val = None

        self.train_f = MIMIC3RealDataset(treatments_train, outcomes_train, vitals_train, outcomes_bucket_train, static_features_train, scaling_params, 'train')
        if split['val'] > 0.0:
            self.val_f = MIMIC3RealDataset(treatments_val, outcomes_val, vitals_val, outcomes_bucket_val, static_features_val, scaling_params, 'val')
        self.test_f = MIMIC3RealDataset(treatments_test, outcomes_test, vitals_test, outcomes_bucket_test, static_features_test, scaling_params, 'test')
        
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