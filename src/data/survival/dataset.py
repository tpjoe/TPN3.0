import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split

from src import ROOT_PATH
from src.data.mimic_iii.real_dataset import MIMIC3RealDataset, MIMIC3RealDatasetCollection
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def load_survival_data(data_path: str,
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
                                 sliding_windows: bool = False,
                                 **kwargs):
    """
    Load and pre-process survival CSV data to match MIMIC-III format
    
    Returns: treatments, outcomes, vitals, static_features, outcomes_unscaled, scaling_params
    """
    logger.info(f"Loading CSV data from {data_path}")
    # Use low_memory=False to avoid DtypeWarning about mixed types
    df = pd.read_csv(data_path, low_memory=False)
    
    # Create sliding windows if requested
    if sliding_windows:
        logger.info(f"Creating sliding windows for survival analysis")
        # Use the first outcome column as the target disease
        # Typically this would be 'dated_bpd' or similar
        target_outcome = outcome_columns[0]
        # Get horizons and filter from kwargs
        horizons = kwargs.get('horizons', None)
        filter_bucket = kwargs.get('filter_bucket', None)
        df = create_sliding_windows(df, vital_columns, static_columns, treatment_columns, 
                                   target_outcome, timestamp_column, horizons, filter_horizon=None, filter_bucket=filter_bucket)
        # Update outcome columns for sliding window format
        # Use event_time_bucket and event_indicator instead of one-hot buckets
        outcome_columns = ['event_time_bucket', 'event_indicator']
        outcome_types = ['continuous', 'binary']  # event_time_bucket is treated as continuous, event_indicator is binary
        
        # IMPORTANT: Remove the original disease column from the dataframe to prevent data leakage
        # The disease information is now encoded in event_indicator and event_time_bucket
        if target_outcome in df.columns:
            logger.info(f"Removing original disease column '{target_outcome}' from sliding window data")
            df = df.drop(columns=[target_outcome])
    
    # First sort by person_id and timestamp to ensure proper time ordering
    # Only sort if not from sliding windows (which are already sorted)
    if not sliding_windows:
        df = df.sort_values(['person_id', timestamp_column])
        
        # Print case/control distribution before any processing
        if 'binary' in outcome_types:
            binary_idx = outcome_types.index('binary')
            disease_col = outcome_columns[binary_idx]
            unique_patients = df.drop_duplicates('person_id')
            n_cases = (unique_patients[disease_col] == 1).sum()
            n_controls = (unique_patients[disease_col] == 0).sum()
            n_total = len(unique_patients)
            logger.info(f"\n{'='*60}")
            logger.info(f"Case/Control Distribution for {disease_col}:")
            logger.info(f"  Total patients: {n_total}")
            logger.info(f"  Cases: {n_cases} ({100*n_cases/n_total:.1f}%)")
            logger.info(f"  Controls: {n_controls} ({100*n_controls/n_total:.1f}%)")
            logger.info(f"{'='*60}\n")
    
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
    
    # Apply right censoring for regular (non-sliding-window) data
    # For sliding windows, right censoring is already handled in create_sliding_windows
    if not sliding_windows:
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
            
            # First, screen out patients where disease occurred before first observation
            patients_to_exclude = []
            excluded_count = 0
            for person_id, patient_data in df.groupby('person_id'):
                min_date = patient_data['DateOrdered'].min()
                disease_status = patient_data[disease_column].iloc[0]
                
                if disease_status == 1:
                    disease_date = patient_data[disease_date_col].iloc[0]
                    if pd.notna(disease_date) and disease_date <= min_date:
                        patients_to_exclude.append(person_id)
                        excluded_count += 1
            
            if excluded_count > 0:
                logger.info(f"Excluding {excluded_count} patients where disease occurred before first observation")
                df = df[~df['person_id'].isin(patients_to_exclude)]
            
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
            
            # Create event_indicator and event_time_bucket for non-sliding-window data
            # Get horizons from kwargs - must be provided
            horizons = kwargs.get('horizons', None)
            if horizons is None:
                raise ValueError("horizons must be provided in the config")
            
            # Create event columns
            df['event_indicator'] = 0
            df['event_time_bucket'] = len(horizons)  # Default to last bucket (censored)
            
            # Process each patient
            for person_id, patient_data in df.groupby('person_id'):
                disease_status = patient_data[disease_column].iloc[0]
                
                if disease_status == 1:
                    disease_date = patient_data[disease_date_col].iloc[0]
                    
                    # For each timestep, calculate days until event
                    for idx, row in patient_data.iterrows():
                        current_date = row['DateOrdered']
                        days_to_event = (disease_date - current_date).days
                        
                        # Find appropriate bucket
                        for bucket_idx, h in enumerate(horizons):
                            if len(h) == 2:
                                # Old format: [name, days]
                                horizon_name, horizon_days = h
                                if days_to_event <= horizon_days:
                                    df.loc[idx, 'event_indicator'] = 1
                                    df.loc[idx, 'event_time_bucket'] = bucket_idx
                                    break
                            else:
                                # New format: [name, start, end]
                                name, start_day, end_day = h
                                if start_day <= days_to_event <= end_day:
                                    df.loc[idx, 'event_indicator'] = 1
                                    df.loc[idx, 'event_time_bucket'] = bucket_idx
                                    break
                # Controls remain with event_indicator=0 and event_time_bucket=len(horizons)
            
            # Update outcome columns to match sliding window format
            # Remove original disease column to prevent confusion
            outcome_columns = ['event_time_bucket', 'event_indicator']
            # IMPORTANT: event_time_bucket should be 'ordinal' not 'continuous' to prevent scaling
            outcome_types = ['ordinal', 'binary']
            
            # Remove the original disease column to prevent data leakage
            if disease_column in df.columns:
                logger.info(f"Removing original disease column '{disease_column}' from non-sliding window data")
                df = df.drop(columns=[disease_column])
            
            # Debug: Print statistics
            event_cases = (df['event_indicator'] == 1).sum()
            total_rows = len(df)
            logger.info(f"Added survival outcome columns for non-sliding-window data")
            logger.info(f"Event cases: {event_cases}/{total_rows} ({100*event_cases/total_rows:.1f}%)")
            logger.info(f"Unique event_time_bucket values: {sorted(df['event_time_bucket'].unique())}")
            
            # Debug: Check for any invalid bucket values
            max_bucket = len(horizons)
            invalid_buckets = df[(df['event_time_bucket'] < 0) | (df['event_time_bucket'] > max_bucket)]
            if len(invalid_buckets) > 0:
                logger.warning(f"Found {len(invalid_buckets)} rows with invalid bucket values!")
                logger.warning(f"Invalid bucket values: {invalid_buckets['event_time_bucket'].unique()}")
    
    # Only create timestep and set index if not already done (i.e., not from sliding windows)
    if not isinstance(df.index, pd.MultiIndex):
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
    static_features = df[static_columns].groupby(level='subject_id').first()
    
    # Filter by sequence length
    user_sizes = df.groupby(level='subject_id').size()
    
    
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
        treatments = treatments.groupby(level='subject_id').tail(max_seq_length)
        outcomes = outcomes.groupby(level='subject_id').tail(max_seq_length)
        vitals = vitals.groupby(level='subject_id').tail(max_seq_length) if not vitals.empty else vitals
        
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
        if otype == 'binary' or col == 'event_time_bucket':
            # Keep binary outcomes and event_time_bucket as is (no scaling)
            # event_time_bucket is categorical and should not be scaled
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
    with open(output_dir / 'data_scaler_survival.pkl', 'wb') as f:
        pickle.dump(scaler_info, f)
    logger.info(f"Saved scaler to {output_dir / 'data_scaler_survival.pkl'}")
    
    # For sliding windows, also return the DataFrame with survival info
    if sliding_windows:
        return treatments, outcomes, vitals, static_features, outcomes_unscaled, scaling_params, df
    else:
        return treatments, outcomes, vitals, static_features, outcomes_unscaled, scaling_params


def create_sliding_windows(df, vital_columns, static_columns, treatment_columns, 
                          target_outcome, timestamp_column='DateOrdered', 
                          horizons=None, filter_horizon=None, filter_bucket=None):
    """Create sliding windows for survival analysis with bucket-based hazard prediction
    
    Args:
        df: Original DataFrame with time series data
        vital_columns: List of vital column names
        static_columns: List of static feature column names
        treatment_columns: List of treatment column names
        target_outcome: The outcome column to use (e.g., 'bpd', 'rop', 'nec')
        timestamp_column: Name of timestamp column
        horizons: List of [name, days] or [name, start_day, end_day] for prediction buckets
                  e.g., [["immediate", 3], ["eventual", 500]] or 
                        [["immediate", 1, 3], ["eventual", 3, 500]]
        filter_horizon: Deprecated - use filter_bucket instead
        filter_bucket: If specified, only include windows with events in this bucket name
    
    Returns:
        DataFrame with sliding windows
    """
    from datetime import timedelta
    
    # Validate outcome column
    if target_outcome.startswith('dated_'):
        raise ValueError(f"Outcome column should not start with 'dated_'. Got: {target_outcome}")
    
    disease = target_outcome
    disease_date_col = f'{disease}_date'
    
    # Horizons must be provided
    if horizons is None:
        raise ValueError("horizons must be provided in the config when using sliding windows")
    
    # Convert horizons to bucket format [name, start, end] if needed
    bucket_horizons = []
    for i, h in enumerate(horizons):
        if len(h) == 2:
            # Old format: [name, days] - convert to buckets
            name, days = h
            if i == 0:
                # First bucket: 1 to days
                bucket_horizons.append([name, 1, days])
            else:
                # Subsequent buckets: previous end + 1 to days
                prev_end = bucket_horizons[-1][2]
                bucket_horizons.append([name, prev_end + 1, days])
        else:
            # Already in [name, start, end] format
            bucket_horizons.append(h)
    
    logger.info(f"Using horizon buckets: {bucket_horizons}")
    
    # Store original horizons for bucket index calculation
    original_horizons = bucket_horizons
    
    # Handle filter parameters
    if filter_horizon is not None:
        logger.warning(f"filter_horizon is deprecated - use filter_bucket instead")
    
    if filter_bucket is not None:
        logger.info(f"Filtering to windows with events in bucket: {filter_bucket}")
    
    # Reset index to work with columns
    if isinstance(df.index, pd.MultiIndex):
        df_reset = df.reset_index()
    else:
        df_reset = df.copy()
    
    # Convert dates (try common formats to avoid warnings)
    for date_format in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', None]:
        try:
            df_reset[timestamp_column] = pd.to_datetime(df_reset[timestamp_column], format=date_format)
            break
        except:
            continue
    
    for date_format in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', None]:
        try:
            df_reset[disease_date_col] = pd.to_datetime(df_reset[disease_date_col], format=date_format)
            break
        except:
            continue
    
    # Group by patient
    patient_groups = df_reset.groupby('person_id')
    
    # Print case/control statistics before sliding windows
    logger.info("\nOriginal case/control distribution (before sliding windows):")
    total_patients = len(patient_groups)
    case_patients = df_reset.groupby('person_id')[disease].first().sum()
    control_patients = total_patients - case_patients
    logger.info(f"  Total patients: {total_patients}")
    logger.info(f"  Cases: {case_patients} ({100*case_patients/total_patients:.1f}%)")
    logger.info(f"  Controls: {control_patients} ({100*control_patients/total_patients:.1f}%)")
    
    # Storage for all windows
    all_window_rows = []
    
    # Track filtering statistics
    cases_no_disease_date = 0
    cases_disease_before_obs = 0
    cases_converted_to_control = 0
    valid_cases = 0
    
    # Process each patient
    for person_id, patient_data in patient_groups:
        min_date = patient_data[timestamp_column].min()
        max_date = patient_data[timestamp_column].max()
        disease_status = patient_data[disease].iloc[0]
        
        if disease_status == 1:
            # This is a case
            disease_date = patient_data[disease_date_col].iloc[0]
            
            # Skip if no disease_date
            if pd.isna(disease_date):
                cases_no_disease_date += 1
                continue
                
            # Check exclusion criteria
            if disease_date <= min_date:
                cases_disease_before_obs += 1
                continue
            elif disease_date > max_date + timedelta(weeks=12):
                # Treat as control (right-censored)
                cases_converted_to_control += 1
                anchor_date = max_date
                is_case = False
            else:
                # Valid case
                valid_cases += 1
                anchor_date = disease_date
                is_case = True
        else:
            # This is a control
            anchor_date = max_date
            is_case = False
        
        # Create sliding windows at regular intervals
        # Use a sliding window approach: create windows every 7 days (configurable)
        window_step_days = 7  # Could make this configurable
        
        # Determine the range for creating windows
        # For cases: create windows up to the disease date
        # For controls: create windows up to the last observation
        window_end_date = anchor_date
        
        # Create windows starting from min_date up to window_end_date
        current_anchor = min_date
        window_idx = 0
        
        while current_anchor <= window_end_date:
            # Skip if we don't have enough data before this anchor
            if current_anchor <= min_date:
                current_anchor += timedelta(days=window_step_days)
                continue
            
            # Get all data up to current anchor
            window_data = patient_data[patient_data[timestamp_column] < current_anchor].copy()
            
            if len(window_data) == 0:
                current_anchor += timedelta(days=window_step_days)
                continue
            
            # Create timestep index for this window
            window_data = window_data.sort_values(timestamp_column)
            window_data['timestep'] = range(len(window_data))
            
            # Determine which bucket this window falls into based on time to disease
            if is_case:
                days_to_disease = (disease_date - current_anchor).days
                
                # Find which bucket this falls into
                bucket_idx = None
                for idx, (name, start_day, end_day) in enumerate(bucket_horizons):
                    if start_day <= days_to_disease <= end_day:
                        bucket_idx = idx
                        bucket_name = name
                        break
                
                if bucket_idx is not None:
                    # Event occurs within one of our buckets
                    window_data['event_time_bucket'] = bucket_idx
                    window_data['event_indicator'] = 1
                else:
                    # Event occurs beyond our last bucket (censored)
                    window_data['event_time_bucket'] = len(bucket_horizons)
                    window_data['event_indicator'] = 0
                    bucket_name = 'censored'
            else:
                # Control: always censored
                window_data['event_time_bucket'] = len(bucket_horizons)
                window_data['event_indicator'] = 0
                bucket_name = 'control'
                days_to_disease = -1  # Placeholder for controls
            
            # Add metadata
            window_data['anchor_date'] = current_anchor
            window_data['days_to_disease'] = days_to_disease
            window_data['bucket_name'] = bucket_name
            # Add dummy horizon_days for compatibility (not used in bucket approach)
            window_data['horizon_days'] = 0
            
            # Store a unique window ID
            window_data['window_id'] = f"{person_id}_w{window_idx}"
            window_data['original_person_id'] = person_id
            
            all_window_rows.append(window_data)
            
            # Move to next window
            current_anchor += timedelta(days=window_step_days)
            window_idx += 1
    
    # Combine all windows
    if all_window_rows:
        combined_df = pd.concat(all_window_rows, ignore_index=True)
        
        # Apply bucket filtering if requested
        if filter_bucket is not None:
            pre_filter_count = len(combined_df.groupby('window_id'))
            
            # Find bucket index for the filter
            bucket_idx = None
            for idx, (name, start, end) in enumerate(bucket_horizons):
                if name == filter_bucket:
                    bucket_idx = idx
                    break
            
            if bucket_idx is None:
                raise ValueError(f"filter_bucket '{filter_bucket}' not found in bucket names")
            
            # Filter to windows where events occur in this bucket
            combined_df = combined_df[
                (combined_df['event_indicator'] == 1) & 
                (combined_df['event_time_bucket'] == bucket_idx)
            ]
            
            post_filter_count = len(combined_df.groupby('window_id'))
            logger.info(f"Bucket filtering: {pre_filter_count} windows -> {post_filter_count} windows")
        
        # Use window_id as the subject_id so each window is treated as a separate sequence
        combined_df = combined_df.set_index(['window_id', 'timestep'])
        combined_df.index.names = ['subject_id', 'timestep']  # Rename to match MIMIC-III convention
        combined_df = combined_df.sort_index()
        
        # Log intermediate case filtering statistics
        logger.info("\nCase filtering before sliding windows:")
        logger.info(f"  Original cases: {case_patients}")
        if cases_no_disease_date > 0:
            logger.info(f"  - Cases excluded (no disease date): {cases_no_disease_date}")
        if cases_disease_before_obs > 0:
            logger.info(f"  - Cases excluded (disease before first observation): {cases_disease_before_obs}")
        if cases_converted_to_control > 0:
            logger.info(f"  - Cases converted to controls (disease >12 weeks after last observation): {cases_converted_to_control}")
        logger.info(f"  Remaining valid cases for sliding windows: {valid_cases}")
        
        logger.info(f"\nCreated {len(all_window_rows)} sliding windows from {combined_df.index.get_level_values('subject_id').nunique()} patients")
        
        # Analyze bucket distribution
        logger.info("\nBucket distribution analysis:")
        
        # Get unique values per window
        unique_windows = combined_df.groupby(level='subject_id').first()
        unique_event_indicators = unique_windows['event_indicator'].values
        unique_event_buckets = unique_windows['event_time_bucket'].values
        unique_bucket_names = unique_windows['bucket_name'].values if 'bucket_name' in unique_windows.columns else None
        
        # Separate cases and controls
        case_mask = unique_event_indicators == 1
        control_mask = unique_event_indicators == 0
        
        logger.info(f"  Total windows: {len(unique_event_indicators)}")
        logger.info(f"  Event windows: {case_mask.sum()} ({100*case_mask.sum()/len(unique_event_indicators):.1f}%)")
        logger.info(f"  Censored windows: {control_mask.sum()} ({100*control_mask.sum()/len(unique_event_indicators):.1f}%)")
        
        # Bucket distribution
        logger.info("\n  Bucket distribution:")
        for bucket_idx, (name, start, end) in enumerate(bucket_horizons):
            bucket_count = (unique_event_buckets == bucket_idx).sum()
            bucket_pct = 100 * bucket_count / len(unique_event_buckets) if len(unique_event_buckets) > 0 else 0
            logger.info(f"    Bucket {bucket_idx} ({name}, {start}-{end} days): {bucket_count} windows ({bucket_pct:.1f}%)")
        
        # Censored bucket
        censored_count = (unique_event_buckets == len(bucket_horizons)).sum()
        censored_pct = 100 * censored_count / len(unique_event_buckets) if len(unique_event_buckets) > 0 else 0
        logger.info(f"    Bucket {len(bucket_horizons)} (censored, >{bucket_horizons[-1][2]} days): {censored_count} windows ({censored_pct:.1f}%)")
        
        # Show average number of windows per patient
        windows_per_patient = combined_df.groupby('original_person_id').size()
        logger.info(f"\n  Windows per patient:")
        logger.info(f"    Mean: {windows_per_patient.mean():.1f}")
        logger.info(f"    Min: {windows_per_patient.min()}")
        logger.info(f"    Max: {windows_per_patient.max()}")
        
        logger.info("")  # Empty line for readability
        
        return combined_df
    else:
        raise ValueError("No valid sliding windows created")


class SurvivalDataset(MIMIC3RealDataset):
    """Custom dataset class that handles survival-specific fields"""
    
    def __init__(self, treatments, outcomes, vitals, static_features, outcomes_unscaled, 
                 scaling_params, subset_name, survival_info=None):
        super().__init__(treatments, outcomes, vitals, static_features, outcomes_unscaled,
                        scaling_params, subset_name)
        self.survival_info = survival_info
    
    def __getitem__(self, index) -> dict:
        result = super().__getitem__(index)
        
        # Add survival-specific fields if available
        if self.survival_info is not None:
            for key, values in self.survival_info.items():
                result[key] = values[index]
        
        return result


class SurvivalDatasetCollection(MIMIC3RealDatasetCollection):
    """
    Dataset collection for survival data - inherits from MIMIC3RealDatasetCollection
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
        # These will be updated if using sliding windows
        self.outcome_columns = outcome_columns
        self.outcome_types = outcome_types
        self.min_length_filter = min_length_filter
        self.one_seq_per_patient_eval = one_seq_per_patient_eval
        
        # Get sliding_windows flag from kwargs
        sliding_windows = kwargs.get('sliding_windows', False)
        self.sliding_windows = sliding_windows
        
        # Store all kwargs for later use
        self.hparams = kwargs
        
        # Remove sliding_windows from kwargs if it exists to avoid duplicate
        kwargs_for_load = kwargs.copy()
        kwargs_for_load.pop('sliding_windows', None)
        
        load_result = load_survival_data(
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
            outcome_shift=outcome_shift,
            sliding_windows=sliding_windows,
            **kwargs_for_load  # Pass through horizons and filter_horizon
        )
        
        if sliding_windows:
            treatments, outcomes, vitals, static_features, outcomes_unscaled, scaling_params, df_with_survival = load_result
            # Update outcome columns for sliding windows
            self.outcome_columns = ['event_time_bucket', 'event_indicator']
            self.outcome_types = ['continuous', 'binary']
        else:
            treatments, outcomes, vitals, static_features, outcomes_unscaled, scaling_params = load_result
        
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

        # Extract survival info
        survival_info_train = None
        survival_info_val = None
        survival_info_test = None
        
        if sliding_windows:
            # For sliding windows, we need to track which window each sequence corresponds to
            # Each patient has multiple windows, and each window has different survival info
            
            def extract_survival_info_for_windows(df, window_ids):
                # Now window_ids are the subject_ids in the processed data
                mask = df.index.get_level_values('subject_id').isin(window_ids)
                df_subset = df.loc[mask]
                
                # Create a list to store survival info for each sequence
                # The order must match the order of sequences in treatments/outcomes
                survival_info_list = []
                
                # Process each window directly since window_id is now the subject_id
                for window_id in window_ids:
                    if window_id in df_subset.index.get_level_values('subject_id'):
                        window_data = df_subset.loc[window_id].iloc[0]  # Get first row for this window
                        
                        survival_info_list.append({
                            'event_time_bucket': int(window_data['event_time_bucket']),
                            'event_indicator': int(window_data['event_indicator']),
                            'horizon_days': float(window_data['horizon_days'])
                        })
                
                # Convert to arrays
                n_sequences = len(survival_info_list)
                return {
                    'event_time_bucket': np.array([s['event_time_bucket'] for s in survival_info_list], dtype=np.int64),
                    'event_indicator': np.array([s['event_indicator'] for s in survival_info_list], dtype=np.int64),
                    'horizon_days': np.array([s['horizon_days'] for s in survival_info_list], dtype=np.float32)
                }
            
            survival_info_train = extract_survival_info_for_windows(df_with_survival, static_features_train.index)
            if split['val'] > 0.0:
                survival_info_val = extract_survival_info_for_windows(df_with_survival, static_features_val.index)
            survival_info_test = extract_survival_info_for_windows(df_with_survival, static_features_test.index)
        else:
            # For non-sliding windows, extract event_indicator and event_time_bucket from outcomes
            # These are already in the outcomes DataFrame with proper time dimension
            def extract_survival_info_for_regular(outcomes_df):
                # outcomes_df has columns ['event_time_bucket', 'event_indicator']
                # with MultiIndex (subject_id, timestep)
                return {
                    'event_time_bucket': outcomes_df['event_time_bucket'].values.astype(np.int64),
                    'event_indicator': outcomes_df['event_indicator'].values.astype(np.int64),
                    # For compatibility, add horizon_days as zeros (not used in regular mode)
                    'horizon_days': np.zeros(len(outcomes_df), dtype=np.float32)
                }
            
            survival_info_train = extract_survival_info_for_regular(outcomes_train)
            if split['val'] > 0.0:
                survival_info_val = extract_survival_info_for_regular(outcomes_val)
            survival_info_test = extract_survival_info_for_regular(outcomes_test)
        
        # Create datasets with survival info
        self.train_f = SurvivalDataset(treatments_train, outcomes_train, vitals_train, static_features_train,
                                      outcomes_unscaled_train, scaling_params, 'train', survival_info_train)
        if split['val'] > 0.0:
            self.val_f = SurvivalDataset(treatments_val, outcomes_val, vitals_val, static_features_val, 
                                        outcomes_unscaled_val, scaling_params, 'val', survival_info_val)
        self.test_f = SurvivalDataset(treatments_test, outcomes_test, vitals_test, static_features_test, 
                                     outcomes_unscaled_test, scaling_params, 'test', survival_info_test)
        

        # Save patient splits after all splits are done
        import pickle
        patient_splits = {
            'train': static_features_train.index.tolist(),
            'val': static_features_val.index.tolist() if split['val'] > 0.0 else [],
            'test': static_features_test.index.tolist()
        }
        with open(ROOT_PATH + '/outputs/patient_splits_survival.pkl', 'wb') as f:
            pickle.dump(patient_splits, f)
        logger.info(f"Saved patient splits - Train: {len(patient_splits['train'])}, Val: {len(patient_splits['val'])}, Test: {len(patient_splits['test'])}")

        self.projection_horizon = projection_horizon
        self.has_vitals = len(vital_columns) > 0 if vital_columns else False
        self.autoregressive = autoregressive
        self.processed_data_encoder = True
        
        # Store sliding window and survival flags
        self.sliding_windows = sliding_windows
        self.survival_loss = kwargs.get('survival_loss', False)
        self.mute_decoder = kwargs.get('mute_decoder', False)
        
        # Apply mute_decoder if enabled
        if self.mute_decoder:
            logger.info("Muting decoder inputs (prev_outputs) for all splits")
            for dataset in [self.train_f, self.val_f, self.test_f]:
                if hasattr(dataset, 'data') and 'prev_outputs' in dataset.data:
                    dataset.data['prev_outputs'] = np.zeros_like(dataset.data['prev_outputs'])