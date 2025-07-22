import pandas as pd
from pandas.core.algorithms import isin
import numpy as np
import torch
from copy import deepcopy
import logging

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from copy import deepcopy

from src import ROOT_PATH
from src.data.dataset_collection import RealDatasetCollection
from src.data.mimic_iii.load_data import load_mimic3_data_processed

logger = logging.getLogger(__name__)


class MIMIC3RealDataset(Dataset):
    """
    Pytorch-style real-world MIMIC-III dataset
    """
    def __init__(self,
                 treatments: pd.DataFrame,
                 outcomes: pd.DataFrame,
                 vitals: pd.DataFrame,
                 outcomes_bucket: pd.DataFrame,
                 static_features: pd.DataFrame,
                 scaling_params: dict,
                 subset_name: str):
        """
        Args:
            treatments: DataFrame with treatments; multiindex by (patient_id, timestep)
            outcomes: DataFrame with outcomes; multiindex by (patient_id, timestep)
            vitals: DataFrame with vitals (time-varying covariates); multiindex by (patient_id, timestep)
            static_features: DataFrame with static features
            scaling_params: Standard normalization scaling parameters
            subset_name: train / val / test
        """
        assert treatments.shape[0] == outcomes.shape[0]
        assert outcomes.shape[0] == vitals.shape[0]

        self.subset_name = subset_name
        user_sizes = vitals.groupby('subject_id').size()

        # Padding with nans
        # Use future_stack=True to avoid FutureWarning about deprecated stack behavior
        treatments = treatments.unstack(fill_value=np.nan, level=0).stack(future_stack=True).swaplevel(0, 1).sort_index()
        outcomes = outcomes.unstack(fill_value=np.nan, level=0).stack(future_stack=True).swaplevel(0, 1).sort_index()
        vitals = vitals.unstack(fill_value=np.nan, level=0).stack(future_stack=True).swaplevel(0, 1).sort_index()
        outcomes_bucket = outcomes_bucket.unstack(fill_value=np.nan, level=0).stack(future_stack=True).swaplevel(0, 1).sort_index()
        active_entries = (~treatments.isna().any(axis=1)).astype(float)
        static_features = static_features.sort_index()
        user_sizes = user_sizes.sort_index()

        # Conversion to np.arrays
        treatments = treatments.fillna(0.0).values.reshape((len(user_sizes), max(user_sizes), -1)).astype(float)
        outcomes = outcomes.fillna(0.0).values.reshape((len(user_sizes), max(user_sizes), -1))
        vitals = vitals.fillna(0.0).values.reshape((len(user_sizes), max(user_sizes), -1))
        outcomes_bucket = outcomes_bucket.fillna(0.0).values.reshape((len(user_sizes), max(user_sizes), -1))
        active_entries = active_entries.values.reshape((len(user_sizes), max(user_sizes), 1))
        static_features = static_features.values
        user_sizes = user_sizes.values

        self.data = {
            'sequence_lengths': user_sizes - 1,
            'prev_treatments': treatments[:, :-1, :],
            'vitals': vitals[:, 1:, :],
            'next_vitals': vitals[:, 2:, :],
            'current_treatments': treatments[:, 1:, :],
            'static_features': static_features,
            'active_entries': active_entries[:, 1:, :],
            'outputs': outcomes[:, 1:, :],
            'outputs_bucket': outcomes_bucket[:, 1:, :],
            'prev_outputs': outcomes[:, :-1, :].copy(),  # Make a copy to avoid aliasing issues
            'prev_outputs_bucket': outcomes_bucket[:, :-1, :].copy()
        }

        self.scaling_params = scaling_params
        self.processed = True
        self.processed_sequential = False
        self.processed_autoregressive = False
        self.exploded = False

        data_shapes = {k: v.shape for k, v in self.data.items()}
        logger.info(f'Shape of processed {self.subset_name} data: {data_shapes}')

        self.norm_const = 1.0

    def __getitem__(self, index) -> dict:
        result = {k: v[index] for k, v in self.data.items()}
        if hasattr(self, 'encoder_r'):
            if 'original_index' in self.data:
                result.update({'encoder_r': self.encoder_r[int(result['original_index'])]})
            else:
                result.update({'encoder_r': self.encoder_r[index]})
        return result

    def __len__(self):
        return len(self.data['active_entries'])

    def create_one_seq_per_patient_for_n_step(self, projection_horizon):
        """
        Create dataset where each patient contributes exactly ONE sequence per n-step prediction.
        For n-step prediction: use first (seq_len - n) timesteps to predict last n timesteps.
        
        Args:
            projection_horizon: Maximum number of steps to predict (will create 1 through projection_horizon+1 step predictions)
        """
        logger.info(f'Creating one sequence per patient for n-step predictions on {self.subset_name}')
        
        # Get original data
        outputs = self.data['outputs']
        outputs_bucket = self.data['outputs_bucket']
        prev_outputs = self.data['prev_outputs']
        prev_outputs_bucket = self.data['prev_outputs_bucket']
        sequence_lengths = self.data['sequence_lengths']
        vitals = self.data['vitals']
        next_vitals = self.data['next_vitals']
        active_entries = self.data['active_entries']
        current_treatments = self.data['current_treatments']
        previous_treatments = self.data['prev_treatments']
        static_features = self.data['static_features']
        
        # Filter to only include patients with enough timesteps for all predictions
        min_required_length = projection_horizon + 1 + 1  # Need at least 1 history timestep
        valid_patients = sequence_lengths >= min_required_length
        valid_indices = np.where(valid_patients)[0]
        
        logger.info(f'Using {len(valid_indices)} patients with at least {min_required_length} timesteps')
        
        # Create a dict to store data for each n-step
        n_step_datasets = {}
        
        # For each n-step prediction (0-step through projection_horizon+1-step)
        # 0-step means use full sequence (no truncation)
        for n_step in range(0, projection_horizon + 2):
            # Arrays for this n-step
            n_patients = len(valid_indices)
            max_seq_length = outputs.shape[1]
            
            seq_outputs = np.zeros((n_patients, max_seq_length, outputs.shape[-1]))
            seq_outputs_bucket = np.zeros((n_patients, max_seq_length, outputs_bucket.shape[-1]))
            seq_prev_outputs = np.zeros((n_patients, max_seq_length, prev_outputs.shape[-1]))
            seq_prev_outputs_bucket = np.zeros((n_patients, max_seq_length, prev_outputs_bucket.shape[-1]))
            seq_vitals = np.zeros((n_patients, max_seq_length, vitals.shape[-1]))
            seq_next_vitals = np.zeros((n_patients, max_seq_length - 1, next_vitals.shape[-1]))
            seq_active_entries = np.zeros((n_patients, max_seq_length, active_entries.shape[-1]))
            seq_current_treatments = np.zeros((n_patients, max_seq_length, current_treatments.shape[-1]))
            seq_previous_treatments = np.zeros((n_patients, max_seq_length, previous_treatments.shape[-1]))
            seq_static_features = np.zeros((n_patients, static_features.shape[-1]))
            seq_sequence_lengths = np.zeros(n_patients)
            
            # For each valid patient
            for idx, patient_idx in enumerate(valid_indices):
                seq_len = int(sequence_lengths[patient_idx])
                
                if n_step == 0:
                    # Special case: 0-step means use the full sequence (no truncation)
                    history_length = seq_len
                else:
                    history_length = seq_len - n_step
                
                if history_length >= 1:  # Need at least 1 timestep of history
                    # Copy the history portion (for 0-step, this is the full sequence)
                    seq_outputs[idx, :history_length, :] = outputs[patient_idx, :history_length, :]
                    seq_prev_outputs[idx, :history_length, :] = prev_outputs[patient_idx, :history_length, :]
                    seq_outputs_bucket[idx, :history_length, :] = outputs_bucket[patient_idx, :history_length, :]
                    seq_prev_outputs_bucket[idx, :history_length, :] = prev_outputs_bucket[patient_idx, :history_length, :]
                    seq_vitals[idx, :history_length, :] = vitals[patient_idx, :history_length, :]
                    seq_next_vitals[idx, :min(history_length, seq_len-1), :] = next_vitals[patient_idx, :min(history_length, seq_len-1), :]
                    seq_active_entries[idx, :history_length, :] = active_entries[patient_idx, :history_length, :]
                    seq_current_treatments[idx, :history_length, :] = current_treatments[patient_idx, :history_length, :]
                    seq_previous_treatments[idx, :history_length, :] = previous_treatments[patient_idx, :history_length, :]
                    seq_static_features[idx] = static_features[patient_idx]
                    seq_sequence_lengths[idx] = history_length
                    
            # Store this n-step dataset with additional metadata
            n_step_data = {
                'outputs': seq_outputs,
                'prev_outputs': seq_prev_outputs,
                'outputs_bucket': seq_outputs_bucket,
                'prev_outputs_bucket': seq_prev_outputs_bucket,
                'vitals': seq_vitals,
                'next_vitals': seq_next_vitals,
                'active_entries': seq_active_entries,
                'current_treatments': seq_current_treatments,
                'prev_treatments': seq_previous_treatments,
                'static_features': seq_static_features,
                'sequence_lengths': seq_sequence_lengths,
                'n_step': n_step,  # Track which n-step this is for
                'true_future_outputs': outputs[valid_indices],  # Keep full sequences for evaluation
                'original_patient_idx': valid_indices,  # Track original patient indices
                'history_lengths': seq_sequence_lengths  # Store history lengths for verification
            }
            
            n_step_datasets[n_step] = n_step_data
            if seq_sequence_lengths[seq_sequence_lengths > 0].size > 0:
                if n_step == 0:
                    logger.info(f'{n_step}-step (full sequence): {n_patients} patients, sequence lengths: {seq_sequence_lengths[seq_sequence_lengths > 0].min():.0f}-{seq_sequence_lengths.max():.0f}')
                else:
                    logger.info(f'{n_step}-step: {n_patients} patients, history lengths: {seq_sequence_lengths[seq_sequence_lengths > 0].min():.0f}-{seq_sequence_lengths.max():.0f}')
        
        # Store all n-step datasets
        self.n_step_datasets = n_step_datasets
        self.valid_patient_indices = valid_indices
        
        return n_step_datasets
    
    def explode_trajectories(self, projection_horizon, min_length_filter=False):
        """
        Convert test dataset to a dataset with rolling origin
        Args:
            projection_horizon: projection horizon
            min_length_filter: If True, only include patients with enough timesteps for all multi-step predictions
        """
        assert self.processed
        
        # Save the truly original data before explosion
        if not hasattr(self, 'data_before_explosion'):
            self.data_before_explosion = deepcopy(self.data)

        logger.info(f'Exploding {self.subset_name} dataset before testing (multiple sequences)')

        outputs = self.data['outputs']
        prev_outputs = self.data['prev_outputs']
        outputs_bucket = self.data['outputs_bucket']
        prev_outputs_bucket = self.data['prev_outputs_bucket']
        sequence_lengths = self.data['sequence_lengths']
        vitals = self.data['vitals']
        next_vitals = self.data['next_vitals']
        active_entries = self.data['active_entries']
        current_treatments = self.data['current_treatments']
        previous_treatments = self.data['prev_treatments']
        static_features = self.data['static_features']
        if 'stabilized_weights' in self.data:
            stabilized_weights = self.data['stabilized_weights']

        num_patients, max_seq_length, num_features = outputs.shape
        
        # Filter patients if requested
        if min_length_filter:
            min_required_length = projection_horizon + 1 + 1  # e.g., for projection_horizon=5, need at least 7 timesteps
            valid_patients = sequence_lengths >= min_required_length
            logger.info(f'Filtering to patients with at least {min_required_length} timesteps: {np.sum(valid_patients)} out of {num_patients} patients')
            
            # Apply filter
            outputs = outputs[valid_patients]
            prev_outputs = prev_outputs[valid_patients]
            outputs_bucket = outputs_bucket[valid_patients]
            prev_outputs_bucket = prev_outputs_bucket[valid_patients]
            sequence_lengths = sequence_lengths[valid_patients]
            vitals = vitals[valid_patients] if vitals.shape[0] > 0 else vitals
            next_vitals = next_vitals[valid_patients] if next_vitals.shape[0] > 0 else next_vitals
            active_entries = active_entries[valid_patients]
            current_treatments = current_treatments[valid_patients]
            previous_treatments = previous_treatments[valid_patients]
            static_features = static_features[valid_patients]
            if 'stabilized_weights' in self.data:
                stabilized_weights = stabilized_weights[valid_patients]
            
            num_patients = outputs.shape[0]
            
        num_seq2seq_rows = num_patients * max_seq_length

        seq2seq_previous_treatments = np.zeros((num_seq2seq_rows, max_seq_length, previous_treatments.shape[-1]))
        seq2seq_current_treatments = np.zeros((num_seq2seq_rows, max_seq_length, current_treatments.shape[-1]))
        seq2seq_static_features = np.zeros((num_seq2seq_rows, static_features.shape[-1]))
        seq2seq_outputs = np.zeros((num_seq2seq_rows, max_seq_length, outputs.shape[-1]))
        seq2seq_prev_outputs = np.zeros((num_seq2seq_rows, max_seq_length, prev_outputs.shape[-1]))
        seq2seq_outputs_bucket = np.zeros((num_seq2seq_rows, max_seq_length, outputs_bucket.shape[-1]))
        seq2seq_prev_outputs_bucket = np.zeros((num_seq2seq_rows, max_seq_length, prev_outputs_bucket.shape[-1]))
        seq2seq_vitals = np.zeros((num_seq2seq_rows, max_seq_length, vitals.shape[-1]))
        seq2seq_next_vitals = np.zeros((num_seq2seq_rows, max_seq_length - 1, next_vitals.shape[-1]))
        seq2seq_active_entries = np.zeros((num_seq2seq_rows, max_seq_length, active_entries.shape[-1]))
        seq2seq_sequence_lengths = np.zeros(num_seq2seq_rows)
        if 'stabilized_weights' in self.data:
            seq2seq_stabilized_weights = np.zeros((num_seq2seq_rows, max_seq_length))

        total_seq2seq_rows = 0  # we use this to shorten any trajectories later

        for i in range(num_patients):
            sequence_length = int(sequence_lengths[i])

            for t in range(projection_horizon, sequence_length):  # shift outputs back by 1
                seq2seq_active_entries[total_seq2seq_rows, :(t + 1), :] = active_entries[i, :(t + 1), :]
                if 'stabilized_weights' in self.data:
                    seq2seq_stabilized_weights[total_seq2seq_rows, :(t + 1)] = stabilized_weights[i, :(t + 1)]
                seq2seq_previous_treatments[total_seq2seq_rows, :(t + 1), :] = previous_treatments[i, :(t + 1), :]
                seq2seq_current_treatments[total_seq2seq_rows, :(t + 1), :] = current_treatments[i, :(t + 1), :]
                seq2seq_outputs[total_seq2seq_rows, :(t + 1), :] = outputs[i, :(t + 1), :]
                seq2seq_prev_outputs[total_seq2seq_rows, :(t + 1), :] = prev_outputs[i, :(t + 1), :]
                seq2seq_outputs_bucket[total_seq2seq_rows, :(t + 1), :] = outputs_bucket[i, :(t + 1), :]
                seq2seq_prev_outputs_bucket[total_seq2seq_rows, :(t + 1), :] = prev_outputs_bucket[i, :(t + 1), :]
                seq2seq_vitals[total_seq2seq_rows, :(t + 1), :] = vitals[i, :(t + 1), :]
                seq2seq_next_vitals[total_seq2seq_rows, :min(t + 1, sequence_length - 1), :] = \
                    next_vitals[i, :min(t + 1, sequence_length - 1), :]
                seq2seq_sequence_lengths[total_seq2seq_rows] = t + 1
                seq2seq_static_features[total_seq2seq_rows] = static_features[i]

                total_seq2seq_rows += 1

        # Filter everything shorter
        seq2seq_previous_treatments = seq2seq_previous_treatments[:total_seq2seq_rows, :, :]
        seq2seq_current_treatments = seq2seq_current_treatments[:total_seq2seq_rows, :, :]
        seq2seq_static_features = seq2seq_static_features[:total_seq2seq_rows, :]
        seq2seq_outputs = seq2seq_outputs[:total_seq2seq_rows, :, :]
        seq2seq_prev_outputs = seq2seq_prev_outputs[:total_seq2seq_rows, :, :]
        seq2seq_outputs_bucket = seq2seq_outputs_bucket[:total_seq2seq_rows, :, :]
        seq2seq_prev_outputs_bucket = seq2seq_prev_outputs_bucket[:total_seq2seq_rows, :, :]
        seq2seq_vitals = seq2seq_vitals[:total_seq2seq_rows, :, :]
        seq2seq_next_vitals = seq2seq_next_vitals[:total_seq2seq_rows, :, :]
        seq2seq_active_entries = seq2seq_active_entries[:total_seq2seq_rows, :, :]
        seq2seq_sequence_lengths = seq2seq_sequence_lengths[:total_seq2seq_rows]

        if 'stabilized_weights' in self.data:
            seq2seq_stabilized_weights = seq2seq_stabilized_weights[:total_seq2seq_rows]

        new_data = {
            'prev_treatments': seq2seq_previous_treatments,
            'current_treatments': seq2seq_current_treatments,
            'static_features': seq2seq_static_features,
            'prev_outputs': seq2seq_prev_outputs,
            'outputs': seq2seq_outputs,
            'prev_outputs_bucket': seq2seq_prev_outputs_bucket,
            'outputs_bucket': seq2seq_outputs_bucket,
            'vitals': seq2seq_vitals,
            'next_vitals': seq2seq_next_vitals,
            'sequence_lengths': seq2seq_sequence_lengths,
            'active_entries': seq2seq_active_entries,
        }
        if 'stabilized_weights' in self.data:
            new_data['stabilized_weights'] = seq2seq_stabilized_weights

        self.data = new_data

        data_shapes = {k: v.shape for k, v in self.data.items()}
        logger.info(f'Shape of processed {self.subset_name} data: {data_shapes}')

    def process_sequential(self, encoder_r, projection_horizon, encoder_outputs=None, save_encoder_r=False):
        """
        Pre-process dataset for multiple-step-ahead prediction: explodes dataset to a larger one with rolling origin
        Args:
            encoder_r: Representations of encoder
            projection_horizon: Projection horizon
            encoder_outputs: One-step-ahead predcitions of encoder
            save_encoder_r: Save all encoder representations (for cross-attention of EDCT)
        """

        assert self.processed

        if not self.processed_sequential:
            logger.info(f'Processing {self.subset_name} dataset before training (multiple sequences)')

            outputs = self.data['outputs']
            prev_outputs = self.data['prev_outputs']
            outputs_bucket = self.data['outputs_bucket']
            prev_outputs_bucket = self.data['prev_outputs_bucket']
            sequence_lengths = self.data['sequence_lengths']
            active_entries = self.data['active_entries']
            current_treatments = self.data['current_treatments']
            previous_treatments = self.data['prev_treatments']
            static_features = self.data['static_features']
            stabilized_weights = self.data['stabilized_weights'] if 'stabilized_weights' in self.data else None

            num_patients, max_seq_length, num_features = outputs.shape

            num_seq2seq_rows = num_patients * max_seq_length

            seq2seq_state_inits = np.zeros((num_seq2seq_rows, encoder_r.shape[-1]))
            seq2seq_active_encoder_r = np.zeros((num_seq2seq_rows, max_seq_length))
            seq2seq_original_index = np.zeros((num_seq2seq_rows, ))
            seq2seq_previous_treatments = np.zeros((num_seq2seq_rows, projection_horizon, previous_treatments.shape[-1]))
            seq2seq_current_treatments = np.zeros((num_seq2seq_rows, projection_horizon, current_treatments.shape[-1]))
            seq2seq_static_features = np.zeros((num_seq2seq_rows, static_features.shape[-1]))
            seq2seq_outputs = np.zeros((num_seq2seq_rows, projection_horizon, outputs.shape[-1]))
            seq2seq_prev_outputs = np.zeros((num_seq2seq_rows, projection_horizon, prev_outputs.shape[-1]))
            seq2seq_outputs_bucket = np.zeros((num_seq2seq_rows, projection_horizon, outputs_bucket.shape[-1]))
            seq2seq_prev_outputs_bucket = np.zeros((num_seq2seq_rows, projection_horizon, prev_outputs_bucket.shape[-1]))
            seq2seq_active_entries = np.zeros((num_seq2seq_rows, projection_horizon, active_entries.shape[-1]))
            seq2seq_sequence_lengths = np.zeros(num_seq2seq_rows)
            seq2seq_stabilized_weights = np.zeros((num_seq2seq_rows, projection_horizon + 1)) \
                if stabilized_weights is not None else None

            total_seq2seq_rows = 0  # we use this to shorten any trajectories later

            for i in range(num_patients):

                sequence_length = int(sequence_lengths[i])

                for t in range(1, sequence_length - projection_horizon):  # shift outputs back by 1
                    seq2seq_state_inits[total_seq2seq_rows, :] = encoder_r[i, t - 1, :]  # previous state output
                    seq2seq_original_index[total_seq2seq_rows] = i
                    seq2seq_active_encoder_r[total_seq2seq_rows, :t] = 1.0

                    max_projection = min(projection_horizon, sequence_length - t)

                    seq2seq_active_entries[total_seq2seq_rows, :max_projection, :] = active_entries[i, t:t + max_projection, :]
                    seq2seq_previous_treatments[total_seq2seq_rows, :max_projection, :] = \
                        previous_treatments[i, t:t + max_projection, :]
                    seq2seq_current_treatments[total_seq2seq_rows, :max_projection, :] = \
                        current_treatments[i, t:t + max_projection, :]
                    seq2seq_outputs[total_seq2seq_rows, :max_projection, :] = outputs[i, t:t + max_projection, :]
                    seq2seq_outputs_bucket[total_seq2seq_rows, :max_projection, :] = outputs_bucket[i, t:t + max_projection, :]
                    seq2seq_sequence_lengths[total_seq2seq_rows] = max_projection
                    seq2seq_static_features[total_seq2seq_rows] = static_features[i]
                    if encoder_outputs is not None:  # For auto-regressive evaluation
                        seq2seq_prev_outputs[total_seq2seq_rows, :max_projection, :] = \
                            encoder_outputs[i, t - 1:t + max_projection - 1, :]
                        seq2seq_prev_outputs_bucket[total_seq2seq_rows, :max_projection, :] = \
                            encoder_outputs[i, t - 1:t + max_projection - 1, :]
                    else:  # train / val of decoder
                        seq2seq_prev_outputs[total_seq2seq_rows, :max_projection, :] = prev_outputs[i, t:t + max_projection, :]
                        seq2seq_prev_outputs_bucket[total_seq2seq_rows, :max_projection, :] = prev_outputs_bucket[i, t:t + max_projection, :]

                    if seq2seq_stabilized_weights is not None:  # Also including SW of one-step-ahead prediction
                        seq2seq_stabilized_weights[total_seq2seq_rows, :] = stabilized_weights[i, t - 1:t + max_projection]

                    total_seq2seq_rows += 1

            # Filter everything shorter
            seq2seq_state_inits = seq2seq_state_inits[:total_seq2seq_rows, :]
            seq2seq_original_index = seq2seq_original_index[:total_seq2seq_rows]
            seq2seq_active_encoder_r = seq2seq_active_encoder_r[:total_seq2seq_rows, :]
            seq2seq_previous_treatments = seq2seq_previous_treatments[:total_seq2seq_rows, :, :]
            seq2seq_current_treatments = seq2seq_current_treatments[:total_seq2seq_rows, :, :]
            seq2seq_static_features = seq2seq_static_features[:total_seq2seq_rows, :]
            seq2seq_outputs = seq2seq_outputs[:total_seq2seq_rows, :, :]
            seq2seq_outputs_bucket = seq2seq_outputs_bucket[:total_seq2seq_rows, :, :]
            seq2seq_prev_outputs = seq2seq_prev_outputs[:total_seq2seq_rows, :, :]
            seq2seq_prev_outputs_bucket = seq2seq_prev_outputs_bucket[:total_seq2seq_rows, :, :]
            seq2seq_active_entries = seq2seq_active_entries[:total_seq2seq_rows, :, :]
            seq2seq_sequence_lengths = seq2seq_sequence_lengths[:total_seq2seq_rows]

            if seq2seq_stabilized_weights is not None:
                seq2seq_stabilized_weights = seq2seq_stabilized_weights[:total_seq2seq_rows]

            # Package outputs
            seq2seq_data = {
                'init_state': seq2seq_state_inits,
                'original_index': seq2seq_original_index,
                'active_encoder_r': seq2seq_active_encoder_r,
                'prev_treatments': seq2seq_previous_treatments,
                'current_treatments': seq2seq_current_treatments,
                'static_features': seq2seq_static_features,
                'prev_outputs': seq2seq_prev_outputs,
                'prev_outputs_bucket': seq2seq_prev_outputs_bucket,
                'outputs': seq2seq_outputs,
                'outputs_bucket': seq2seq_outputs_bucket,
                'sequence_lengths': seq2seq_sequence_lengths,
                'active_entries': seq2seq_active_entries,
            }
            if seq2seq_stabilized_weights is not None:
                seq2seq_data['stabilized_weights'] = seq2seq_stabilized_weights

            self.data_original = deepcopy(self.data)
            self.data_processed_seq = deepcopy(seq2seq_data)  # For auto-regressive evaluation (self.data will be changed)
            self.data = seq2seq_data

            data_shapes = {k: v.shape for k, v in self.data.items()}
            logger.info(f'Shape of processed {self.subset_name} data: {data_shapes}')

            if save_encoder_r:
                self.encoder_r = encoder_r[:, :max_seq_length, :]

            self.processed_sequential = True
            self.exploded = True

        else:
            logger.info(f'{self.subset_name} Dataset already processed (multiple sequences)')

        return self.data

    def process_sequential_test(self, projection_horizon, encoder_r=None, save_encoder_r=False):
        """
        Pre-process test dataset for multiple-step-ahead prediction: takes the last n-steps according to the projection horizon
        Args:
            projection_horizon: Projection horizon
            encoder_r: Representations of encoder
            save_encoder_r: Save all encoder representations (for cross-attention of EDCT)
        """

        assert self.processed

        if not self.processed_sequential:
            logger.info(f'Processing {self.subset_name} dataset before testing (multiple sequences)')

            outputs = self.data['outputs']
            prev_outputs = self.data['prev_outputs']
            sequence_lengths = self.data['sequence_lengths']
            current_treatments = self.data['current_treatments']
            previous_treatments = self.data['prev_treatments']
            # vitals = self.data['vitals']

            num_patient_points, max_seq_length, num_features = outputs.shape

            if encoder_r is not None:
                seq2seq_state_inits = np.zeros((num_patient_points, encoder_r.shape[-1]))
            seq2seq_active_encoder_r = np.zeros((num_patient_points, max_seq_length - projection_horizon))
            seq2seq_previous_treatments = np.zeros((num_patient_points, projection_horizon, previous_treatments.shape[-1]))
            seq2seq_current_treatments = np.zeros((num_patient_points, projection_horizon, current_treatments.shape[-1]))
            seq2seq_outputs = np.zeros((num_patient_points, projection_horizon, outputs.shape[-1]))
            seq2seq_prev_outputs = np.zeros((num_patient_points, projection_horizon, outputs.shape[-1]))
            seq2seq_active_entries = np.zeros((num_patient_points, projection_horizon, 1))
            seq2seq_sequence_lengths = np.zeros(num_patient_points)
            seq2seq_original_index = np.zeros(num_patient_points)

            for i in range(num_patient_points):
                fact_length = int(sequence_lengths[i]) - projection_horizon
                if encoder_r is not None:
                    seq2seq_state_inits[i] = encoder_r[i, fact_length - 1]
                seq2seq_active_encoder_r[i, :fact_length] = 1.0
                seq2seq_original_index[i] = i

                seq2seq_active_entries[i] = np.ones(shape=(projection_horizon, 1))
                seq2seq_previous_treatments[i] = previous_treatments[i, fact_length:fact_length + projection_horizon, :]
                seq2seq_current_treatments[i] = current_treatments[i, fact_length:fact_length + projection_horizon, :]
                seq2seq_outputs[i] = outputs[i, fact_length: fact_length + projection_horizon, :]
                seq2seq_prev_outputs[i] = prev_outputs[i, fact_length: fact_length + projection_horizon, :]

                seq2seq_sequence_lengths[i] = projection_horizon

            # Package outputs
            seq2seq_data = {
                'original_index': seq2seq_original_index,
                'active_encoder_r': seq2seq_active_encoder_r,
                'prev_treatments': seq2seq_previous_treatments,
                'current_treatments': seq2seq_current_treatments,
                'static_features': self.data['static_features'],
                'prev_outputs': seq2seq_prev_outputs,
                'outputs': seq2seq_outputs,
                'sequence_lengths': seq2seq_sequence_lengths,
                'active_entries': seq2seq_active_entries,
            }
            if encoder_r is not None:
                seq2seq_data['init_state'] = seq2seq_state_inits

            self.data_original = deepcopy(self.data)
            self.data_processed_seq = deepcopy(seq2seq_data)  # For auto-regressive evaluation (self.data will be changed)
            self.data = seq2seq_data

            data_shapes = {k: v.shape for k, v in self.data.items()}
            logger.info(f'Shape of processed {self.subset_name} data: {data_shapes}')

            if save_encoder_r and encoder_r is not None:
                self.encoder_r = encoder_r[:, :max_seq_length - projection_horizon, :]

            self.processed_sequential = True

        else:
            logger.info(f'{self.subset_name} Dataset already processed (multiple sequences)')

        return self.data

    def process_autoregressive_test(self, encoder_r, encoder_outputs, projection_horizon, save_encoder_r=False):
        """
        Pre-process test dataset for multiple-step-ahead prediction: axillary dataset placeholder for autoregressive prediction
        Args:
            projection_horizon: Projection horizon
            encoder_r: Representations of encoder
            save_encoder_r: Save all encoder representations (for cross-attention of EDCT)
        """

        assert self.processed_sequential

        if not self.processed_autoregressive:
            logger.info(f'Processing {self.subset_name} dataset before testing (autoregressive)')

            current_treatments = self.data_original['current_treatments']
            prev_treatments = self.data_original['prev_treatments']

            sequence_lengths = self.data_original['sequence_lengths']
            num_patient_points = current_treatments.shape[0]

            current_dataset = dict()  # Same as original, but only with last n-steps
            current_dataset['prev_treatments'] = np.zeros((num_patient_points, projection_horizon,
                                                           self.data_original['prev_treatments'].shape[-1]))
            current_dataset['current_treatments'] = np.zeros((num_patient_points, projection_horizon,
                                                              self.data_original['current_treatments'].shape[-1]))
            current_dataset['prev_outputs'] = np.zeros((num_patient_points, projection_horizon,
                                                        self.data_original['outputs'].shape[-1]))
            current_dataset['init_state'] = np.zeros((num_patient_points, encoder_r.shape[-1]))
            current_dataset['active_encoder_r'] = np.zeros((num_patient_points, int(sequence_lengths.max() - projection_horizon)))
            current_dataset['active_entries'] = np.ones((num_patient_points, projection_horizon, 1))

            for i in range(num_patient_points):
                fact_length = int(sequence_lengths[i]) - projection_horizon
                current_dataset['init_state'][i] = encoder_r[i, fact_length - 1]
                current_dataset['prev_outputs'][i, 0, :] = encoder_outputs[i, fact_length - 1]
                current_dataset['active_encoder_r'][i, :fact_length] = 1.0

                current_dataset['prev_treatments'][i] = \
                    prev_treatments[i, fact_length - 1:fact_length + projection_horizon - 1, :]
                current_dataset['current_treatments'][i] = current_treatments[i, fact_length:fact_length + projection_horizon, :]

            current_dataset['static_features'] = self.data_original['static_features']

            self.data_processed_seq = deepcopy(self.data)
            self.data = current_dataset
            data_shapes = {k: v.shape for k, v in self.data.items()}
            logger.info(f'Shape of processed {self.subset_name} data: {data_shapes}')

            if save_encoder_r:
                self.encoder_r = encoder_r[:, :int(max(sequence_lengths) - projection_horizon), :]

            self.processed_autoregressive = True

        else:
            logger.info(f'{self.subset_name} Dataset already processed (autoregressive)')

        return self.data

    def process_sequential_multi(self, projection_horizon):
        """
        Pre-process test dataset for multiple-step-ahead prediction for multi-input model: marking rolling origin with
            'future_past_split'
        Args:
            projection_horizon: Projection horizon
        """

        assert self.processed_sequential

        if not self.processed_autoregressive:
            self.data_processed_seq = self.data
            self.data = deepcopy(self.data_original)
            self.data['future_past_split'] = self.data['sequence_lengths'] - projection_horizon
            self.processed_autoregressive = True

        else:
            logger.info(f'{self.subset_name} Dataset already processed (autoregressive)')

        return self.data


class MIMIC3RealDatasetCollection(RealDatasetCollection):
    """
    Dataset collection (train_f, val_f, test_f)
    """
    def __init__(self,
                 path: str,
                 min_seq_length: int = 30,
                 max_seq_length: int = 60,
                 seed: int = 100,
                 max_number: int = None,
                 split: dict = {'val': 0.2, 'test': 0.2},
                 projection_horizon: int = 5,
                 autoregressive=True,
                 **kwargs):
        """
        Args:
            path: Path with MIMIC-3 dataset (HDFStore)
            min_seq_length: Min sequence lenght in cohort
            max_seq_length: Max sequence lenght in cohort
            seed: Seed for random cohort patient selection
            max_number: Maximum number of patients in cohort
            split: Ratio of train / val / test split
            projection_horizon: Range of tau-step-ahead prediction (tau = projection_horizon + 1)
            autoregressive:
        """
        super(MIMIC3RealDatasetCollection, self).__init__()
        self.seed = seed
        treatments, outcomes, vitals, static_features, scaling_params = \
            load_mimic3_data_processed(ROOT_PATH + '/' + path, min_seq_length=min_seq_length, max_seq_length=max_seq_length,
                                       max_number=max_number, data_seed=seed, **kwargs)

        # Train/val/test random_split
        static_features, static_features_test = train_test_split(static_features, test_size=split['test'], random_state=seed)
        treatments, outcomes, vitals, treatments_test, outcomes_test, vitals_test = \
            treatments.loc[static_features.index], \
            outcomes.loc[static_features.index], \
            vitals.loc[static_features.index], \
            treatments.loc[static_features_test.index], \
            outcomes.loc[static_features_test.index], \
            vitals.loc[static_features_test.index]

        if split['val'] > 0.0:
            static_features_train, static_features_val = train_test_split(static_features,
                                                                          test_size=split['val'] / (1 - split['test']),
                                                                          random_state=2 * seed)
            treatments_train, outcomes_train, vitals_train, treatments_val, outcomes_val, vitals_val = \
                treatments.loc[static_features_train.index], \
                outcomes.loc[static_features_train.index], \
                vitals.loc[static_features_train.index], \
                treatments.loc[static_features_val.index], \
                outcomes.loc[static_features_val.index], \
                vitals.loc[static_features_val.index]
        else:
            static_features_train = static_features
            treatments_train, outcomes_train, vitals_train = treatments, outcomes, vitals

        self.train_f = MIMIC3RealDataset(treatments_train, outcomes_train, vitals_train, static_features_train, scaling_params, 'train')
        if split['val'] > 0.0:
            self.val_f = MIMIC3RealDataset(treatments_val, outcomes_val, vitals_val, static_features_val, scaling_params, 'val')
        self.test_f = MIMIC3RealDataset(treatments_test, outcomes_test, vitals_test, static_features_test, scaling_params, 'test')

        self.projection_horizon = projection_horizon
        self.has_vitals = True
        self.autoregressive = autoregressive
        self.processed_data_encoder = True
