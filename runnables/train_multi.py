import logging
import hydra
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint

from src import ROOT_PATH
from src.models.utils import AlphaRise, FilteringMlFlowLogger, load_checkpoint_with_ema
from src.models.time_varying_model import LossBreakdownCallback, GradNormCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)

# Set specific loggers to higher level to reduce output
logging.getLogger('src.data').setLevel(logging.WARNING)
logging.getLogger('src.models').setLevel(logging.WARNING)
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)


@hydra.main(version_base="1.1", config_name=f'config.yaml', config_path='../config/')
def main(args: DictConfig):
    """
    Training / evaluation script for CT (Causal Transformer)
    Args:
        args: arguments of run as DictConfig

    Returns: dict with results (one and nultiple-step-ahead RMSEs)
    """

    results = {}

    # Non-strict access to fields
    OmegaConf.set_struct(args, False)
    OmegaConf.register_new_resolver("sum", lambda x, y: x + y, replace=True)
    
    # Set dimension values based on column lists before resolving config
    if hasattr(args.dataset, 'treatment_columns'):
        args.model.dim_treatments = len(args.dataset.treatment_columns)
    if hasattr(args.dataset, 'vital_columns'):
        args.model.dim_vitals = len(args.dataset.vital_columns)
    if hasattr(args.dataset, 'static_columns'):
        args.model.dim_static_features = len(args.dataset.static_columns)
    
    # Comment out verbose config logging
    # logger.info('\n' + OmegaConf.to_yaml(args, resolve=True))
    
    # Initialisation of data
    seed_everything(args.exp.seed)
    dataset_collection = instantiate(args.dataset, _recursive_=True)
    dataset_collection.process_data_multi()
    
    # Mute decoder inputs if enabled
    if hasattr(args.dataset, 'mute_decoder') and args.dataset.mute_decoder:
        logger.info("Muting decoder inputs (autoregressive outputs) during training")
        # Zero out all autoregressive outputs in training data - use copy to avoid affecting original outcomes
        dataset_collection.train_f.data['prev_outputs'] = np.zeros_like(dataset_collection.train_f.data['prev_outputs'])
        # Zero out in validation data
        dataset_collection.val_f.data['prev_outputs'] = np.zeros_like(dataset_collection.val_f.data['prev_outputs'])
        # Zero out in test data if it exists
        if hasattr(dataset_collection, 'test_f'):
            dataset_collection.test_f.data['prev_outputs'] = np.zeros_like(dataset_collection.test_f.data['prev_outputs'])
    # Set dim_outcomes - only override if not already set in config
    if not hasattr(args.model, 'dim_outcomes') or args.model.dim_outcomes == '???':
        args.model.dim_outcomes = dataset_collection.train_f.data['outputs'].shape[-1]
    # Otherwise keep the value from config (which could be a list)
    # Double-check dimensions match the actual data
    args.model.dim_treatments = dataset_collection.train_f.data['current_treatments'].shape[-1]
    args.model.dim_vitals = dataset_collection.train_f.data['vitals'].shape[-1] if dataset_collection.has_vitals else 0
    args.model.dim_static_features = dataset_collection.train_f.data['static_features'].shape[-1]

    # Train_callbacks
    multimodel_callbacks = [AlphaRise(rate=args.exp.alpha_rate), LossBreakdownCallback()]
    
    # Add model checkpoint callback to save best model
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='outputs/checkpoints',
        filename='best-model-{epoch:02d}-{val_loss:.3f}',
        save_top_k=1,
        mode='min',
        save_weights_only=True
    )
    multimodel_callbacks.append(checkpoint_callback)
    
    # Add early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,  # Minimum change of 0.001 to qualify as an improvement
        patience=5,       # Number of epochs with no improvement after which training will be stopped
        verbose=True,
        mode='min'        # For loss, we want to minimize
    )
    multimodel_callbacks.append(early_stopping)
    
    # Add GradNorm callback if enabled
    if hasattr(args.dataset, 'use_gradnorm') and args.dataset.use_gradnorm:
        gradnorm_alpha = getattr(args.dataset, 'gradnorm_alpha', 1.5)
        multimodel_callbacks.append(GradNormCallback(alpha=gradnorm_alpha))

    # MlFlow Logger
    if args.exp.logging:
        experiment_name = f'{args.model.name}/{args.dataset.name}'
        mlf_logger = FilteringMlFlowLogger(filter_submodels=[], experiment_name=experiment_name, tracking_uri=args.exp.mlflow_uri)
        multimodel_callbacks += [LearningRateMonitor(logging_interval='epoch')]
        artifacts_path = hydra.utils.to_absolute_path(mlf_logger.experiment.get_run(mlf_logger.run_id).info.artifact_uri)
    else:
        mlf_logger = None
        artifacts_path = None

    # ============================== Initialisation & Training of multimodel ==============================
    multimodel = instantiate(args.model.multi, args, dataset_collection, _recursive_=False)
    # Debug: Model correctly has 2 buckets
    # print(multimodel.br_treatment_outcome_head.num_buckets)  # outputs: 2
    # print(multimodel.br_treatment_outcome_head.linear7.out_features)  # outputs: 2
    if args.model.multi.tune_hparams:
        multimodel.finetune(resources_per_trial=args.model.multi.resources_per_trial)

    multimodel_trainer = Trainer(gpus=eval(str(args.exp.gpus)), logger=mlf_logger, max_epochs=args.exp.max_epochs,
                                 callbacks=multimodel_callbacks, terminate_on_nan=True,
                                 gradient_clip_val=args.model.multi.max_grad_norm)
    multimodel_trainer.fit(multimodel)
    
    # Save the trained model
    import os
    from pathlib import Path
    root_dir = Path(__file__).parent.parent
    output_dir = root_dir / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    # Check if we have a best model from checkpoint
    if checkpoint_callback.best_model_path:
        # Load best model weights with EMA if available
        checkpoint = load_checkpoint_with_ema(multimodel, checkpoint_callback.best_model_path, use_ema=True)
        logger.info(f"Loaded best model from epoch with val_loss={checkpoint_callback.best_model_score:.4f}")
    
    torch.save(multimodel.state_dict(), output_dir / 'trained_model.pt')
    logger.info(f"Saved trained model to {output_dir / 'trained_model.pt'}")

    # Validation factual evaluation
    val_dataloader = DataLoader(dataset_collection.val_f, batch_size=args.dataset.val_batch_size, shuffle=False, num_workers=4)
    val_test_results = multimodel_trainer.validate(multimodel, dataloaders=val_dataloader, ckpt_path='last')
    # multimodel.visualize(dataset_collection.val_f, index=0, artifacts_path=artifacts_path)
    
    # Calculate metrics based on outcome types
    val_metrics = {}

    # Single outcome - original behavior
    val_auc_roc, val_auc_pr, _, _ = multimodel.get_binary_classification_metrics(dataset_collection.val_f)
    val_auc_roc_bucket, val_auc_pr_bucket = multimodel.get_bucket_classification_metrics(dataset_collection.val_f)
    logger.info(f'Val AUC-ROC Binary: {val_auc_roc}; Val AUC-PR Binary: {val_auc_pr}')
    # logger.info(f'Val AUC-ROC Bucket: {val_auc_roc_bucket}; Val AUC-PR Bucket: {val_auc_pr_bucket}')
    val_rmse_orig, val_rmse_all = None, None
    
    encoder_results = {}
    if hasattr(dataset_collection, 'outcome_types'):
        # Multiple outcomes - add all val metrics with encoder prefix
        for key, value in val_metrics.items():
            encoder_results[f'encoder_{key}'] = value
        
        # Test set evaluation for multiple outcomes
        if hasattr(dataset_collection, 'test_f'):
            # Run trainer.validate on test set
            test_dataloader = DataLoader(dataset_collection.test_f, batch_size=args.dataset.val_batch_size, shuffle=False, num_workers=4)
            test_results = multimodel_trainer.validate(multimodel, dataloaders=test_dataloader)
            
            # Calculate metrics for each outcome
            for i, (outcome_name, otype) in enumerate(zip(dataset_collection.outcome_columns, dataset_collection.outcome_types)):
                test_auc_roc, test_auc_pr, n_cases, n_controls = multimodel.get_binary_classification_metrics(dataset_collection.test_f)
                test_auc_roc_bucket, test_auc_pr_bucket = multimodel.get_bucket_classification_metrics(dataset_collection.test_f)
                
                # Get the ground truth used for AUC calculation
                test_data = dataset_collection.test_f.data
                seq_lengths = test_data['sequence_lengths'] if isinstance(test_data['sequence_lengths'], np.ndarray) else test_data['sequence_lengths'].numpy()
                
                # Use the counts from get_binary_classification_metrics which includes landmark filtering
                logger.info(f'Test {outcome_name}: {n_cases} cases, {n_controls} controls (from {n_cases + n_controls} patients)')
                logger.info(f'Minimum sequence length in test data: {seq_lengths.min()}')
                logger.info(f'Test {outcome_name} AUC-ROC Binary: {test_auc_roc}; AUC-PR Binary: {test_auc_pr}')
                logger.info(f'Test {outcome_name} AUC-ROC Bucket: {test_auc_roc_bucket}; AUC-PR Bucket: {test_auc_pr_bucket}')
                
                # Get landmarks if available
                landmarks = getattr(dataset_collection, 'landmarks', None)
                
                if landmarks:
                    # Store all landmark results
                    landmark_results = {}
                    
                    # Collect all metrics for CSV
                    all_metrics_rows = []
                    
                    # Print per-landmark tables (including full sequence)
                    for landmark in landmarks + [max(landmarks) + 1]:
                        landmark_label = "Full Sequence" if landmark == max(landmarks) + 1 else f"Landmark {landmark}"
                        
                        # Get binary classification metrics for this landmark
                        binary_auc_roc, binary_auc_pr, binary_n_cases, binary_n_controls = multimodel.get_binary_classification_metrics(
                            dataset_collection.test_f, landmark=landmark)
                        
                        # Add binary metrics to CSV collection
                        if binary_n_cases + binary_n_controls > 0:
                            binary_prevalence = binary_n_cases / (binary_n_cases + binary_n_controls)
                            all_metrics_rows.append({
                                'outcome': outcome_name,
                                'landmark': landmark_label,
                                'bucket': 'binary',
                                'horizon': 'overall',
                                'n_cases': binary_n_cases,
                                'n_controls': binary_n_controls,
                                'prevalence': binary_prevalence,
                                'auc_roc': binary_auc_roc,
                                'auc_pr': binary_auc_pr
                            })
                        
                        # Get per-bucket metrics for this landmark
                        per_bucket_results = multimodel.get_bucket_classification_metrics(
                            dataset_collection.test_f, per_bucket=True, landmark=landmark)
                        test_auc_roc_per_bucket, test_auc_pr_per_bucket, n_cases_per_bucket, n_controls_per_bucket = per_bucket_results
                        
                        # Store results
                        landmark_results[landmark] = {
                            'auc_roc': test_auc_roc_per_bucket,
                            'auc_pr': test_auc_pr_per_bucket,
                            'n_cases': n_cases_per_bucket,
                            'n_controls': n_controls_per_bucket
                        }
                        
                        if test_auc_roc_per_bucket and test_auc_pr_per_bucket:
                            logger.info(f"\n{landmark_label}")
                            logger.info("-" * 110)
                            logger.info(f"{'Bucket':<10} {'Horizon':<25} {'Cases':<10} {'Controls':<10} {'Prevalence':<12} {'AUC-ROC':<10} {'AUC-PR':<10}")
                            logger.info("-" * 110)
                            
                            # Get horizon definitions if available
                            horizons = getattr(dataset_collection, 'horizons', None)
                            
                            for b in range(len(test_auc_roc_per_bucket)):
                                horizon_str = ""
                                if horizons and b < len(horizons):
                                    _, min_days, max_days = horizons[b]
                                    horizon_str = f"[{min_days}, {max_days}] days"
                                
                                n_cases = n_cases_per_bucket[b]
                                n_controls = n_controls_per_bucket[b]
                                prevalence = n_cases / (n_cases + n_controls) if (n_cases + n_controls) > 0 else 0
                                auc_roc_b = test_auc_roc_per_bucket[b]
                                auc_pr_b = test_auc_pr_per_bucket[b]
                                
                                logger.info(f"{b:<10} {horizon_str:<25} {n_cases:<10} {n_controls:<10} {prevalence:<12.3f} {auc_roc_b:<10.4f} {auc_pr_b:<10.4f}")
                                
                                # Collect row for CSV
                                all_metrics_rows.append({
                                    'outcome': outcome_name,
                                    'landmark': landmark_label,
                                    'bucket': b,
                                    'horizon': horizon_str,
                                    'n_cases': n_cases,
                                    'n_controls': n_controls,
                                    'prevalence': prevalence,
                                    'auc_roc': auc_roc_b,
                                    'auc_pr': auc_pr_b
                                })
                            
                            logger.info("-" * 110)
                    
                    # Save landmark results to file for later analysis
                    import json
                    output_dir = Path(ROOT_PATH) / 'outputs'
                    with open(output_dir / f'landmark_results_{outcome_name}.json', 'w') as f:
                        # Convert numpy values to native Python types for JSON serialization
                        json_safe_results = {}
                        for lm, results in landmark_results.items():
                            json_safe_results[str(lm)] = {
                                'auc_roc': [float(x) if not np.isnan(x) else None for x in results['auc_roc']],
                                'auc_pr': [float(x) if not np.isnan(x) else None for x in results['auc_pr']],
                                'n_cases': [int(x) for x in results['n_cases']],
                                'n_controls': [int(x) for x in results['n_controls']]
                            }
                        json.dump(json_safe_results, f, indent=2)
                    logger.info(f"Saved landmark results to {output_dir / f'landmark_results_{outcome_name}.json'}")
                    
                    # Save all collected metrics to CSV
                    if all_metrics_rows:
                        import pandas as pd
                        df = pd.DataFrame(all_metrics_rows)
                        csv_path = output_dir / 'results.csv'
                        df.to_csv(csv_path, index=False)
                        logger.info(f"Saved all metrics to {csv_path}")
                else:
                    # No landmarks - get overall metrics
                    per_bucket_results = multimodel.get_bucket_classification_metrics(
                        dataset_collection.test_f, per_bucket=True)
                    test_auc_roc_per_bucket, test_auc_pr_per_bucket, n_cases_per_bucket, n_controls_per_bucket = per_bucket_results
                    
                    if test_auc_roc_per_bucket and test_auc_pr_per_bucket:
                        logger.info(f"\nPer-Bucket Metrics for {outcome_name}:")
                        logger.info("-" * 110)
                        logger.info(f"{'Bucket':<10} {'Horizon':<25} {'Cases':<10} {'Controls':<10} {'Prevalence':<12} {'AUC-ROC':<10} {'AUC-PR':<10}")
                        logger.info("-" * 110)
                        
                        # Get horizon definitions if available
                        horizons = getattr(dataset_collection, 'horizons', None)
                        
                        for b in range(len(test_auc_roc_per_bucket)):
                            horizon_str = ""
                            if horizons and b < len(horizons):
                                _, min_days, max_days = horizons[b]
                                horizon_str = f"[{min_days}, {max_days}] days"
                            
                            n_cases = n_cases_per_bucket[b]
                            n_controls = n_controls_per_bucket[b]
                            prevalence = n_cases / (n_cases + n_controls) if (n_cases + n_controls) > 0 else 0
                            auc_roc_b = test_auc_roc_per_bucket[b]
                            auc_pr_b = test_auc_pr_per_bucket[b]
                            
                            logger.info(f"{b:<10} {horizon_str:<25} {n_cases:<10} {n_controls:<10} {prevalence:<12.3f} {auc_roc_b:<10.4f} {auc_pr_b:<10.4f}")
                        
                        logger.info("-" * 110 + "\n")
                
                # Load patient splits to get actual person IDs
                import pickle
                from pathlib import Path
                # Use ROOT_PATH to ensure we find the file regardless of working directory
                splits_path = Path(ROOT_PATH) / 'outputs' / 'patient_splits.pkl'
                if splits_path.exists():
                    with open(splits_path, 'rb') as f:
                        patient_splits = pickle.load(f)
                    test_person_ids = patient_splits['test']
                else:
                    logger.warning(f"Could not find patient splits file at {splits_path}")
                    test_person_ids = list(range(len(y_true_last)))  # Use indices as fallback
                
                
                encoder_results.update({
                    f'encoder_test_{outcome_name}_auc_roc': test_auc_roc,
                    f'encoder_test_{outcome_name}_auc_pr': test_auc_pr
                })
    else:
        # Continuous outcomes - use RMSE
        if hasattr(dataset_collection, 'test_cf_one_step'):  # Test one_step_counterfactual rmse
            test_rmse_orig, test_rmse_all, test_rmse_last = multimodel.get_normalised_masked_rmse(dataset_collection.test_cf_one_step,
                                                                                                  one_step_counterfactual=True)
            logger.info(f'Test normalised RMSE (all): {test_rmse_all}; '
                        f'Test normalised RMSE (orig): {test_rmse_orig}; '
                        f'Test normalised RMSE (only counterfactual): {test_rmse_last}')
            encoder_results = {
                'encoder_val_rmse_all': val_rmse_all,
                'encoder_val_rmse_orig': val_rmse_orig,
                'encoder_test_rmse_all': test_rmse_all,
                'encoder_test_rmse_orig': test_rmse_orig,
                'encoder_test_rmse_last': test_rmse_last
            }
        elif hasattr(dataset_collection, 'test_f'):  # Test factual rmse
            test_rmse_orig, test_rmse_all = multimodel.get_normalised_masked_rmse(dataset_collection.test_f)
            logger.info(f'Test normalised RMSE (all): {test_rmse_all}; '
                        f'Test normalised RMSE (orig): {test_rmse_orig}.')
            encoder_results = {
                'encoder_val_rmse_all': val_rmse_all,
                'encoder_val_rmse_orig': val_rmse_orig,
                'encoder_test_rmse_all': test_rmse_all,
                'encoder_test_rmse_orig': test_rmse_orig
            }

    mlf_logger.log_metrics(encoder_results) if args.exp.logging else None
    results.update(encoder_results)

    test_rmses = {}
    test_pearson_rs = {}
    test_auc_rocs = {}
    test_auc_prs = {}
    
    if hasattr(dataset_collection, 'outcome_types'):
        # Multiple outcomes - n-step prediction
        # For now, we'll focus on the single-step predictions done above
        # Full multi-step prediction for multiple outcomes would require extensive changes
        logger.info("Multi-step prediction for multiple outcomes not yet fully implemented")
    elif hasattr(dataset_collection, 'outcome_type') and dataset_collection.outcome_type == 'binary':
        # Binary classification metrics for n-step prediction
        # Check if we should use one sequence per patient evaluation
        use_one_seq_per_patient = getattr(dataset_collection, 'one_seq_per_patient_eval', False)
        
        if hasattr(dataset_collection, 'test_cf_treatment_seq'):
            if use_one_seq_per_patient:
                result = multimodel.evaluate_one_seq_per_patient_binary(
                    dataset_collection.test_f, projection_horizon=dataset_collection.projection_horizon, 
                    use_ground_truth_feedback=True)
                # Check if result is a tuple (teacher forcing) or lists (multi-step)
                if isinstance(result[0], (float, np.float64)):
                    # Single values from teacher forcing - wrap in lists
                    auc_rocs = [result[0]]
                    auc_prs = [result[1]]
                else:
                    # Lists from multi-step prediction
                    auc_rocs, auc_prs = result
            else:
                auc_rocs, auc_prs = multimodel.get_binary_n_step_classification_metrics(dataset_collection.test_cf_treatment_seq)
            # Same logic for this branch
            if dataset_collection.projection_horizon == 0:
                test_auc_rocs = {f'{k}-step': v for (k, v) in enumerate(auc_rocs)}
                test_auc_prs = {f'{k}-step': v for (k, v) in enumerate(auc_prs)}
            else:
                test_auc_rocs = {f'{k+1}-step': v for (k, v) in enumerate(auc_rocs)}
                test_auc_prs = {f'{k+1}-step': v for (k, v) in enumerate(auc_prs)}
        elif hasattr(dataset_collection, 'test_f_multi'):
            if use_one_seq_per_patient:
                result = multimodel.evaluate_one_seq_per_patient_binary(
                    dataset_collection.test_f, projection_horizon=dataset_collection.projection_horizon, 
                    use_ground_truth_feedback=True)
                # Check if result is a tuple (teacher forcing) or lists (multi-step)
                if isinstance(result[0], (float, np.float64)):
                    # Single values from teacher forcing - wrap in lists
                    auc_rocs = [result[0]]
                    auc_prs = [result[1]]
                else:
                    # Lists from multi-step prediction
                    auc_rocs, auc_prs = result
            else:
                auc_rocs, auc_prs = multimodel.get_binary_n_step_classification_metrics(dataset_collection.test_f_multi)
            # Handle both 0-step and n-step cases
            # If projection_horizon is 0, we have 0-step and 1-step
            # Otherwise, we have 1-step through (projection_horizon+1)-step
            if dataset_collection.projection_horizon == 0:
                test_auc_rocs = {f'{k}-step': v for (k, v) in enumerate(auc_rocs)}
                test_auc_prs = {f'{k}-step': v for (k, v) in enumerate(auc_prs)}
            else:
                test_auc_rocs = {f'{k+1}-step': v for (k, v) in enumerate(auc_rocs)}
                test_auc_prs = {f'{k+1}-step': v for (k, v) in enumerate(auc_prs)}
        
        logger.info(f'Test AUC-ROC (n-step prediction): {test_auc_rocs}')
        logger.info(f'Test AUC-PR (n-step prediction): {test_auc_prs}')
    else:
        # RMSE metrics for continuous outcomes
        if hasattr(dataset_collection, 'test_cf_treatment_seq'):  # Test n_step_counterfactual rmse
            rmses, pearson_rs = multimodel.get_normalised_n_step_rmses(dataset_collection.test_cf_treatment_seq)
            test_rmses = rmses
            test_pearson_rs = pearson_rs
        elif hasattr(dataset_collection, 'test_f_multi'):  # Test n_step_factual rmse
            rmses, pearson_rs = multimodel.get_normalised_n_step_rmses(dataset_collection.test_f_multi)
            test_rmses = rmses
            test_pearson_rs = pearson_rs
        test_rmses = {f'{k+2}-step': v for (k, v) in enumerate(test_rmses)}
        test_pearson_rs = {f'{k+2}-step': v for (k, v) in enumerate(test_pearson_rs)}

        logger.info(f'Test normalised RMSE (n-step prediction): {test_rmses}')
        logger.info(f'Test Pearson correlation (n-step prediction): {test_pearson_rs}')
    # Initialize decoder results based on outcome types
    decoder_results = {}
    
    # Only add RMSE metrics if they exist (for continuous outcomes)
    if 'val_rmse_all' in locals() and val_rmse_all is not None:
        decoder_results['decoder_val_rmse_all'] = val_rmse_all
        decoder_results['decoder_val_rmse_orig'] = val_rmse_orig
    # Add test metrics based on what was calculated
    if test_auc_rocs:  # Binary outcomes
        decoder_results.update({('decoder_test_auc_roc_' + k): v for (k, v) in test_auc_rocs.items()})
        decoder_results.update({('decoder_test_auc_pr_' + k): v for (k, v) in test_auc_prs.items()})
    if test_rmses:  # Continuous outcomes
        decoder_results.update({('decoder_test_rmse_' + k): v for (k, v) in test_rmses.items()})
        decoder_results.update({('decoder_test_pearson_r_' + k): v for (k, v) in test_pearson_rs.items()})

    mlf_logger.log_metrics(decoder_results) if args.exp.logging else None
    results.update(decoder_results)

    mlf_logger.experiment.set_terminated(mlf_logger.run_id) if args.exp.logging else None

    return results


if __name__ == "__main__":
    main()

