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

from src.models.utils import AlphaRise, FilteringMlFlowLogger
from src.models.time_varying_model_survival import LossBreakdownCallback, GradNormCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.set_default_dtype(torch.double)

# Set specific loggers to higher level to reduce output
logging.getLogger('src.data').setLevel(logging.WARNING)
logging.getLogger('src.models').setLevel(logging.WARNING)


@hydra.main(config_name=f'config.yaml', config_path='../config/')
def main(args: DictConfig):
    """
    Training / evaluation script for CTSurvival (Causal Transformer Survival)
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
    try:
        dim_outcomes_value = args.model.dim_outcomes
        if dim_outcomes_value == '???':
            args.model.dim_outcomes = dataset_collection.train_f.data['outputs'].shape[-1]
    except:
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
        dirpath='outputs/checkpoints_survival',
        filename='best-model-survival-{epoch:02d}-{val_loss:.3f}',
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
        # Load best model weights
        checkpoint = torch.load(checkpoint_callback.best_model_path, weights_only=False)
        # PyTorch Lightning checkpoint contains state_dict as a key
        if 'state_dict' in checkpoint:
            multimodel.load_state_dict(checkpoint['state_dict'])
        else:
            multimodel.load_state_dict(checkpoint)
        logger.info(f"Loaded best model from epoch with val_loss={checkpoint_callback.best_model_score:.4f}")
    
    model_filename = 'trained_model_survival.pt'
    torch.save(multimodel.state_dict(), output_dir / model_filename)
    logger.info(f"Saved trained model to {output_dir / model_filename}")

    # Validation factual evaluation
    val_dataloader = DataLoader(dataset_collection.val_f, batch_size=args.dataset.val_batch_size, shuffle=False)
    val_test_results = multimodel_trainer.validate(multimodel, dataloaders=val_dataloader, ckpt_path='last')
    # multimodel.visualize(dataset_collection.val_f, index=0, artifacts_path=artifacts_path)
    
    # Manually calculate validation metrics (sometimes not displayed in on_validation_epoch_end)
    val_bucket_metrics = multimodel.calculate_bucket_metrics(dataset_collection.val_f, 'val')
    
    val_treatment_pearson_r = multimodel.calculate_treatment_pearson_r(dataset_collection.val_f, 'val')
    logger.info(f"\nFinal Validation Treatment Pearson R (avg across treatments): {val_treatment_pearson_r:.4f}")
    
    # Calculate validation binary metrics
    val_binary_metrics = multimodel.calculate_binary_metrics(dataset_collection.val_f, 'val')
    logger.info(f"\nFinal Validation Binary Classification Metrics (Ever Disease):")
    logger.info(f"  AUROC: {val_binary_metrics.get('val_binary_auc_roc', np.nan):.4f}")
    logger.info(f"  AUPRC: {val_binary_metrics.get('val_binary_auc_pr', np.nan):.4f}")
    logger.info(f"  Cases: {val_binary_metrics.get('val_binary_n_cases', 0)}")
    logger.info(f"  Controls: {val_binary_metrics.get('val_binary_n_controls', 0)}")
    logger.info(f"  Prevalence: {val_binary_metrics.get('val_binary_prevalence', np.nan):.2%}")
    
    logger.info(f"\nFinal Validation Binary + Treatment Metrics (Fourth Head - Overall):")
    logger.info(f"  AUROC: {val_binary_metrics.get('val_binary_treatment_auc_roc', np.nan):.4f}")
    logger.info(f"  AUPRC: {val_binary_metrics.get('val_binary_treatment_auc_pr', np.nan):.4f}")
    logger.info(f"  Cases: {val_binary_metrics.get('val_binary_n_cases', 0)}")
    logger.info(f"  Controls: {val_binary_metrics.get('val_binary_n_controls', 0)}")
    logger.info(f"  Prevalence: {val_binary_metrics.get('val_binary_prevalence', np.nan):.2%}")
    
    # Calculate bucket-specific metrics for fourth head
    val_bucket_specific_metrics = multimodel.calculate_bucket_specific_metrics(dataset_collection.val_f, 'val')
    logger.info(f"\nFinal Validation Binary + Treatment Metrics (Fourth Head - Bucket-Specific):")
    logger.info(f"  Bucket 0 (immediate: 0-1 days):")
    logger.info(f"    AUROC: {val_bucket_specific_metrics.get('val_bucket0_auc_roc', np.nan):.4f}")
    logger.info(f"    AUPRC: {val_bucket_specific_metrics.get('val_bucket0_auc_pr', np.nan):.4f}")
    logger.info(f"    Events: {val_bucket_specific_metrics.get('val_bucket0_n_events', 0)}")
    logger.info(f"    Controls: {val_bucket_specific_metrics.get('val_bucket0_n_controls', 0)}")
    logger.info(f"    Prevalence: {val_bucket_specific_metrics.get('val_bucket0_prevalence', 0):.2%}")
    logger.info(f"  Bucket 1 (eventual: 2+ days):")
    logger.info(f"    AUROC: {val_bucket_specific_metrics.get('val_bucket1_auc_roc', np.nan):.4f}")
    logger.info(f"    AUPRC: {val_bucket_specific_metrics.get('val_bucket1_auc_pr', np.nan):.4f}")
    logger.info(f"    Events: {val_bucket_specific_metrics.get('val_bucket1_n_events', 0)}")
    logger.info(f"    Controls: {val_bucket_specific_metrics.get('val_bucket1_n_controls', 0)}")
    logger.info(f"    At Risk: {val_bucket_specific_metrics.get('val_bucket1_n_at_risk', 0)}")
    logger.info(f"    Excluded: {val_bucket_specific_metrics.get('val_bucket1_n_excluded', 0)}")
    logger.info(f"    Prevalence: {val_bucket_specific_metrics.get('val_bucket1_prevalence', 0):.2%}")
    
    # Display bucket-specific metrics for validation
    if val_bucket_metrics:
        logger.info("\nFinal Validation Metrics by Bucket:")
        logger.info("Bucket         | AUC-ROC | AUC-PR | Prevalence | Events/No-Events")
        logger.info("---------------|---------|--------|------------|------------------")
        
        for bucket_name in sorted(val_bucket_metrics.keys()):
            metrics = val_bucket_metrics[bucket_name]
            logger.info(f"{bucket_name:<14} | {metrics['auc_roc']:>7.4f} | {metrics['auc_pr']:>6.4f} | "
                      f"{metrics['prevalence']:>9.1f}% | {metrics['n_events']:>3}/{metrics['n_controls']:<3}")
    encoder_results = {}

    # Debug dataset splits
    logger.info(f"\nDataset splits:")
    logger.info(f"  Train size: {len(dataset_collection.train_f) if hasattr(dataset_collection, 'train_f') else 'N/A'}")
    logger.info(f"  Val size: {len(dataset_collection.val_f) if hasattr(dataset_collection, 'val_f') else 'N/A'}")
    logger.info(f"  Test exists: {hasattr(dataset_collection, 'test_f')}")
    logger.info(f"  Test is None: {getattr(dataset_collection, 'test_f', 'not found') is None}")
    if hasattr(dataset_collection, 'test_f') and dataset_collection.test_f is not None:
        logger.info(f"  Test size: {len(dataset_collection.test_f)}")
    
    # Test set evaluation
    if hasattr(dataset_collection, 'test_f') and dataset_collection.test_f is not None:
        logger.info("\n" + "="*60)
        logger.info("Starting test set evaluation with best model...")
        logger.info("="*60)
        
        # Create test dataloader
        test_dataloader = DataLoader(dataset_collection.test_f, batch_size=args.dataset.val_batch_size, shuffle=False)
        
        # Run test evaluation - this will trigger test_step
        test_results = multimodel_trainer.test(multimodel, dataloaders=test_dataloader)
        
        # Manually calculate and display test metrics
        # (PyTorch Lightning doesn't always call on_test_epoch_end properly)
        logger.info("\nCalculating test set metrics...")
        
        # Calculate bucket metrics
        test_bucket_metrics = multimodel.calculate_bucket_metrics(dataset_collection.test_f, 'test')
        
        # Calculate patient-level bucket metrics (comparable to binary)
        test_patient_bucket_metrics = multimodel.calculate_patient_level_bucket_metrics(dataset_collection.test_f, 'test')
        
        # Calculate treatment Pearson R
        test_treatment_pearson_r = multimodel.calculate_treatment_pearson_r(dataset_collection.test_f, 'test')
        logger.info(f"\nTest Treatment Pearson R (avg across treatments): {test_treatment_pearson_r:.4f}")
        
        # Calculate binary classification metrics
        test_binary_metrics = multimodel.calculate_binary_metrics(dataset_collection.test_f, 'test')
        logger.info(f"\nTest Binary Classification Metrics (Ever Disease):")
        logger.info(f"  AUROC: {test_binary_metrics.get('test_binary_auc_roc', np.nan):.4f}")
        logger.info(f"  AUPRC: {test_binary_metrics.get('test_binary_auc_pr', np.nan):.4f}")
        logger.info(f"  Cases: {test_binary_metrics.get('test_binary_n_cases', 0)}")
        logger.info(f"  Controls: {test_binary_metrics.get('test_binary_n_controls', 0)}")
        logger.info(f"  Prevalence: {test_binary_metrics.get('test_binary_prevalence', np.nan):.2%}")
        
        logger.info(f"\nTest Binary + Treatment Metrics (Fourth Head - Overall):")
        logger.info(f"  AUROC: {test_binary_metrics.get('test_binary_treatment_auc_roc', np.nan):.4f}")
        logger.info(f"  AUPRC: {test_binary_metrics.get('test_binary_treatment_auc_pr', np.nan):.4f}")
        logger.info(f"  Cases: {test_binary_metrics.get('test_binary_n_cases', 0)}")
        logger.info(f"  Controls: {test_binary_metrics.get('test_binary_n_controls', 0)}")
        logger.info(f"  Prevalence: {test_binary_metrics.get('test_binary_prevalence', np.nan):.2%}")
        
        # Calculate bucket-specific metrics for fourth head
        test_bucket_specific_metrics = multimodel.calculate_bucket_specific_metrics(dataset_collection.test_f, 'test')
        logger.info(f"\nTest Binary + Treatment Metrics (Fourth Head - Bucket-Specific):")
        logger.info(f"  Bucket 0 (immediate: 0-1 days):")
        logger.info(f"    AUROC: {test_bucket_specific_metrics.get('test_bucket0_auc_roc', np.nan):.4f}")
        logger.info(f"    AUPRC: {test_bucket_specific_metrics.get('test_bucket0_auc_pr', np.nan):.4f}")
        logger.info(f"    Events: {test_bucket_specific_metrics.get('test_bucket0_n_events', 0)}")
        logger.info(f"    Controls: {test_bucket_specific_metrics.get('test_bucket0_n_controls', 0)}")
        logger.info(f"    Prevalence: {test_bucket_specific_metrics.get('test_bucket0_prevalence', 0):.2%}")
        logger.info(f"  Bucket 1 (eventual: 2+ days):")
        logger.info(f"    AUROC: {test_bucket_specific_metrics.get('test_bucket1_auc_roc', np.nan):.4f}")
        logger.info(f"    AUPRC: {test_bucket_specific_metrics.get('test_bucket1_auc_pr', np.nan):.4f}")
        logger.info(f"    Events: {test_bucket_specific_metrics.get('test_bucket1_n_events', 0)}")
        logger.info(f"    Controls: {test_bucket_specific_metrics.get('test_bucket1_n_controls', 0)}")
        logger.info(f"    At Risk: {test_bucket_specific_metrics.get('test_bucket1_n_at_risk', 0)}")
        logger.info(f"    Excluded: {test_bucket_specific_metrics.get('test_bucket1_n_excluded', 0)}")
        logger.info(f"    Prevalence: {test_bucket_specific_metrics.get('test_bucket1_prevalence', 0):.2%}")
        
        # Display bucket-specific metrics
        if test_bucket_metrics:
            logger.info("\nTest Metrics by Bucket:")
            logger.info("Bucket         | AUC-ROC | AUC-PR | Prevalence | Events/No-Events")
            logger.info("---------------|---------|--------|------------|------------------")
            
            for bucket_name in sorted(test_bucket_metrics.keys()):
                metrics = test_bucket_metrics[bucket_name]
                logger.info(f"{bucket_name:<14} | {metrics['auc_roc']:>7.4f} | {metrics['auc_pr']:>6.4f} | "
                          f"{metrics['prevalence']:>9.1f}% | {metrics['n_events']:>3}/{metrics['n_controls']:<3}")
        
        # Calculate horizon metrics
        test_horizon_metrics = multimodel.calculate_horizon_metrics(dataset_collection.test_f, 'test')
        
        if test_horizon_metrics:
            logger.info("\nTest Metrics by Horizon:")
            logger.info("Horizon | AUC-ROC | AUC-PR | Prevalence | Events/Controls")
            logger.info("--------|---------|--------|------------|----------------")
            
            for horizon in sorted(test_horizon_metrics.keys()):
                metrics = test_horizon_metrics[horizon]
                logger.info(f"{horizon:>6}d | {metrics['auc_roc']:>7.4f} | {metrics['auc_pr']:>6.4f} | "
                          f"{metrics['prevalence']:>9.1f}% | {metrics['n_events']:>3}/{metrics['n_controls']:<3}")
        
        logger.info("\nTest evaluation completed.")
    else:
        logger.info("No test set available for evaluation.")

    mlf_logger.log_metrics(encoder_results) if args.exp.logging else None
    results.update(encoder_results)

    mlf_logger.experiment.set_terminated(mlf_logger.run_id) if args.exp.logging else None

    return results


if __name__ == "__main__":
    main()

