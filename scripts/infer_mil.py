"""
MIL Inference Pipeline

Run inference on trained MIL model for batch slide-level prediction.

Usage:
    # Single slide
    python scripts/infer_mil.py \
        --model checkpoints/best_model.pth \
        --features data/features_topk/slide_001.h5 \
        --output results/predictions.csv
    
    # Batch inference
    python scripts/infer_mil.py \
        --model checkpoints/best_model.pth \
        --features data/features_topk \
        --output results/predictions.csv
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
import h5py

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mil import AttentionMIL
from src.utils import setup_logger, load_config, get_device
from src.utils.logger import get_logger

logger = None  # Will be initialized in main()


def load_model(checkpoint_path: str, device: torch.device) -> AttentionMIL:
    """
    Load trained MIL model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        Loaded AttentionMIL model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config from checkpoint or use defaults
    model = AttentionMIL(
        input_dim=2048,
        hidden_dim=512,
        attn_dim=256,
        num_classes=2,
        dropout=0.25
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"✓ Loaded model from {checkpoint_path}")
    logger.info(f"  Epoch: {checkpoint['epoch']}")
    logger.info(f"  Best AUC: {checkpoint.get('best_auc', 'N/A')}")
    
    return model


def load_features_from_hdf5(hdf5_path: str) -> torch.Tensor:
    """
    Load features from HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 file
    
    Returns:
        Feature tensor (K, 2048)
    """
    with h5py.File(hdf5_path, 'r') as f:
        features = f['features'][:]
    
    return torch.from_numpy(features).float()


def get_hdf5_files(features_path: Path):
    """Get list of HDF5 files."""
    if features_path.is_file():
        return [features_path]
    elif features_path.is_dir():
        return sorted(list(features_path.glob('*.h5')) + list(features_path.glob('*.hdf5')))
    else:
        raise ValueError(f"Path does not exist: {features_path}")


def run_inference(
    model: AttentionMIL,
    features_path: str,
    output_csv: str,
    device: torch.device
):
    """
    Run batch inference and save predictions.
    
    Args:
        model: Trained MIL model
        features_path: Path to HDF5 file or directory
        output_csv: Output CSV path
        device: Device
    """
    features_p = Path(features_path)
    hdf5_files = get_hdf5_files(features_p)
    
    logger.info(f"Found {len(hdf5_files)} slides for inference")
    
    results = []
    
    for hdf5_file in tqdm(hdf5_files, desc="Running inference"):
        slide_name = hdf5_file.stem
        
        try:
            # Load features
            features = load_features_from_hdf5(str(hdf5_file)).to(device)
            
            # Run inference
            prediction_dict = model.predict_slide(features, return_attention=False)
            
            # Calculate confidence percentage
            probability = prediction_dict['probability']
            confidence_percent = probability * 100.0
            
            # Interpretation (research-grade, not clinical)
            label = prediction_dict['prediction']
            if label == 1:
                interpretation = f"Malignant ({confidence_percent:.1f}% confidence)"
            else:
                interpretation = f"Benign ({100.0 - confidence_percent:.1f}% confidence)"
            
            results.append({
                'slide_name': slide_name,
                'predicted_label': prediction_dict['prediction'],
                'probability': probability,
                'confidence_percent': confidence_percent,
                'interpretation': interpretation,
                'logit': prediction_dict['logit']
            })
            
            logger.debug(f"{slide_name}: {interpretation}")
        
        except Exception as e:
            logger.error(f"Failed to process {slide_name}: {e}")
            results.append({
                'slide_name': slide_name,
                'predicted_label': -1,
                'probability': -1.0,
                'logit': -999.0
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    logger.info(f"\n✓ Predictions saved to {output_path}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("INFERENCE SUMMARY")
    logger.info("="*60)
    logger.info(f"Total slides: {len(results_df)}")
    logger.info(f"Predicted benign (0): {(results_df['predicted_label'] == 0).sum()}")
    logger.info(f"Predicted malignant (1): {(results_df['predicted_label'] == 1).sum()}")
    logger.info(f"Failed: {(results_df['predicted_label'] == -1).sum()}")
    logger.info(f"Mean probability: {results_df[results_df['probability'] >= 0]['probability'].mean():.4f}")
    logger.info("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run inference on trained MIL model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    
    parser.add_argument(
        '--features',
        type=str,
        required=True,
        help='Path to HDF5 feature file or directory'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results/predictions.csv',
        help='Output CSV path'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    # Setup logging
    global logger
    logger = setup_logger(
        name='mil_inference',
        log_dir=config['paths']['logs'],
        level=config['logging']['level'],
        save_to_file=config['logging']['save_logs']
    )
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Log configuration
    logger.info("="*60)
    logger.info("MIL INFERENCE")
    logger.info("="*60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Features: {args.features}")
    logger.info(f"Output: {args.output}")
    logger.info("="*60)
    
    try:
        # Load model
        model = load_model(args.model, device)
        
        # Run inference
        run_inference(model, args.features, args.output, device)
        
        logger.info("\n✓ Inference complete!")
    
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
