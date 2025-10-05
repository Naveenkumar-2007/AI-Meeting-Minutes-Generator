"""
Training script for fine-tuning the summarization model
This is optional - you can use pre-trained models without training
"""
import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline.training_pipeline import SummarizerTrainingPipeline


def main():
    parser = argparse.ArgumentParser(description='Train summarization model')
    parser.add_argument('--data', type=str, default='data/transcribed',
                        help='Input directory with training data')
    parser.add_argument('--output', type=str, default='models/summarizer',
                        help='Output directory for trained model')
    parser.add_argument('--base-model', type=str, default='google/pegasus-xsum',
                        help='Base model to fine-tune')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    args = parser.parse_args()
    
    print(f"Training Configuration:")
    print(f"  Data: {args.data}")
    print(f"  Output: {args.output}")
    print(f"  Base Model: {args.base_model}")
    print(f"  Epochs: {args.epochs}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if training data exists
    data_dir = Path(args.data)
    transcript_files = list(data_dir.glob('*.json'))
    
    if len(transcript_files) < 10:
        print(f"\n⚠ Warning: Found only {len(transcript_files)} training samples")
        print("Fine-tuning requires at least 100+ samples for good results")
        print("Skipping training and using pre-trained model instead")
        
        # Create a placeholder to indicate pre-trained model is being used
        placeholder_file = output_dir / "using_pretrained.txt"
        with open(placeholder_file, 'w') as f:
            f.write(f"Using pre-trained model: {args.base_model}\n")
            f.write(f"Training skipped due to insufficient data ({len(transcript_files)} samples)\n")
            f.write("Minimum recommended: 100+ transcript samples\n")
        
        # Create metrics file
        metrics_dir = Path("metrics")
        metrics_dir.mkdir(exist_ok=True)
        metrics_file = metrics_dir / "summarization_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump({
                "status": "using_pretrained",
                "base_model": args.base_model,
                "training_samples": len(transcript_files),
                "training_performed": False
            }, f, indent=2)
        
        print(f"\n✓ Placeholder created: {placeholder_file}")
        return
    
    try:
        # Initialize training pipeline
        print("\nInitializing training pipeline...")
        pipeline = SummarizerTrainingPipeline(
            base_model_name=args.base_model,
            output_dir=str(output_dir),
            data_dir=args.data
        )
        
        # Prepare data
        print("Preparing training data...")
        train_dataset, val_dataset = pipeline.prepare_data()
        
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        
        # Train model
        print("\nTraining model...")
        pipeline.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=args.epochs
        )
        
        # Save metrics
        metrics_dir = Path("metrics")
        metrics_dir.mkdir(exist_ok=True)
        metrics_file = metrics_dir / "summarization_metrics.json"
        
        # Get training metrics from pipeline
        metrics = {
            "status": "training_completed",
            "base_model": args.base_model,
            "training_samples": len(train_dataset),
            "validation_samples": len(val_dataset),
            "epochs": args.epochs,
            "output_dir": str(output_dir)
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✓ Training complete!")
        print(f"  Model saved to: {output_dir}")
        print(f"  Metrics saved to: {metrics_file}")
        
    except Exception as e:
        print(f"\n✗ Training failed: {str(e)}")
        print("Using pre-trained model instead")
        
        # Create placeholder
        placeholder_file = output_dir / "training_failed.txt"
        with open(placeholder_file, 'w') as f:
            f.write(f"Training failed: {str(e)}\n")
            f.write(f"Using pre-trained model: {args.base_model}\n")
        
        # Create error metrics
        metrics_dir = Path("metrics")
        metrics_dir.mkdir(exist_ok=True)
        metrics_file = metrics_dir / "summarization_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump({
                "status": "training_failed",
                "error": str(e),
                "using_pretrained": args.base_model
            }, f, indent=2)


if __name__ == "__main__":
    main()
