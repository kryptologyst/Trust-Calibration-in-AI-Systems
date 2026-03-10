#!/usr/bin/env python3
"""Training script for Trust Calibration in AI Systems."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import yaml
from omegaconf import OmegaConf

from src.main import main as run_analysis


def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level.
        log_file: Log file path (optional).
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Trust Calibration Models")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        help="Override dataset name"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        help="Override output directory"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        help="Override random seed"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else config.get("logging", {}).get("level", "INFO")
    log_file = config.get("logging", {}).get("file")
    setup_logging(log_level, log_file)
    
    # Override config with command line arguments
    if args.dataset:
        config["dataset"]["name"] = args.dataset
    if args.output_dir:
        config["output"]["results_dir"] = args.output_dir
    if args.seed:
        config["dataset"]["random_state"] = args.seed
    
    # Create output directories
    output_dir = Path(config["output"]["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plots_dir = Path(config["output"]["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    models_dir = Path(config["output"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    
    logs_dir = Path(config["output"]["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analysis
    logging.info("Starting Trust Calibration Analysis")
    logging.info(f"Configuration: {config}")
    
    try:
        run_analysis()
        logging.info("Analysis completed successfully")
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
