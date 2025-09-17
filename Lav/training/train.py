# -*- coding: utf-8 -*-
# @Date     : 2025/03/05
# @Author   : Chao Shuai
# @File     : train.py
import os
import random
import datetime
import yaml
import logging
import argparse

import torch
import torch.nn.parallel
import torch.utils.data
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from dataset.ijcai_dataset import DeepfakeAbstractBaseDataset
from trainer.trainer import Trainer
from detectors import DETECTOR


def load_config(args):
    """
    Load and merge configuration parameters from YAML file and command line
    
    Args:
        args: Command line arguments containing path to YAML config
        
    Returns:
        dict: Merged configuration dictionary with CLI arguments taking priority
        
    The configuration hierarchy is:
    1. Base configuration from YAML file
    2. Command line argument overrides for specific keys
    3. Environment variables for distributed training parameters
    """
    # Load base configuration from YAML
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments for specific keys
    override_keys = ['train_dataset', 'test_dataset', 'save_ckpt', 'save_feat']
    for key in override_keys:
        if getattr(args, key, None) is not None:
            config[key] = getattr(args, key)
    return config


def init_seed(config):
    """
    Initialize random number generators for reproducibility
    
    Seeds Python, NumPy (implicit through random), PyTorch CPU/CUDA RNGs
    If no manual seed provided, generates random seed between 1-10000
    """
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])
        
        
def setup_environment(config):
    """Configure fundamental training environment settings"""
    init_seed(config)
    # Enable CuDNN auto-tuner to find optimal convolution algorithms
    torch.backends.cudnn.benchmark = config['cudnn']
    
def setup_distributed(args):
    """
    Configure distributed training parameters
    
    Retrieves environment variables set by torch.distributed.launch or 
    cluster scheduler to initialize process group
    
    Environment Variables:
        WORLD_SIZE: Total number of processes (GPUs × nodes)
        RANK: Global rank across all nodes
        LOCAL_RANK: Local rank within current node
    
    Returns augmented args object with distributed parameters
    """
    args.world_size = int(os.environ.get('WORLD_SIZE', 1))
    args.rank = int(os.environ.get('RANK', 0))
    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    args.distributed = args.world_size > 1
    if args.distributed:
        # Assign GPU per process based on local rank
        torch.cuda.set_device(args.local_rank)
        # Initialize NCCL backend for GPU communication
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )
    return args


def is_main_process(args):
    """
    Determine if current process is primary in distributed setup
    
    Main process handles logging, checkpoint saving, and validation
    to avoid redundant operations across processes
    """
    return not args.distributed or (args.distributed and args.local_rank == 0)


def create_data_loader(dataset, config, args, mode='train'):
    """
    Create optimized DataLoader with distributed sampling support
    
    Features:
    - Automatic sharding via DistributedSampler in distributed mode
    - Persistent workers to maintain worker pools between epochs
    - Pinned memory for faster host-to-device transfers
    """
    sampler = None
    if mode == 'train' and args.distributed:
        # Split dataset across processes without overlap
        sampler = DistributedSampler(
            dataset, 
            num_replicas=args.world_size, 
            rank=args.local_rank
        )
        
    return DataLoader(
        dataset,
        batch_size=config[f'{mode}_batchSize'],
        # Only shuffle in non-distributed training
        shuffle=(mode == 'train' and sampler is None),
        num_workers=config['workers'],
        sampler=sampler,
        pin_memory=True,  # Use page-locked memory for DMA
        persistent_workers=True,  # Maintain worker processes
        drop_last=True
    )
    

def get_device(args):
    """Determine execution device based on availability and arguments"""
    if args.gpu is not None:  # Explicit GPU selection
        return torch.device(f'cuda:{args.gpu}')
    if args.distributed:  # Use assigned GPU in distributed mode
        return torch.device(f'cuda:{args.local_rank}')
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def build_optimizer(model, config):
    """
    Construct optimizer from configuration
    
    Supported Optimizers:
    - SGD: With momentum, weight decay, and Nesterov
    - Adam: With configurable betas, weight decay, and AMSGrad
    - AdamW: Improved Adam with decoupled weight decay
    
    All optimizers support automatic mixed precision via PyTorch AMP
    """
    optimizer_name = config['optimizer']['type']
    lr = config.get('lr', 0.001)

    if optimizer_name == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config['optimizer']['sgd'].get('momentum', 0.9),
            weight_decay=config['optimizer']['sgd'].get('weight_decay', 0.0),
            nesterov=config['optimizer']['sgd'].get('nesterov', False)
        )
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(
                config['optimizer']['adam'].get('beta1', 0.9),
                config['optimizer']['adam'].get('beta2', 0.999)
            ),
            weight_decay=config['optimizer']['adam'].get('weight_decay', 1e-5),
            eps=config['optimizer']['adam'].get('eps', 1e-8),
            amsgrad=config['optimizer']['adam'].get('amsgrad', False)
        )
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(
                config['optimizer']['adamw'].get('beta1', 0.9),
                config['optimizer']['adamw'].get('beta2', 0.999)
            ),
            weight_decay=1e-2,
            eps=config['optimizer']['adamw'].get('eps', 1e-8),
            amsgrad=config['optimizer']['adamw'].get('amsgrad', False)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    return optimizer


def build_scheduler(optimizer, config):
    """
    Configure learning rate scheduler
    
    Supported Schedulers:
    - Step: Fixed interval LR decay
    - Cosine: Cyclical LR with annealing
    - ReduceOnPlateau: Dynamic LR based on validation metrics
    
    Note: ReduceOnPlateau requires validation metrics to be tracked
    """
    scheduler_name = config.get('lr_scheduler')
    if scheduler_name is None:
        return None

    if scheduler_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get('lr_step', 30),
            gamma=config.get('lr_gamma', 0.1)
        )
    elif scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('lr_T_max', 30),
            eta_min=config.get('lr_eta_min', 0)
        )
    else:
        pass

    return scheduler


def resume_training(model, args, config, optimizer, scaler=None):
    """Resume training from a saved checkpoint"""
    # Validate checkpoint existence
    if not os.path.isfile(config['resume']):
        raise FileNotFoundError(f"Checkpoint file not found: {config['resume']}")
    
    # Load checkpoint data
    checkpoint = torch.load(config['resume'], map_location='cpu')
    # Handle different checkpoint formats
    suffix = config['resume'].split('.')[-1]
    if suffix == 'p':
        checkpoint = checkpoint.state_dict()
        
    # Model state loading
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)

    # Restore training progress
    config['start_epoch'] = checkpoint.get('epoch', 0) + 1
    config['global_step'] = checkpoint.get('global_step', 0)
    
    if 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
        
    if 'optimizer' in checkpoint:    
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Log resumption details
    if is_main_process(args):
        args.logger.info(f"Resumed training from checkpoint: {config['resume']}")
        args.logger.info(f"Resuming from epoch {config['start_epoch']}, global step {config['global_step']}")


def initialize_trainer(config, args, model, data_loaders):
    """
    Initialize training supervisor with configured components
    
    Combines:
    - Model architecture
    - Optimizer
    - Learning rate scheduler
    - Data loaders
    - Logging/checkpoint configurations
    
    The Trainer handles the complete training loop, validation,
    and model management
    """
    optimizer = build_optimizer(model, config)
    scaler = torch.amp.GradScaler(enabled=config.get('amp', False))
    
    # Resume from checkpoint if specified
    if config['resume']:
        resume_training(model, args, config, optimizer, scaler)
    
    
    return Trainer(
        config=config,
        args=args,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=scaler,
        train_loader=data_loaders['train'],
        test_loader=data_loaders.get('test', None)
    )
    

def wrap_distributed_model(model, args):
    """
    Wrap model for distributed training with NCCL backend
    
    Features:
    - Automatic gradient synchronization
    - Multi-GPU parameter averaging
    - Device placement based on local rank
    
    Note: find_unused_parameters=True allows for dynamic computation graphs
    but may increase memory usage
    """
    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True
    )

def cleanup(args, trainer):
    """Release distributed resources and close loggers"""
    if args.distributed:
        dist.destroy_process_group()
    if is_main_process(args):
        for writer in trainer.writers.values():
            writer.close()
            

def log_configuration(config, logger):
    """Log full configuration parameters with hierarchical formatting"""
    logger.info('Save log to {}'.format(config['log_dir']))
    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        if isinstance(value, dict):
            params_string += f"{key}:\n"
            for sub_key, sub_value in value.items():
                params_string += f"  {sub_key}: {sub_value}\n"
        else:
            params_string += f"{key}: {value}\n"
    logger.info(params_string)
    
    
def setup_logging(config):
    """
    Configure hierarchical logging system
    
    Features:
    - Time-stamped log directories
    - Model-specific subdirectories
    - Dual file and console logging
    - Automatic directory creation
    """
    timenow = datetime.datetime.now().strftime('%Y-%m-%d')
    # Handle special model configurations in directory names
    log_name = config['model_name'] if "efficient" not in config['model_name'] else config['backbone_config']['model']
    use_other = "_srm_" if config['backbone_config'].get('use_srm', False) else "_"
    use_other = "_f3net_" if config['backbone_config'].get('use_f3net', False) else use_other
    log_dir = os.path.join(config['log_dir'], log_name + use_other + timenow)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, 'training.log')
    logger = create_logger(log_file)

    config['log_dir'] = log_dir
    return logger


def create_logger(log_path):
    """
    Create configured logger instance
    
    Features:
    - File handler for persistent logs
    - Stream handler for real-time monitoring
    - Standardized log format with timestamps
    - Automatic directory structure creation
    """
    if os.path.isdir(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler for persistent logging
    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Console handler for real-time monitoring
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


def main():
    """
    Main training pipeline execution
    
    Workflow:
    1. Parse command line arguments
    2. Load and merge configurations
    3. Initialize distributed environment
    4. Set up logging and diagnostics
    5. Prepare datasets and model
    6. Execute training loop
    7. Handle graceful shutdown
    """
    parser = argparse.ArgumentParser(description='Deepfake Detection Training')
    parser.add_argument('--detector_path', type=str, 
                        default='/path/to/default_config.yaml',
                        help='Path to detector YAML configuration')
    parser.add_argument('--world-size', type=int, default=1, 
                       help='Total number of distributed processes')
    parser.add_argument('--local-rank', type=int, default=0,
                       help='Local process rank for GPU assignment')
    parser.add_argument('--rank', type=int, default=0,
                       help='Global process rank for distributed communication')
    parser.add_argument('--dist-backend', type=str, default='nccl',
                       help='Distributed communication backend')
    parser.add_argument('--dist-url', type=str, default='env://',
                       help='URL for distributed initialization')
    parser.add_argument('--gpu', type=int, default=None,
                       help='Explicit GPU ID for single-GPU training')
    args = parser.parse_args()
    
    # Configuration and environment setup
    config = load_config(args)
    setup_environment(config)
    args = setup_distributed(args)
    args.device = get_device(args)
    
    # Logging initialization
    if is_main_process(args):
        logger = setup_logging(config)
        log_configuration(config, logger)
        args.logger = logger 
    else:
        args.logger = None
    
    # Model initialization
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(args.device)
    preprocess = model._preprocess
        
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = wrap_distributed_model(model, args)
    
    # Dataset preparation
    train_dataset = DeepfakeAbstractBaseDataset(config, args, mode='train', preprocess=preprocess)
    data_loaders = {
        'train': create_data_loader(train_dataset, config, args, 'train')
    }
    # Training execution
    trainer = initialize_trainer(config, args, model, data_loaders)
    try:
        for epoch in range(config['start_epoch'], config['nEpochs'] + 1):
            if args.distributed:
                data_loaders['train'].sampler.set_epoch(epoch)
            
            trainer.train_epoch(epoch)
            
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    finally:
        cleanup(args, trainer)


if __name__ == '__main__':
    main()