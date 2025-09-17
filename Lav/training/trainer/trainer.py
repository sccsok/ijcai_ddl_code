# -*- coding: utf-8 -*-
# @Date     : 2025/03/05
# @Author   : Chao Shuai
# @File     : trainer.py
import os
import math
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from metrics.base_metrics_class import Recorder


class Trainer:
    def __init__(
        self,
        config, 
        args,
        model, 
        optimizer=None, 
        scheduler=None,
        scaler=None,
        train_loader=None,
        test_loader=None
    ):
        """
        Main training controller class
        
        Args:
            config: Configuration dictionary containing training parameters
            args: Command line arguments
            model: Model to be trained
            optimizer: Optimization algorithm
            scheduler: Learning rate scheduler
            train_loader: Training data loader
            test_loader: Validation data loader
        """
        # Initialize core components
        self._init_core_components(config, args, model, optimizer, scheduler, scaler, train_loader, test_loader)
        # Set up logging infrastructure
        self._init_logging_system()
        # Initialize training state variables
        self._init_training_state()
        
    def _init_core_components(self, config, args, model, optimizer, scheduler, scaler, train_loader, test_loader):
        """Initialize fundamental training components"""
        self.config = config
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        # Initialize recorders for tracking loss and metrics
        self.train_recorder_loss = defaultdict(Recorder)
        self.train_recorder_metric = defaultdict(Recorder)
        
        # Device configuration
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Mixed precision gradient scaler
        self.scaler = scaler
        # Progress bar initialization
        self.pbar = self._create_progress_bar(desc="Training Progress", dynamic_ncols=True)
        self._adjust_learning_rate(config['start_epoch'])

    def _init_logging_system(self):
        """Initialize metrics tracking and visualization system"""
        self.writers = {}
        # Configure primary evaluation metric
        self.metric_scoring = self.config.get('metric_scoring', 'auc')
        # Initialize metric storage with appropriate extremes
        self.best_metrics = defaultdict(
            lambda: defaultdict(lambda: -math.inf if self.metric_scoring != 'eer' else math.inf)
        )

    def _init_training_state(self):
        """Initialize counters and state trackers"""
        self.current_epoch = 0
        self.global_step = 0
        self.early_stop_counter = 0

    def _prepare_batch_data(self, data_dict):
        """Prepare batch data for GPU processing with async transfers"""
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            # Move tensors to device with non-blocking transfers
            data_dict = {
                k: v.to(self.args.device, non_blocking=True) 
                if isinstance(v, torch.Tensor) else v 
                for k, v in data_dict.items()
            }
        # Synchronize with current stream
        torch.cuda.current_stream().wait_stream(stream)
        return data_dict
    
    def train_epoch(self, epoch):
        """Execute one complete training epoch"""
        self.model.train()
        self.current_epoch = epoch

        # Update progress bar description
        if self._is_main_process():
            self.pbar.set_description(f"Epoch: {epoch}/{self.config['nEpochs']}")
            
        # Batch iteration loop
        for batch_idx, data_dict in enumerate(self.train_loader):
            # Prepare batch and update step counter
            data_dict = self._prepare_batch_data(data_dict)
            self.global_step += 1
            
            # Mixed precision forward pass
            with torch.amp.autocast(device_type=self.device_type, enabled=self.config.get('amp', False)):
                predictions = self.model(data_dict)
                losses = self._compute_losses(data_dict, predictions)
            
                if torch.isnan(losses['overall']).any():
                    raise ValueError(f"NaN detected in loss at batch epoch {epoch}, iteration {batch_idx}")
                
                loss = losses['overall'] / self.config.get('accum_iter', 1)

            self.scaler.scale(loss).backward()
            # Backward pass and parameter update
            if (batch_idx + 1) % self.config.get('accum_iter', 1) == 0:
                if self.config.get('grad_clip'):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self._adjust_learning_rate(epoch + batch_idx / len(self.train_loader)) 

            # Metrics calculation and logging
            self._compute_metrics(data_dict, predictions)
            
            if self._is_main_process():
                self._log_batch_stats(batch_idx)
                self.pbar.update(1)
                self.pbar.set_postfix(loss=losses['overall'].item()) 

        # Epoch completion tasks        
        self._save_checkpoint() 
        # Finalize progress bar
        if self._is_main_process() and epoch == self.config['nEpochs']:
            self.pbar.close()
    
    def _adjust_learning_rate(self, epoch):
        """Adjust learning rate according to schedule"""
        if self.scheduler is not None:
            # Use external scheduler (e.g., torch.optim.lr_scheduler.CosineAnnealingLR)
            self.scheduler.step()
        else:
            # Manual scheduling
            warmup_epochs = self.config.get('warmup_epoch', 0)
            total_epochs = self.config['nEpochs']

            # Warmup
            if epoch < warmup_epochs:
                lr = self.config['lr'] * epoch / warmup_epochs
            else:
                lr_scheduler = self.config.get('lr_scheduler')
                if lr_scheduler == 'cosine':
                    # Cosine Annealing without min_lr
                    lr = self.config['lr'] * 0.5 * (
                        1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs))
                    )
                elif lr_scheduler == 'mincosine':
                    # Cosine with minimum learning rate
                    min_lr = self.config.get('min_lr', 1e-7)
                    lr = min_lr + (self.config['lr'] - min_lr) * 0.5 * (
                        1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs))
                    )
                elif lr_scheduler == 'step':
                    # Step decay
                    steps = (epoch - warmup_epochs) // self.config['lr_step']
                    lr = self.config['lr'] * (self.config['lr_gamma'] ** steps)
                else:
                    lr = self.config['lr']

            for param_group in self.optimizer.param_groups:
                base_lr = lr
                if "lr_scale" in param_group:
                    base_lr *= param_group["lr_scale"]
                param_group["lr"] = base_lr
                
    def _compute_metrics(self, data_dict, predictions):
        """Calculate and store training metrics"""
        # Handle distributed model wrapper
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            batch_metrics = self.model.module.get_train_metrics(data_dict, predictions)
        else:
            batch_metrics =  self.model.get_train_metrics(data_dict, predictions)
        # Update metric recorders
        for name, value in batch_metrics.items():
            self.train_recorder_metric[name].update(value)
            
        return batch_metrics
        
    def _compute_losses(self, data_dict, predictions):
        """Calculate and store training losses"""
        # Handle distributed model wrapper
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            losses = self.model.module.get_losses(data_dict, predictions)
        else:
            losses = self.model.get_losses(data_dict, predictions)
        # Update loss recorders
        for name, value in losses.items():
            self.train_recorder_loss[name].update(value.item())
        return losses
        
    def _log_batch_stats(self, batch_idx):
        """Log metrics to TensorBoard and console"""
        if not self._is_main_process():
            return

        # Compile log data
        log_data = {
            **{f'loss/{k}': v.average() for k, v in self.train_recorder_loss.items()},
            **{f'metric/{k}': v.average() for k, v in self.train_recorder_metric.items()},
            'lr': self.optimizer.param_groups[0]['lr']
        }
        
        # Log at specified intervals
        if (batch_idx + 1) % int(self.config.get('rec_iter', 500)) == 0:
            # Format log string
            log_str = " | ".join([
                f"{k}: {v:.4f}" if isinstance(v, (int, float)) else
                f"{k}: {v.average()}"
                for k, v in log_data.items()
            ])
            
            # Write to TensorBoard
            for metric_name, value in log_data.items():
                self._get_writer('train').add_scalar(metric_name, value, self.global_step)

    def _get_writer(self, phase):
        """Get TensorBoard writer for specific phase (train/val/test)"""
        if phase not in self.writers:
            # Create new writer if needed
            writer_path = os.path.join(self.config['log_dir'], phase)
            os.makedirs(writer_path, exist_ok=True)
            self.writers[phase] = SummaryWriter(writer_path)
        return self.writers[phase]
    
    def _save_checkpoint(self, is_best=False):
        """Save training state snapshot"""
        if not self._should_save_checkpoint():
            return
        
        # Prepare checkpoint state
        state = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        
        target_path = self._get_ckpt_path(f'last_epoch_{self.current_epoch - 1}')
        if os.path.exists(target_path):
            os.remove(target_path)
        # Save standard checkpoint
        torch.save(state, self._get_ckpt_path(f'last_epoch_{self.current_epoch}'))

        # Save best model checkpoint
        if is_best:
            torch.save(state, self._get_ckpt_path(F'best_epoch_{self.current_epoch}'))

        # Periodic checkpointing
        if self.current_epoch % self.config.get('save_interval', 5) == 0:
            torch.save(state, self._get_ckpt_path(f'epoch_{self.current_epoch}'))

    def _get_ckpt_path(self, name):
        """Generate checkpoint file path"""
        os.makedirs(os.path.join(self.config['log_dir'], 'ckpt'), exist_ok=True)
        return os.path.join(self.config['log_dir'], 'ckpt', f'{name}.pth')

    def _should_save_checkpoint(self):
        """Determine if checkpoint should be saved (main process only)"""
        return self._is_main_process() and self.config.get('save_ckpt', True)

    def _is_main_process(self):
        """Check if current process is the main process in distributed training"""
        return self.args.local_rank in [-1, 0]

    def _create_progress_bar(self, **kwargs):
        """Initialize training progress bar"""
        if self._is_main_process():
            total_steps = len(self.train_loader) * (self.config['nEpochs'] - self.config['start_epoch'])
            print(f"Training iterations: {len(self.train_loader)} batches/epoch, {total_steps} total steps")
            return tqdm(total=total_steps, **kwargs)
        return None

    def _cleanup_resources(self):
        """Release system resources and clean up distributed processes"""
        for writer in self.writers.values():
            writer.close()
        if self.args.distributed:
            dist.destroy_process_group()