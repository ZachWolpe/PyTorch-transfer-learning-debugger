"""
--------------------------------------------------------------------
transfer_learning_debugger.py

: 09.12.24
: zachcolinwolpe@gmail.com
--------------------------------------------------------------------
"""

import torch
import torch.nn as nn
from collections import Counter
import time
import numpy as np
from torch.utils.data import DataLoader

class TransferLearningDebugger:
    def __init__(self, model, dataloader, criterion, optimizer):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = next(model.parameters()).device

    # Check input dimensions and requirements
    def debug_input_size(self):
        sample_batch, _ = next(iter(self.dataloader))
        print(f"Batch input shape: {sample_batch.shape}")
        print(f"Input range: [{sample_batch.min():.3f}, {sample_batch.max():.3f}]")
        return sample_batch.shape

    # Verify data normalization
    def check_normalization(self):
        sample_batch, _ = next(iter(self.dataloader))
        print(f"Mean: {torch.mean(sample_batch, dim=[0,2,3])}")
        print(f"Std: {torch.std(sample_batch, dim=[0,2,3])}")
        print(f"Min: {torch.min(sample_batch)}, Max: {torch.max(sample_batch)}")

    # Analyze class distribution
    def analyze_class_distribution(self):
        all_labels = []
        for _, labels in self.dataloader:
            all_labels.extend(labels.tolist())
        distribution = Counter(all_labels)
        print("Class distribution:", distribution)
        return distribution

    # Check layer freezing status
    def check_layer_freezing(self):
        for name, param in self.model.named_parameters():
            print(f"{name}:")
            print(f"  Requires grad: {param.requires_grad}")
            print(f"  Shape: {param.shape}")

    # Monitor weight changes
    def monitor_weight_changes(self, initial_weights=None):
        if initial_weights is None:
            return {name: param.clone().detach() 
                   for name, param in self.model.named_parameters()}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                diff = torch.mean(torch.abs(param - initial_weights[name]))
                print(f"{name} mean change: {diff.item():.6f}")

    # Check learning rates
    def check_learning_rates(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            print(f"Group {i} learning rate: {param_group['lr']}")

    # Monitor memory usage
    def check_memory_usage(self):
        if torch.cuda.is_available():
            print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            print(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")

    # Time training operations
    def time_training_step(self):
        start_time = time.time()
        sample_batch, labels = next(iter(self.dataloader))
        sample_batch, labels = sample_batch.to(self.device), labels.to(self.device)
        
        self.optimizer.zero_grad()
        outputs = self.model(sample_batch)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        
        end_time = time.time()
        print(f"Training step time: {end_time - start_time:.4f} seconds")

    # Monitor train/val metrics
    def track_metrics(self, train_loader, val_loader):
        train_loss, train_acc = self.evaluate(train_loader)
        val_loss, val_acc = self.evaluate(val_loader)
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        return train_loss, train_acc, val_loss, val_acc

    # Evaluate model
    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        self.model.train()
        return running_loss / len(dataloader), correct / total

    # Check model mode
    def verify_model_mode(self):
        print(f"Model in training mode: {self.model.training}")
        print("Batch norm layers status:")
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                print(f"  {name}: training = {module.training}")



# ================================================================================================>>
# Example Usage
# ================================================================================================>>
# Usage example
# debugger = TransferLearningDebugger(model, train_loader, criterion, optimizer)

# # Run various checks
# debugger.debug_input_size()
# debugger.check_normalization()
# debugger.check_layer_freezing()
# debugger.check_memory_usage()

# # Monitor training
# initial_weights = debugger.monitor_weight_changes()
# debugger.time_training_step()
# debugger.track_metrics(train_loader, val_loader)
# debugger.monitor_weight_changes(initial_weights)
# ================================================================================================>>
# Example Usage
# ================================================================================================>>
