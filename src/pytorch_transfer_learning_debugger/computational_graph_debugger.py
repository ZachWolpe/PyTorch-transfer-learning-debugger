"""
--------------------------------------------------------------------
computational_graph_debugger.py

: 09.12.24
: zachcolinwolpe@gmail.com
--------------------------------------------------------------------
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import numpy as np
from torch.cuda.amp import GradScaler
import warnings

class ComputationalGraphDebugger:
    def __init__(self, model: nn.Module, criterion: nn.Module):
        self.model = model
        self.criterion = criterion
        self.device = next(model.parameters()).device
        self.grad_history: Dict[str, List[float]] = {}
        self.scaler = GradScaler()  # for mixed precision training

    # Monitor gradient flow through the network
    def check_gradient_flow(self) -> Dict[str, float]:
        """
        Returns mean gradient for each layer
        """
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                gradients[name] = param.grad.abs().mean().item()
                print(f"{name}: gradient = {gradients[name]:.6f}")
            else:
                print(f"{name}: No gradient available")
        return gradients

    # Track gradient history over time
    def update_gradient_history(self):
        """
        Store gradient history for tracking changes
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if name not in self.grad_history:
                    self.grad_history[name] = []
                self.grad_history[name].append(param.grad.abs().mean().item())

    # Check for exploding gradients
    def check_exploding_gradients(self, threshold: float = 1.0) -> bool:
        """
        Returns True if gradients are exploding
        """
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > threshold:
                    print(f"Exploding gradient in {name}: {grad_norm}")
                    return True
        return False

    # Verify tensor device placement
    def check_device_consistency(self) -> bool:
        """
        Returns True if all model parameters are on the same device
        """
        devices = set()
        for name, param in self.model.named_parameters():
            devices.add(param.device)
            print(f"{name}: {param.device}")
        return len(devices) == 1

    # Monitor computational graph connectivity
    def verify_graph_connectivity(self, loss: torch.Tensor) -> None:
        """
        Print the computational graph structure
        """
        print("Computational graph structure:")
        print(f"Loss grad_fn: {type(loss.grad_fn).__name__}")
        node = loss.grad_fn
        while node is not None:
            print(f"→ {type(node).__name__}")
            node = node.next_functions[0][0] if node.next_functions else None

    # Check for detached tensors
    def check_requires_grad(self) -> Dict[str, bool]:
        """
        Returns dictionary of parameters and their requires_grad status
        """
        grad_status = {}
        for name, param in self.model.named_parameters():
            grad_status[name] = param.requires_grad
            print(f"{name}: requires_grad = {param.requires_grad}")
        return grad_status

    # Monitor loss values for stability
    def check_loss_stability(self, loss: torch.Tensor) -> bool:
        """
        Returns False if loss is unstable (nan/inf)
        """
        loss_value = loss.item()
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Unstable loss value: {loss_value}")
            return False
        print(f"Loss value: {loss_value}")
        return True

    # Debug mixed precision training
    def debug_mixed_precision(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
        """
        Perform one step of mixed precision training with debugging
        """
        try:
            # Scale loss and compute gradients
            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward()
            
            # Check for inf/nan gradients
            self.scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            print(f"Gradient norm after unscaling: {grad_norm}")
            
            # Update weights
            self.scaler.step(optimizer)
            self.scaler.update()
            
        except RuntimeError as e:
            print(f"Mixed precision error: {str(e)}")

    # Monitor memory usage of tensors
    def check_memory_usage(self) -> None:
        """
        Print memory usage statistics
        """
        if torch.cuda.is_available():
            print("\nMemory Usage:")
            print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            print(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")

    # Verify backward hooks
    def add_gradient_hooks(self):
        """
        Add hooks to monitor gradient flow
        """
        def hook_fn(grad):
            print(f"Gradient shape: {grad.shape}")
            print(f"Gradient mean: {grad.mean().item()}")
            return grad

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.register_hook(lambda grad, name=name: 
                    print(f"Gradient for {name}: mean={grad.mean().item()}, std={grad.std().item()}"))

    # Check reduction in loss function
    def verify_loss_reduction(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Compare different reduction methods in loss computation
        """
        reductions = ['mean', 'sum', 'none']
        for reduction in reductions:
            criterion = type(self.criterion)(reduction=reduction)
            loss = criterion(outputs, targets)
            print(f"Loss with {reduction} reduction: {loss}")

    # Full debugging report
    def generate_debug_report(self, loss: torch.Tensor) -> Dict:
        """
        Generate comprehensive debugging report
        """
        report = {
            'gradient_flow': self.check_gradient_flow(),
            'device_consistency': self.check_device_consistency(),
            'requires_grad': self.check_requires_grad(),
            'loss_stability': self.check_loss_stability(loss),
            'memory_usage': None
        }
        
        if torch.cuda.is_available():
            report['memory_usage'] = {
                'allocated': torch.cuda.memory_allocated(),
                'cached': torch.cuda.memory_reserved()
            }
        
        return report



# ================================================================================================>>
# Example Usage
# ================================================================================================>>

# # Initialize debugger
# debugger = ComputationalGraphDebugger(model, criterion)

# # During training loop
# def training_step(model, inputs, targets, optimizer):
#     # Add hooks for gradient monitoring
#     debugger.add_gradient_hooks()
    
#     # Forward pass
#     outputs = model(inputs)
#     loss = criterion(outputs, targets)
    
#     # Check computational graph
#     debugger.verify_graph_connectivity(loss)
    
#     # Check loss stability
#     if not debugger.check_loss_stability(loss):
#         warnings.warn("Unstable loss detected!")
    
#     # Backward pass with gradient monitoring
#     loss.backward()
#     debugger.check_gradient_flow()
    
#     # Check for exploding gradients
#     if debugger.check_exploding_gradients():
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
#     # Update weights
#     optimizer.step()
#     optimizer.zero_grad()
    
#     # Generate debug report
#     debug_report = debugger.generate_debug_report(loss)
#     return loss, debug_report

# ================================================================================================>>
# Example Usage
# ================================================================================================>>
