"""
--------------------------------------------------------------------
torch_debugger.py

main file containing the debugger logic.

: 06.12.24
: zachcolinwolpe@gmail.com
--------------------------------------------------------------------
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from tqdm import tqdm
import matplotlib.pyplot as plt

cudnn.benchmark = True
# plt.ion()   # interactive mode


class torch_debugger:
    """
    ------------------------------------------------------------------------
    pyTorch Transfer Learning Debugger
    ----------------------------------

    This class provides functionality to debug and visualize the training 
    process of models using transfer learning in PyTorch. It includes methods 
    for logging weight statistics, tracking weight updates, and displaying 
    messages during the training process.

    Methods
    -------
    log_weights_mean(named_parameters, keys=None):
        Logs the mean of specified model parameter(s) during training.

    initial_model_weights_state:
        Property to get or set the initial state of model weights - for benchmarking.

    log_stage(msg: str):
        Logs the current stage of the training process with a message.

    plot_weight_updates(parameter_name='conv1.weight'):
        Plots the updates of the specified parameter weights over time.

    log_predefined_message(progress_str, message_id=0):
        Logs a predefined message based on the provided message ID.

    ------------------------------------------------------------------------
    """

    def __init__(self, enable_log_stage=True) -> None:
        self.log_stage_counter = 0
        self._initial_model_weights_state = None
        self.enable_log_stage = enable_log_stage
        self.ave_grads = []
        self.layers = []
        self._track_weights = {}

    def log_weights_mean(self, named_parameters, keys=None):
        if keys is None:
            keys = ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var', 'bn1.num_batches_tracked',]
            keys = ['conv1.weight', 'bn1.weight']

        for name, param in named_parameters:
            if param.requires_grad and name in keys:
                print(f'{name}   : {torch.mean(param):^10}')

    @property
    def initial_model_weights_state(self):
        return self._initial_model_weights_state

    @initial_model_weights_state.setter
    def initial_model_weights_state(self, value):
        self._initial_model_weights_state = value

    def log_stage(self, msg: str = 'Moving data to device'):
        if self.enable_log_stage:
            print(f"    >> Step {self.log_stage_counter} : {msg:^10}")
            self.log_stage_counter += 1

    def plot_weight_updates(self, parameter_name='conv1.weight'):
        assert self._track_weights[parameter_name] is not None, 'Parameter not tracked.'

        weights_numpy = [tensor.detach().cpu().numpy() for tensor in self._track_weights[parameter_name]]
        plt.figure(figsize=(10, 5))
        plt.plot(weights_numpy)
        plt.title('Weights of conv1 Layer')
        plt.xlabel('Weight Index')
        plt.ylabel('Weight Value')
        plt.grid()
        plt.show()

    def log_predefined_message(self, progress_str, message_id=0):

        if message_id == 0:
            _msg = f"""
            -----------------------------------------------------------------------------
            Launching proc job: {progress_str}
            -----------------------------------------------------------------------------
            """
            print(_msg)
            return

        # Replace 'raise inv' with proper exception
        raise ValueError(f"Invalid message_id: {message_id}. Only message_id in [0] is currently supported.")

    def check_gradient_flow(self, named_parameters):
        """
        Checks if gradients are flowing properly through the network
        """
        ave_grads = []
        layers = []
        for n, p in named_parameters:
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                if p.grad is None:
                    ave_grads.append(0)
                    print(f"No gradient for {n}")
                else:
                    ave_grads.append(p.grad.abs().mean().item())
            self.ave_grads.append(ave_grads)
            self.layers.append(layers)
            return ave_grads, layers

    def DEPRECATED_log_change_in_gradients(self, named_parameters, initial_weights: dict):
        """
        Log change in gradients.
        Test whether or not the gradients are being updated.

        *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** 
        DEPRECATED: MOVED TO TESTING CHANGE IN MODULE WEIGHTS.
        *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** 
        """
        WEIGHTS_UPDATED = False
        for name, param in named_parameters:
            if param.requires_grad:
                old_weight = initial_weights[name]
                new_weight = param.data
                diff = torch.sum(torch.abs(new_weight - old_weight)).item()
                if diff > 0:
                    print(f"    > {name}: Changed by {diff:.6f}")
                    WEIGHTS_UPDATED = True
                print(f"> Warning {name} weight not updated.", end='--')
        return WEIGHTS_UPDATED

    def check_loss(self, batch_loss):
        # After loss calculation:
        if batch_loss.item() < 1e-8:
            print(f"Warning: Very small loss value: {batch_loss.item()}")
            return False
        return True

    def verify_optimizer_state(self, optimizer):
        """
        ------------------------------------------------------------------------------
        Verifies optimizer is properly configured.

        Optimizer should:
            - Have a learning rate that is appropriate for the training process.
            - Include parameters that require gradients.
            - Be set up with the correct weight decay and momentum (if applicable).
            - Not have any parameters that are set to None.
            - Be reset or re-initialized if the training process is restarted.

        Example Usage
        -------------
            Add before training loop:
            Verify_optimizer_state(self.optimizer)

        ------------------------------------------------------------------------------ 
        """
        print("\nOptimizer State:")
        for i, group in enumerate(optimizer.param_groups):
            print(f"\nParameter group {i}:")
            print(f"Learning rate: {group['lr']}")
            params_with_grad = [p for p in group['params'] if p.requires_grad]
            print(f"Parameters requiring gradient: {len(params_with_grad)}")
            if len(params_with_grad) == 0:
                print("Warning: No parameters require gradients in this group!")

    def DEPRECATED_track_weight_updates(self, model, initial_weights=None):
        """
        Tracks weight changes during training

        *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** 
        DEPRECATED: MOVED TO TESTING CHANGE IN MODULE WEIGHTS.
        *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** 
        """
        if initial_weights is None:
            # Store initial weights
            return {name: param.clone().detach() for name, param in model.named_parameters()}
        else:
            # Compare current weights with initial
            print("\nWeight changes:")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    diff = (param - initial_weights[name]).abs().mean().item()
                    print(f"{name}: mean absolute change = {diff:.8f}")

    def check_gradients_are_not_null(self, named_parameters, log=False):
        """
        ------------------------------------------------------------------------------
        Test if the gradients exist (!= None).

        Check gradients before and after calling optimizer.step()

        Example Usage
        -------------
            torch_debugger.check_gradients_are_not_null(self.model.named_parameters())
            self.optimizer.step()
            torch_debugger.check_gradients_are_not_null(self.model.named_parameters())

        Returns
        -------------
            bool
            TRUE  : if any gradients are not null.
            FALSE : if all gradients are null

        ------------------------------------------------------------------------------
        """
        # Before optimizer.step()
        NULL_GRADIENTS = True
        for name, param in named_parameters:
            if param.requires_grad:
                if log:
                    print(f"{name}: grad exists: {param.grad is not None}")
                if param.grad is not None:
                    if log:
                        print(f"grad norm: {param.grad.norm().item()}")
                    NULL_GRADIENTS = True
        return NULL_GRADIENTS

    def track_weights(self, named_parameters):
        """
        Track weight changes - to ensure that weights are being updated.
        """

        for name, param in named_parameters:
            if param.requires_grad:
                if name in self._track_weights.keys():
                    self._track_weights[name].append(torch.mean(param))
                else:
                    self._track_weights[name] = [torch.mean(param)]

    def verify_optimizer_step(self, model, optimizer):
        """
        Verify optimizer step.

        *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** 
        DEPRECATED: MOVED TO TESTING CHANGE IN MODULE WEIGHTS.
        *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** *** 
        """
        # Store initial weights
        initial_weights = {name: param.clone().detach() for name, param in model.named_parameters()}
        # Create a dummy optimization step
        dummy_loss = sum([p.sum() for p in model.parameters() if p.requires_grad])
        optimizer.zero_grad()
        dummy_loss.backward()
        optimizer.step()

        # Check if weights changed
        changed = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                diff = (param - initial_weights[name]).abs().sum().item()
                if diff > 0:
                    changed = True
                    print(f"Parameter {name} changed by {diff}")
        return changed


# ----------------------------------------------------------------------------------------->>
# Implementation Guide -------------------------------------------------------------------->>

# Implementation

# Check gradient flow
# ave_grads, layers = torch_debugger.check_gradient_flow(named_parameters)

# Batch Loss
# torch_debugger.check_loss(batch_loss)

# Additional
# print(f'batch_loss      :: {batch_loss}')
# print(f' self.optimizer :: {self.optimizer}')
# After computing the loss
# print("Loss computational graph:")
# print(batch_loss.grad_fn)

# Implementation Guide -------------------------------------------------------------------->>
# ----------------------------------------------------------------------------------------->>

