# PyTorch Transfer Learning Debugger (v0.1.1)

Diagnose and debug common issues when performing in transfer learning & fine-tuning with `PyTorch`.



![Computational Graph](https://github.com/ZachWolpe/PyTorch-transfer-learning-debugger/blob/main/assets/comp-graph2.png)



**_An interactive debugging experience for PyTorch code, enabling users to diagnose issues related to weight updates during training, as well as to address problems with slow or unstable convergence._**



---
# Installation (PyPI)

pip install the package:

```bash
pip install pytorch-transfer-learning-debugger
```

or

```bash
pip install git+https://github.com/ZachWolpe/PyTorch-transfer-learning-debugger.git
```

[Access the `PyPI` source](https://pypi.org/project/pytorch-transfer-learning-debugger/).

----
# Benefits

- Helps identify if weights are properly updating during training
- Provides visibility into each stage of the training process
- Enables early detection of training issues
- Supports both high-level and granular debugging approaches

This debugging tool is particularly useful when:
- Troubleshooting training convergence issues
- Verifying proper weight updates
- Understanding the flow of data through the model
- Diagnosing optimization problems


----
# Getting Started


### Purpose
_An interactive debugging tool for PyTorch code that helps diagnose issues related to weight updates during training and addresses problems with slow or unstable convergence._

### Key Components

## 1. Debug Mode Features
The debugger is implemented through a `torch_debugger` class that provides two main debugging modes:

1. **Basic Debug Mode** (`debug_mode=True`):
   - Tracks model weight updates
   - Only monitors weights with `requires_grad=True`
   - Populates `torch_debugger._track_weights` dictionary for analysis

2. **Granular Debug Mode** (`granular_logging=True`):
   - Exits after one cycle through the dataloader
   - Logs each stage of the computational graph:
     - Optimizer verification
     - Gradient zeroing
     - Data movement to device
     - Forward pass and loss computation
     - Gradient computation and optimizer steps

## 2. Implementation Example

Here's how the debugger is integrated into the training loop:

```python
def train_model(model, criterion, optimizer, scheduler, num_epochs=25, 
                debug_mode=True, granular_logging=False):
    # Initialize debugger if debug_mode is enabled
    torch_debugger_inst = None
    if debug_mode:
        torch_debugger_inst = torch_debugger(enable_log_stage=granular_logging)
        torch_debugger_inst.verify_optimizer_state(optimizer)
        torch_debugger_inst.initial_model_weights_state = model.state_dict().copy()
        
    # Training loop with debugging hooks
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            for inputs, labels in dataloaders[phase]:
                if debug_mode:
                    torch_debugger_inst.log_stage('Moving data to device')
                    # ... more logging stages ...
                    
                if phase == 'train':
                    torch_debugger_inst.track_weights(model.named_parameters())
```

## 3. Usage

To use the debugger:

```python
torch_debugger_inst, model = train_model(
    model_ft,
    criterion,
    optimizer_ft,
    exp_lr_scheduler,
    num_epochs=25,
    debug_mode=True,
    granular_logging=True  # For detailed stage logging
)
```

## 4. Analysis Tools

The debugger provides visualization tools to analyze training:

```python
# Plot weight updates over time
torch_debugger_inst.plot_weight_updates()

# Access tracked weights
weights = torch_debugger_inst._track_weights['conv1.weight']
```

## 5. Results

![Weight Updates](https://github.com/ZachWolpe/PyTorch-transfer-learning-debugger/blob/main/assets/result.png)


----
# Comprehensive Tutorial

For a comprehensive tutorial, see the `GettingStarted.ipynb` Juypyter notebook.


----
# Potential Issues during Transfer Learning

| Category | Issue | Symptoms | Solutions |
|----------|--------|-----------|-----------|
| Learning Rate | Improper Configuration | • Too high: unstable training<br>• Too low: slow learning/stuck<br>• Poor adjustment for transfer | • Start with 10-3 or 10-4 of original LR<br>• Use LR finder<br>• Implement LR scheduling<br>• Use different LRs for layers |
| Layer Management | Freezing Issues | • Wrong layers frozen<br>• Too many/few layers frozen<br>• No gradual unfreezing<br>• Model not learning | • Start by freezing all but final layers<br>• Gradually unfreeze from top<br>• Monitor layer gradients<br>• Selective layer unfreezing |
| Data Preparation | Input Problems | • Runtime errors<br>• Poor model performance<br>• Slow convergence<br>• Class imbalance | • Adjust transforms<br>• Match source preprocessing<br>• Balance classes<br>• Use validation set<br>• Data augmentation |
| Architecture | Structure Mismatches | • Bad final layer modifications<br>• Size/channel mismatches<br>• Poor layer initialization<br>• Training fails | • Verify input/output dimensions<br>• Use proper initialization<br>• Match pretrained architecture<br>• Add adaptation layers |
| Training Process | Optimization Issues | • Not converging<br>• Unstable training<br>• Poor generalization<br>• Memory errors | • Use Adam/AdamW for fine-tuning<br>• Verify loss matches task<br>• Start with small batches<br>• Use gradient accumulation |
| Implementation | Technical Errors | • Model not in train mode<br>• Gradients not zeroed<br>• Wrong device (CPU/GPU)<br>• Memory leaks | • Use training checklist<br>• Implement proper train/eval<br>• Check device placement<br>• Monitor memory usage |
| Resource Management | Performance Issues | • OOM errors<br>• Slow training<br>• Resource inefficiency | • Batch size adjustment<br>• Gradient checkpointing<br>• Efficient data loading<br>• Use GPU acceleration |
| Monitoring & Validation | Quality Control | • Poor metric tracking<br>• Missing validation<br>• Overfitting<br>• Inconsistent results | • Use debugging tools<br>• Implement validation loops<br>• Track multiple metrics<br>• Monitor gradient flow |
| Pretrained Model | Weight Issues | • Wrong pretrained weights<br>• Corrupted weights<br>• Version incompatibility | • Verify model source<br>• Check model checksums<br>• Match framework versions |
| Feature Extraction | Transfer Problems | • Suboptimal transfer<br>• Poor adaptation<br>• Loss of pretrained features | • Choose appropriate layers<br>• Add adaptation layers<br>• Fine-tune feature extractors<br>• Knowledge distillation |

----
# Potential Issues with the Computational Graph


| Issue Category | Specific Problem | Symptoms | Solutions |
|----------------|------------------|-----------|------------|
| Gradient Flow | Vanishing Gradients | - Near-zero gradients in early layers<br>- Model not learning | - Use gradient clipping<br>- Add residual connections<br>- Change activation functions |
| | Exploding Gradients | - NaN losses<br>- Large gradient values | - Implement gradient clipping<br>- Reduce learning rate<br>- Check initialization |
| | Disconnected Graphs | - Some parameters not updating<br>- Partial learning | - Remove accidental detach()<br>- Verify tensor operations maintain graph |
| Tensor Operations | In-place Operations | - Backward pass errors<br>- "Leaf variable modified" error | - Replace in-place ops (+=) with regular ops (+)<br>- Create new tensors instead of modifying |
| | Device Mismatches | - Runtime errors<br>- CUDA errors | - Use .to(device) consistently<br>- Implement device checker |
| | Detached Tensors | - No gradients flowing<br>- Parts of model not learning | - Set requires_grad=True where needed<br>- Remove unnecessary detach() calls |
| Autograd Engine | Broken Computational Paths | - Gradients not computed<br>- backward() errors | - Fix graph connections<br>- Ensure proper tensor operations |
| | Mixed Precision Errors | - NaN losses<br>- Unstable training | - Use GradScaler<br>- Adjust loss scaling<br>- Check dtype consistency |
| Loss Computation | Zero/NaN Losses | - Model not learning<br>- Training instability | - Verify loss function implementation<br>- Check input normalization |
| | Wrong Reduction | - Incorrect gradient scaling<br>- Slow convergence | - Set appropriate reduction method<br>- Verify batch dimension handling |



---
```
: zachcolinwolpe@gmail.com
: 09.12.2024
```
