# PyTorch Transfer Learning Debugger




---
# Getting Started

pip install the package:

```bash
pip install torch-transfer-learning-debugger
```

or

```bash
pip install git+<URL>
```

Then, run the debugger by incorporating it in your training loop.

```python


```



----
# Potential Faults during Transfer Learning

| Category | Failure Points | Common Solutions |
|----------|---------------|------------------|
| Learning Rate | • Too high: unstable training<br>• Too low: slow learning/stuck<br>• Improper adjustment for transfer learning | • Start with 10-3 or 10-4 of original LR<br>• Use LR finder<br>• Implement LR scheduling |
| Layer Freezing | • Wrong layers frozen<br>• Too many/few layers frozen<br>• No gradual unfreezing | • Start by freezing all but final layers<br>• Gradually unfreeze from top<br>• Monitor layer gradients |
| Data Issues | • Insufficient data<br>• Poor quality/preprocessing<br>• Domain shift<br>• Class imbalance<br>• Wrong normalization | • Data augmentation<br>• Match source preprocessing<br>• Balance classes<br>• Use validation set |
| Architecture | • Bad final layer modifications<br>• Size/channel mismatches<br>• Poor layer initialization<br>• Wrong output dimensions | • Verify input/output dimensions<br>• Use proper initialization<br>• Match pretrained architecture |
| Optimization | • Wrong optimizer choice<br>• Incorrect loss function<br>• Bad batch size<br>• Poor momentum settings | • Use Adam/AdamW for fine-tuning<br>• Verify loss matches task<br>• Start with small batches |
| Implementation | • Model not in train mode<br>• Gradients not zeroed<br>• Wrong device (CPU/GPU)<br>• Memory leaks | • Use training checklist<br>• Implement proper train/eval<br>• Check device placement<br>• Monitor memory usage |
| Pretrained Model | • Wrong pretrained weights<br>• Corrupted weights<br>• Version incompatibility | • Verify model source<br>• Check model checksums<br>• Match framework versions |
| Monitoring | • Poor metric tracking<br>• No early stopping<br>• Missing validation<br>• No gradient monitoring | • Use debugging tools<br>• Implement validation loops<br>• Track multiple metrics<br>• Monitor gradient flow |


----
# Potential (General) Issues with Transfer Learning

| Category | Issue | Symptoms | Debugging Steps | Solutions |
|----------|--------|----------|-----------------|-----------|
| Data Preparation | Input Size Mismatch | - Runtime errors<br>- Poor model performance | pythonprint(f"Input shape: {x.shape}")print(f"Expected: {model.input_size}") | - Adjust transforms<br>- Verify model input requirements<br>- Use proper resizing |
| | Incorrect Normalization | - Slow convergence<br>- Poor performance | pythonprint(f"Mean: {torch.mean(x)}")print(f"Std: {torch.std(x)}") | - Use pretrained model stats<br>- Verify normalization values<br>- Check data range |
| | Class Imbalance | - Biased predictions<br>- Poor minority class performance | pythonfor c in classes: print(f"Class {c}: {len(data[c])}") | - Class weights in loss<br>- Oversampling/undersampling<br>- Augmentation for minority classes |
| Model Architecture | Layer Freezing | - Model not learning<br>- Overfitting/underfitting | pythonfor name, param in model.named_parameters(): print(f"{name}: {param.requires_grad}") | - Selective layer unfreezing<br>- Progressive unfreezing<br>- Fine-tune specific layers |
| | Final Layer Mismatch | - Runtime errors<br>- Training fails | pythonprint(f"Final layer in: {model.fc.in_features}")print(f"Final layer out: {model.fc.out_features}") | - Adjust final layer dimensions<br>- Add proper adaptation layers<br>- Verify architecture |
| | Feature Extraction | - Suboptimal transfer<br>- Poor adaptation | python# Check which features are being usedfeatures = model.features(x)print(f"Feature shape: {features.shape}") | - Choose appropriate layers<br>- Add custom feature extractors<br>- Modify architecture |
| Training Process | Learning Rate | - Not converging<br>- Unstable training | pythonfor param_group in optimizer.param_groups: print(f"LR: {param_group['lr']}") | - Use different LRs for layers<br>- Implement LR scheduling<br>- Start with small LR |
| | Catastrophic Forgetting | - Poor generalization<br>- Loss of pretrained features | python# Monitor pretrained layer weightsinitial_weights = {}for name, param in model.named_parameters(): initial_weights[name] = param.clone() | - Gradual fine-tuning<br>- Regularization techniques<br>- Knowledge distillation |
| | Batch Size | - Memory errors<br>- Poor convergence | pythonprint(f"Batch size: {len(x)}")print(f"Memory used: {torch.cuda.memory_allocated()}") | - Adjust based on GPU memory<br>- Use gradient accumulation<br>- Find optimal batch size |
| Resource Management | Memory Usage | - OOM errors<br>- Slow training | pythondef print_memory_usage(): print(f"GPU memory: {torch.cuda.memory_allocated()/1e9}GB") | - Reduce batch size<br>- Use mixed precision<br>- Implement memory efficient loading |
| | Training Time | - Slow convergence<br>- Resource inefficiency | pythonstart_time = time.time()# Training loopprint(f"Time per epoch: {time.time() - start_time}") | - Use efficient data loading<br>- Optimize batch size<br>- Implement early stopping |
| Validation/Testing | Overfitting | - High train/low val accuracy<br>- Poor generalization | pythonprint(f"Train acc: {train_acc}")print(f"Val acc: {val_acc}") | - Add regularization<br>- Increase dropout<br>- Use data augmentation |
| | Evaluation Mode | - Inconsistent results<br>- Poor inference | pythonprint(f"Training: {model.training}")# Should use model.eval() for inference | - Use proper model modes<br>- Handle batch norm<br>- Consistent evaluation |

----
# Potential Issues with the Computational Graph

| Issue Category | Specific Problem | Symptoms | Debugging Steps | Solution Example |
|----------------|------------------|-----------|-----------------|------------------|
| Gradient Flow |
| | Vanishing Gradients | - Near-zero gradients in early layers<br>- Model not learning | pythondef check_gradients(model): for name, param in model.named_parameters(): if param.grad is not None: print(f"{name}: {param.grad.abs().mean()}") | - Use gradient clipping<br>- Implement residual connections<br>- Change activation functions |
| | Exploding Gradients | - NaN losses<br>- Large gradient values | pythontorch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) | - Add gradient clipping<br>- Reduce learning rate<br>- Check initialization |
| | Disconnected Graphs | - Some parameters not updating<br>- Partial learning | pythondef verify_graph_connectivity(loss): print(f"Grad fn chain: {loss.grad_fn}") | - Ensure all operations maintain gradients<br>- Check for accidental .detach() |
| Tensor Operations |
| | In-place Operations | - Backward pass errors<br>- "Leaf variable modified" error | python# Bad:x += 1# Good:x = x + 1 | - Avoid in-place operations<br>- Use out-of-place alternatives |
| | Device Mismatches | - Runtime errors<br>- CUDA errors | pythondef check_tensor_devices(model): for name, param in model.named_parameters(): print(f"{name}: {param.device}") | - Use .to(device) consistently<br>- Check all tensor operations |
| | Detached Tensors | - No gradients flowing<br>- Parts of model not learning | pythondef verify_requires_grad(model): for name, param in model.named_parameters(): print(f"{name}: {param.requires_grad}") | - Remove unnecessary .detach()<br>- Check requires_grad settings |
| Autograd Engine |
| | Broken Computational Paths | - Gradients not computed<br>- backward() errors | pythondef check_backward_hook(grad): print(f"Gradient shape: {grad.shape}") return grad | - Verify graph construction<br>- Add gradient hooks for debugging |
| | Mixed Precision Errors | - NaN losses<br>- Unstable training | pythonscaler = torch.cuda.amp.GradScaler()with torch.cuda.amp.autocast(): output = model(input) | - Use proper scaling<br>- Check dtype consistency |
| Loss Computation |
| | Zero/NaN Losses | - Model not learning<br>- Training instability | pythondef monitor_loss(loss): if torch.isnan(loss) or torch.isinf(loss): raise ValueError("Loss is NaN/Inf") | - Check loss function implementation<br>- Verify input normalization |
| | Wrong Reduction | - Incorrect gradient scaling<br>- Slow convergence | pythoncriterion = nn.CrossEntropyLoss(reduction='mean') | - Verify reduction method<br>- Adjust batch size accordingly |
| Memory Management |
| | Memory Leaks | - OOM errors<br>- Increasing memory usage | pythondef print_memory_usage(): print(torch.cuda.memory_allocated()/1e9) | - Clear cache between iterations<br>- Delete unused tensors |
| | Retained Graphs | - Memory accumulation<br>- Slow training | python# Clear after backwardloss.backward()optimizer.step()optimizer.zero_grad(set_to_none=True) | - Clear gradients properly<br>- Don't retain unnecessary graphs |
| Custom Layers |
| | Incorrect Forward/Backward | - Wrong gradients<br>- Training instability | pythonclass CustomLayer(nn.Module): def forward(self, x): self.save_for_backward(x) return output | - Implement custom autograd function<br>- Verify gradient computation |
| | Shape Mismatches | - Runtime errors<br>- Dimension errors | pythondef check_shapes(x): print(f"Input shape: {x.shape}") return x | - Add shape assertions<br>- Print intermediate shapes |

----
# For Developers: Fork

A debugger for running PyTorch transfer-learning &amp; fine-tuning jobs.


----
# How to Publish

[How to Publish a Python Package on PyPI with Twine and GitHub Actions](https://medium.com/@blackary/publishing-a-python-package-from-github-to-pypi-in-2024-a6fb8635d45d)


```
: zachcolinwolpe@gmail.com
: 06.12.2024
```
