
##  FLOW DIAGRAM
 `train_model` function down **step-by-step**, 

```
FOR epoch in range(num_epochs):
    SET model to training mode
    INIT loss to 0
    FOR each batch (src, tgt) in dataloader:
        Move data to device
        Zero out gradients
        Forward pass through model → Get output
        Calculate loss using output and target
        Backpropagate
        Clip gradients
        Update weights using optimizer
        Track loss
    END
    Log average loss
SAVE model
```

