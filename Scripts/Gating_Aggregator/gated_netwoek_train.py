# # Gated Network Training

def train_gated_combination_model(X1_t, X2_t, X3_t, X4_t, tensor_score_o,
                                  X1_val, X2_val, X3_val, X4_val, tensor_score_val,
                                  epochs=120, batch_size=32, learning_rate=0.001, weight_decay=1e-5):
    """
    Trains the GatedCombination model with training and validation data, using ReduceLROnPlateau scheduler.
    """

    # Create datasets and DataLoaders
    train_dataset = TensorDataset(X1_t, X2_t, X3_t, X4_t, tensor_score_o)
    val_dataset = TensorDataset(X1_val, X2_val, X3_val, X4_val, tensor_score_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GatedCombination(X1_t.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Use ReduceLROnPlateau scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    criterion = WeightedBCELoss(pos_weight=positive_weight).to(device)

    train_losses, val_losses = [], []

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        total_train_loss, y_true_train, y_pred_train = 0.0, [], []

        for batch_X1, batch_X2, batch_X3, batch_X4, batch_y in train_loader:
            batch_X1, batch_X2, batch_X3, batch_X4, batch_y = (batch_X1.to(device), batch_X2.to(device),
                                                               batch_X3.to(device), batch_X4.to(device), batch_y.to(device))
            optimizer.zero_grad()
            outputs = model(batch_X1, batch_X2, batch_X3, batch_X4)
            loss = criterion(outputs, batch_y.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            y_true_train.extend(batch_y.cpu().numpy())
            y_pred_train.extend((outputs > 0.2).float().cpu().numpy())

        train_f1 = f1_score(y_true_train, y_pred_train)
        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        total_val_loss, y_true_val, y_pred_val = 0.0, [], []
        with torch.no_grad():
            for batch_X1, batch_X2, batch_X3, batch_X4, batch_y in val_loader:
                batch_X1, batch_X2, batch_X3, batch_X4, batch_y = (batch_X1.to(device), batch_X2.to(device),
                                                                   batch_X3.to(device), batch_X4.to(device), batch_y.to(device))
                outputs = model(batch_X1, batch_X2, batch_X3, batch_X4)
                val_loss = criterion(outputs, batch_y.unsqueeze(1).float())
                total_val_loss += val_loss.item()
                y_true_val.extend(batch_y.cpu().numpy())
                y_pred_val.extend((outputs > 0.4).float().cpu().numpy())

        val_f1 = f1_score(y_true_val, y_pred_val)
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Step the scheduler with validation loss
        scheduler.step(avg_val_loss)

        # Print training and validation metrics
        print(f"Epoch [{epoch + 1}/{epochs}] Training Loss: {train_loss:.4f}, F1 Score: {train_f1:.4f} | "
              f"Validation Loss: {avg_val_loss:.4f}, F1 Score: {val_f1:.4f}")

    end_time = time.time()

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss", marker='o')
    plt.plot(val_losses, label="Validation Loss", marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    print(f"Training complete! Total time: {end_time - start_time:.2f} seconds")
    return model
