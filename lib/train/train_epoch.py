"""Module providingFunction printing python version."""

def train_epoch():
    model = trainer.get_model()

    trainer.train_logger.log_start()
    for epoch in range(trainer.start_epoch, trainer.epochs + 1):
        epoch_loss = 0.

        for batch_index, batch in enumerate(trainer.dataloader, 1):
            inputs, labels = batch[0].to(trainer.device), batch[1].to(trainer.device)

            trainer.hyp_loader.optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            logger.log_batch(batch_index, loss.item())

            epoch_loss += loss.item()

        trainer.train_logger.log_epoch_end(epoch, epoch_loss)

        if trainer.train_logger.checkpoint_save_type is not None:
            trainer.train_logger.save_checkpoint(epoch, model.state_dict(), optimizer.state_dict())
