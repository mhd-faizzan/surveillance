import torch
import torch.nn as nn

from src.models.lstm_classifier import get_device


class Autoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, sequence_length: int):
        super(Autoencoder, self).__init__()

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=input_size,
            num_layers=1,
            batch_first=True
        )

        self.sequence_length = sequence_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encode
        _, (hidden, _) = self.encoder(x)

        # repeat hidden state across sequence length for decoder input
        decoder_input = hidden.squeeze(0).unsqueeze(1).repeat(1, self.sequence_length, 1)

        # decode
        reconstruction, _ = self.decoder(decoder_input)
        return reconstruction


def train_autoencoder(normal_sequences: torch.Tensor, config: dict) -> Autoencoder:
    device = get_device()
    logger = __import__('logging').getLogger(__name__)

    model = Autoencoder(
        input_size=config["features"]["input_size"],
        hidden_size=config["model"]["autoencoder"]["hidden_size"],
        sequence_length=config["features"]["sequence_length"]
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["model"]["autoencoder"]["learning_rate"]
    )
    criterion = nn.MSELoss()

    normal_sequences = normal_sequences.to(device)

    for epoch in range(config["model"]["autoencoder"]["epochs"]):
        model.train()
        optimizer.zero_grad()
        reconstruction = model(normal_sequences)
        loss = criterion(reconstruction, normal_sequences)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            logger.info("Autoencoder epoch %d/%d loss: %.6f",
                epoch + 1, config["model"]["autoencoder"]["epochs"], loss.item())

    __import__('os').makedirs("models", exist_ok=True)
    __import__('torch').save(model.state_dict(), config["model"]["autoencoder"]["save_path"])
    logger.info("Saved autoencoder to %s", config["model"]["autoencoder"]["save_path"])

    return model


def get_anomaly_score(model: Autoencoder, sequence: torch.Tensor) -> float:
    """
    Returns reconstruction error for a sequence.
    High error = anomaly (pattern not seen during training).
    """
    model.eval()
    device = get_device()
    sequence = sequence.to(device)

    with torch.no_grad():
        reconstruction = model(sequence.unsqueeze(0))
        score = nn.MSELoss()(reconstruction, sequence.unsqueeze(0)).item()

    return score