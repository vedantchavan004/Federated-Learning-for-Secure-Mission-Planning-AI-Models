from client_simulation import get_clients_data
from model import create_model
import numpy as np

def federated_train(rounds=5, clients=3):
    global_model = create_model()

    for round in range(rounds):
        print(f"\n--- Federated Training Round {round+1} ---")
        client_weights = []

        for i, (X, y) in enumerate(get_clients_data(clients)):
            print(f"Client {i+1} training on local data...")
            local_model = create_model()
            local_model.set_weights(global_model.get_weights())
            local_model.fit(X, y, epochs=3, verbose=0)
            client_weights.append(local_model.get_weights())
            loss, acc = local_model.evaluate(X, y, verbose=0)
            print(f"Client {i+1} accuracy: {acc:.2f}")

        # Secure aggregation: average weights
        avg_weights = [np.mean(layer, axis=0) for layer in zip(*client_weights)]
        global_model.set_weights(avg_weights)

    global_model.save('mission_model.h5')
    print("\nFederated training completed. Global model saved as mission_model.h5")

if __name__ == "__main__":
    federated_train()
