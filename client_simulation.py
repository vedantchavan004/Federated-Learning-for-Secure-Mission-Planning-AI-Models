import numpy as np

def generate_mission_data(samples=200):
    # Simulate mission features: [flight_time, altitude, threat_level, sensor_data, comm_status]
    X = np.random.rand(samples, 5)
    y = (X[:, 2] > 0.6).astype(int)  # binary threat classification
    return X, y

def get_clients_data(num_clients=3):
    return [generate_mission_data() for _ in range(num_clients)]
