from typing import Dict, Callable
import torch
from graphnet.models.detector import Detector

nr_of_timesteps = 2048

class RnogStation23(Detector):
    geometry_table_path = "/home/rcamphyn/graphnet/graphnet_rc/data/geometry_tables/rno-g/station23.parquet"
    xyz = ["sensor_x", "sensor_y", "sensor_z"]
    string_id_column = "string_id"
    sensor_id_column = "sensor_id"

    def feature_map(self) -> Dict[str, Callable]:
        feature_map = {
            "sensor_x": self._sensor_xyz,
            "sensor_y": self._sensor_xyz,
            "sensor_z": self._sensor_xyz
            }
        voltages = {
            str(i) : self._sensor_voltage for i in range(nr_of_timesteps)
            }
        feature_map.update(voltages)
        return feature_map

    def _sensor_xyz(self, x: torch.tensor) -> torch.tensor:
        return x / 500.0
    
    def _sensor_voltage(self, x : torch.tensor) -> torch.tensor:
        return x