import numpy as np
import pandas as pd


if __name__ == "__main__":
    nr_of_channels = 15
    # feautures in addition to time charge e.g. antenna possitions
    nr_of_timesteps = 2048
    nr_of_extra_features = 4 # x, y, z, channel_id

    detector = pd.read_csv("rnog.csv")
    # placeholder for voltages
    time_traces_shape = np.ones((nr_of_channels, nr_of_timesteps))
    detector = pd.concat([pd.DataFrame(time_traces_shape, columns = [str(i) for i in np.arange(nr_of_timesteps)]), detector], axis = 1)
    print(detector)

    geometry_table = detector.set_index(['sensor_x', 'sensor_y', 'sensor_z'])
    print(geometry_table)
    geometry_table.to_parquet("rnog_station23_geometry_table.parquet")