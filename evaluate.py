import glob
import pandas as pd

from graphnet.models import Model
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import NodesAsPulses

from detector.RNO_G import RnogStation23
import data as io

if __name__ == "__main__":

    datamodule = io.ClassifierData

    sim_files = glob.glob("/home/ruben/Documents/RNO-G_eventfiltering/data/simulations/16-16.7/event*.npz")
    lt_files = glob.glob("/home/ruben/Documents/RNO-G_eventfiltering/data/lt/run*.npz")
    ft_files = glob.glob("/home/ruben/Documents/RNO-G_eventfiltering/data/ft/run*.npz")

    # data
    # ----------

    detector = RnogStation23()
    graph_definition = KNNGraph(
        detector = detector,
        node_definition = NodesAsPulses(),
        input_feature_names = [str(i) for i in range(2048)],
        nb_nearest_neighbours = 4, # 4 to get the phased array connected , might look into different nr of edges
    )

    data_loader_kwargs = dict(shuffle=False, batch_size=1, drop_last=True, num_workers = 0)
        
    dataset_kwargs = dict(normal_pos=2, normal_width=2, signal_fraction=0.5, roll_n_samples=250, custom_channel_functions=[])

    datamodule = io.ClassifierData(lt_files, ft_files, sim_files, [0.75, 0.125, 0.125], representation = "time_trace",
                                    dataset_kwargs=dataset_kwargs, dataloader_kwargs=data_loader_kwargs, graph_definition = graph_definition)

    test_dataloader = datamodule.test_dataloader()

    model = Model.load("test.pth")

    prediction = model.predict_as_dataframe(test_dataloader)
    print(prediction)
    prediction.to_csv("./prediction.csv")