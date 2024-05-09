import glob

from RNO_G import RnogStation23
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import NodeAsDOMTimeSeries, NodesAsPulses
from graphnet.models.gnn.dynedge import DynEdge
from graphnet.models.task.classification import BinaryClassificationTask
from graphnet.training.loss_functions import BinaryCrossEntropyLoss
from graphnet.models import StandardModel
from graphnet.utilities.logging import Logger

import data as io

logger = Logger()

sim_files = glob.glob("/CECI/proj/rnog/ml_data/simulations/without_noise/*/event*.npz")
lt_files = glob.glob("/CECI/proj/rnog/ml_data/station23_data_npz/lt/run*.npz")
ft_files = glob.glob("/CECI/proj/rnog/ml_data/station23_data_npz/ft/run*.npz")

# data
# ----------

detector = RnogStation23()
graph_definition = KNNGraph(
    detector = detector,
    node_definition = NodeDefinition(
        input_feature_names = detector.feature_map().keys()
    ),
    nb_nearest_neighbours = 4, # 4 to get the phased array connected , might look into different nr of edges
)

data_loader_kwargs = dict(shuffle=False, batch_size=16, drop_last=True, num_workers = 5)
    
dataset_kwargs = dict(normal_pos=2, normal_width=2, signal_fraction=0.5, roll_n_samples=250, custom_channel_functions=[])

datamodule = io.ClassifierData(lt_files, ft_files, sim_files, [0.75, 0.125, 0.125], representation = "time_trace",
                                dataset_kwargs=dataset_kwargs, dataloader_kwargs=data_loader_kwargs, graph_definition = graph_definition)

# model
# -----


backbone = DynEdge(
    nb_inputs = graph_definition.nb_outputs,
    global_pooling_schemes  = ["min", "mean", "max"],
)

task = BinaryClassificationTask(
    hidden_size = backbone.nb_outputs,
    target_labels = ["noise", "signal"],
    loss_function = BinaryCrossEntropyLoss()
)

model = StandardModel(
    graph_definition = graph_definition,
    backbone = backbone,
    tasks = [task],
)

train_dataloader = datamodule.train_dataloader()

if __name__ == "__main__":
    model.fit(train_dataloader, max_epochs = 1)

    model.save("test.pth")