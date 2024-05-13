import os
import glob

from pytorch_lightning.loggers import TensorBoardLogger

from detector.RNO_G import RnogStation23
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import NodeAsDOMTimeSeries, NodesAsPulses, NodeDefinition
from graphnet.models.gnn.dynedge import DynEdge
from graphnet.models.task.classification import BinaryClassificationTask
from graphnet.training.loss_functions import BinaryCrossEntropyLoss
from graphnet.models import StandardModel
from graphnet.utilities.logging import Logger

import data as io

graph_logger = Logger()

tb_logger = TensorBoardLogger("tb_logs",
                            default_hp_metric=False,
                            log_graph=True,
                            name = "alpha")

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

data_loader_kwargs = dict(batch_size=1, drop_last=True, num_workers = 0)
    
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
    target_labels = ["label"],
    loss_function = BinaryCrossEntropyLoss()
)

model = StandardModel(
    graph_definition = graph_definition,
    backbone = backbone,
    tasks = [task],
)

train_dataloader = datamodule.train_dataloader()
val_dataloader = datamodule.val_dataloader()

if __name__ == "__main__":
    model.fit(train_dataloader, val_dataloader = val_dataloader, 
              logger = tb_logger,
              max_epochs = 5, limit_train_batches = 128, limit_val_batches = 16)

    model.save("test.pth")