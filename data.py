import sys
import logging 
logging.basicConfig()
from typing import List

import numpy as np
import torch
from torch_geometric.data import Data, Batch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from scipy import signal

from graphnet.models.graphs import KNNGraph

nr_of_timesteps = 2048


def baseline_correction(wfs, n_bins=128, func=np.median):
    """
    Simple baseline correction function. Determines baseline in discrete chuncks of "n_bins" with
    the function specified (i.e., mean or median).
    
    Parameters
    ----------
    
    wfs: np.array(n_events, n_channels, n_samples)
        Waveforms of several events/channels.
        
    n_bins: int
        Number of samples/bins in one "chunck". If None, calculate median/mean over entire trace. (Default: 128)
        
    func: np.mean or np.median
        Function to calculate pedestal
    
    Returns
    -------
    
    wfs_corrected: np.array(n_events, n_channels, n_samples)
        Baseline/pedestal corrected waveforms
    """
     
    # Example: Get baselines in chunks of 128 bins
    # wfs in (n_events, n_channels, 2048)
    # np.split -> (16, n_events, n_channels, 128) each waveform split in 16 chuncks
    # func -> (16, n_events, n_channels) pedestal for each chunck
    if n_bins is not None:
        baseline_values = func(np.split(wfs, 2048 // n_bins, axis=-1), axis=-1)

        # np.repeat -> (2048, n_events, n_channels) concatenate the 16 chuncks to one baseline
        baseline_traces = np.repeat(baseline_values, n_bins % 2048, axis=0)
    else:
        baseline_values = func(wfs, axis=-1)
        # np.repeat -> (2048, n_events, n_channels) concatenate the 16 chuncks to one baseline
        baseline_traces = np.repeat(baseline_values, 2048, axis=0)
             
    # np.moveaxis -> (n_events, n_channels, 2048)
    baseline_traces = np.moveaxis(baseline_traces, 0, -1)
     
    return wfs - baseline_traces

def get_splited_paths(paths, fractions):
    return np.split(paths, get_split(len(paths), fractions))

def get_split(n, fractions):
    splits = np.array([np.floor(n * f) for f in fractions], dtype=int)
    
    if n < len(fractions):
        sys.exit("You did not provide sufficient data files.")
    
    if np.any(splits == 0):
        # raise ValueError("Have an empty split")
        
        # that code is stupid and can go wrong easily
        splits[np.argmax(splits)] -= np.sum(splits == 0)
        splits[splits == 0] += 1
        
    if np.sum(splits) < n:
        splits[0] += n - np.sum(splits)
        
    csplits = np.cumsum(splits)[:-1]
    print(f"Split set of length {n} into a set of {splits} ({csplits})")
    
    return csplits

def calibration(wfs):
    
    wfs = baseline_correction(wfs)
    
    wfs *= 2.5 / 4095  # 2.5 V / (2^12 - 1)
    
    return wfs


def add_scaled_signal_normal(noise_trace, signal_trace, normal_pos=1, normal_width=1, channels=[0, 1, 2, 3]):
    
    # Take mean of rms / max amplitude over PA channel
    noise_rms = np.mean(np.std(noise_trace, axis=1)[channels])
    max_amp = np.amax(signal_trace[channels])
    
    scale = abs(np.random.normal(0, normal_width)) + normal_pos  # truncated gaussian
    signal_trace *= scale * noise_rms / max_amp            
    
    return noise_trace + signal_trace, scale

def time_trace_to_stft(wfs, sampling_rate = 3.2e9, nperseg = 128):
    """converts time traces to a spectrogram representation"""
    freqs, times, stft = signal.stft(wfs, fs = sampling_rate, nperseg = nperseg)
    return freqs, times, stft
    
def collate_fn(graphs: List[Data]) -> Batch:
    """COPIED FROM GRAPHNET SRC COUDE under training.utils
    Remove graphs with less than two DOM hits.

    Should not occur in "production".
    """
    graphs = [g for g in graphs if g.n_pulses > 1]
    return Batch.from_data_list(graphs)

class ClassifierData(pl.core.LightningDataModule):

    def __init__(self, lt_noise_files, ft_noise_files, signal_files, data_split, representation, graph_definition, dataset_kwargs={}, dataloader_kwargs={}):
        super().__init__()
        
        self.graph_definition = graph_definition
        self.dataloader_kwargs = dataloader_kwargs
        self.dataset_kwargs = dataset_kwargs
        self.py_logger = logging.getLogger(f"ClassifierData")

        if representation == "time_trace":
            self.py_logger.info("Using time trace representation")
            self.DatasetClass = TimeTraceDataset
        elif representation == "stft":
            self.py_logger.info("Using stft data representation")
            self.DatasetClass = STFTDataset
        else:
            self.py_logger.error("Unknown representation")
            sys.exit()

        train_noise_paths, valid_noise_paths, test_noise_paths = \
            get_splited_paths(lt_noise_files, data_split)
        
        train_signal_noise_paths, valid_signal_noise_paths, test_signal_noise_paths = \
            get_splited_paths(ft_noise_files, data_split)
        
        train_signal_paths, valid_signal_paths, test_signal_paths = \
            get_splited_paths(signal_files, data_split)
        
        self.train_paths = dict(noise_file_list=train_noise_paths, signal_file_list=train_signal_paths,
                                signal_noise_file_list=train_signal_noise_paths)

        self.valid_paths = dict(noise_file_list=valid_noise_paths, signal_file_list=valid_signal_paths,
                                signal_noise_file_list=valid_signal_noise_paths)

        self.test_paths = dict(noise_file_list=test_noise_paths, signal_file_list=test_signal_paths,
                                signal_noise_file_list=test_signal_noise_paths)

        
        self.save_hyperparameters() # option tells lightning to save used hyperparameters in the checkpoints

    
    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        self.dataset_train = self.DatasetClass(
            **self.train_paths, **self.dataset_kwargs, label="-Train",
            graph_definition = self.graph_definition)
        return DataLoader(self.dataset_train, **self.dataloader_kwargs,
                          collate_fn=collate_fn)#, persistent_workers=True, prefetch_factor=2)

    def val_dataloader(self):
        self.dataset_valid = self.DatasetClass(
            **self.valid_paths, **self.dataset_kwargs, label="-Valid",
            graph_definition = self.graph_definition)
        return DataLoader(self.dataset_valid, **self.dataloader_kwargs,
                          collate_fn=collate_fn)#, persistent_workers=True, prefetch_factor=2)
    
    def test_dataloader(self, l_kwargs={}):
        self.dataloader_kwargs.update(l_kwargs)
        self.dataset_test = self.DatasetClass(
            **self.test_paths, **self.dataset_kwargs, label="-Test",
            graph_definition = self.graph_definition)
        return DataLoader(self.dataset_test, **self.dataloader_kwargs,
                          collate_fn=collate_fn)#, persistent_workers=True, prefetch_factor=2)

   # teardown is used to clean up after the run is finished
   # this function is called at the end of train + validate, validate, test or predict
   # this function is run on every accelerator (e.g. on every gpu when training on several gpu's)
   # specific use: to avoid loading both train and test dataset in memory at same time
    def teardown(self, stage):
        # This function might be called several times
        if str(stage) == "TrainerFn.FITTING":
            self.dataset_train.summary("Train")
            self.dataset_valid.summary("Validation")
            self.dataset_train = None
            self.dataset_valid = None
        elif str(stage) == "TrainerFn.TESTING":
            self.dataset_test.summary("Testing")
            self.dataset_test = None


class FractionalSignalDatasetIter(torch.utils.data.IterableDataset):
        
    def __init__(self, noise_file_list, signal_file_list=None, signal_noise_file_list=None,
                 normal_pos=2, normal_width=2, 
                 signal_fraction=0.5, label="", roll_n_samples=0,
                 custom_channel_functions = [],
                 graph_definition = KNNGraph):
        """
        Base dataset class agnostic to what represenatation is chosen for the feature dataset
        This class includes all functionality related to constructing signal + noise convolutions, calibration,...
        """
        super().__init__()

        self.logger = logging.getLogger(f"FractionalSignalDataset{label}")
        self.logger.setLevel(logging.DEBUG)

        self.signal_fraction = signal_fraction
        self.normal_pos = normal_pos
        self.normal_width = normal_width
        self.roll_n_samples = roll_n_samples

        self.custom_channel_functions = custom_channel_functions

        self._num_noise = 0
        self._num_sig = 0

        self.graph_definition = graph_definition

        
        # Noise defines the total amount of data
        self._noise_wfs = None
        self.noise_file_list = noise_file_list

        if signal_file_list is not None:
            self.set_signals(signal_file_list)
            
        if signal_noise_file_list is not None:
            self.set_signal_noise(signal_noise_file_list)
        
        
    def __iter__(self):
        np.random.shuffle(self.noise_file_list)
        for path in self.noise_file_listexi:
            with np.load(path) as f:
                noise_traces = f["wfs"] # shape (events, channels, 2048)
            
            noise_traces = self._noise_preprocessing(noise_traces)
            for noise in noise_traces:
                self._num_noise += 1

                # whether to inject pure noise or noise + signal
                while np.random.random_sample() < self.signal_fraction:
                    self._num_sig += 1
                    signal_noise, scale = self.get_waveform_with_signal()
                    signal_noise = self._add_custom_channels(signal_noise, custom_functions = self.custom_channel_functions)
                    signal_noise = self._transform(signal_noise)
                    signal_noise, _ =  self._normalization_func(signal_noise)
                    signal_noise = self._waveform_processing(signal_noise)
                    signal_noise = self.convert_to_graph(signal_noise, 1, graph_definition = self.graph_definition)
                    yield signal_noise # traces, label

                noise = self._add_custom_channels(noise, custom_functions = self.custom_channel_functions)
                noise = self._transform(noise)
                noise, _ =  self._normalization_func(noise)
                noise = self._waveform_processing(noise)
                noise = self.convert_to_graph(noise, 0, graph_definition = self.graph_definition)  
                yield noise  # traces, label
 
            
    def set_signals(self, signal_file_list):
        
        # Read signals
        self._signal_wfs = []
        for path in signal_file_list:
            with np.load(path) as f:
                self._signal_wfs.append(f["wfs"].reshape(-1, 15, 2048))
        
        self._signal_wfs = np.vstack(self._signal_wfs, dtype=np.float32) # shape = (total_nr_of_runs, channels, samples)

        mask = np.array([np.any(np.isnan(ele)) for ele in self._signal_wfs])
        
        if np.any(mask):
            self._signal_wfs = self._signal_wfs[~mask]
            self.logger.warn(f"Found {np.sum(mask)} events with a NaN trace (out of {len(mask)}). Remove those")
        
        self._nsignals = len(self._signal_wfs)
        self._simulations_used = np.zeros(self._nsignals)  # for bookkeeping
        self.logger.info(f"Read {self._nsignals} signal events. Will inject them with a fraction of {self.signal_fraction:.3f}")
        if self._noise_wfs is not None:
            self.logger.info(f"On-average use of each signal event: {len(self._noise_wfs) / self._nsignals * self.signal_fraction:.3f}")


    def set_signal_noise(self, paths):
                
        # Read signals
        self._signal_noise_wfs = []
        for path in paths:
            with np.load(path) as f:
                self._signal_noise_wfs.append(f["wfs"])
        self._signal_noise_wfs = np.vstack(self._signal_noise_wfs)
        self._n_signal_noise = len(self._signal_noise_wfs)

        self._signal_noise_used = np.zeros(self._n_signal_noise)  # for bookkeeping

        # uses too much memory - do in getter -> done in _noise_preprocessing function
        # self._signal_noise_wfs = np.array(calibration(self._signal_noise_wfs), dtype=np.float32)
    
        if np.any(np.isnan(self._signal_noise_wfs)):
            sys.exit("Found Nan in signal noise data. Stop")
            
        self.logger.info(f"Read {self._n_signal_noise} signal noise events")

  
    def get_signal_traces(self):
        """ Get noise-less neutrino signals """
        random_idx = np.random.random_integers(0, self._nsignals - 1)
        sig = self._signal_wfs[random_idx] / 1000 # in units of mV
        self._simulations_used[random_idx] += 1
        
        if self.roll_n_samples != 0:
            sig = np.roll(sig, -self.roll_n_samples)
            sig[:, -self.roll_n_samples:] = 0  # set this to 0 (otherwise unphysical)
        
        return sig
    

    def get_signal_noise_traces(self):
        """ Get (measured) noise traces to be added to simulated signals (i.e., forced triggers) """
        random_idx = np.random.random_integers(0, self._n_signal_noise - 1)
        noise = self._noise_preprocessing(self._signal_noise_wfs[random_idx])
        self._signal_noise_used[random_idx] += 1
        
        return noise
    

    def get_waveform_with_signal(self):

        sig = self.get_signal_traces()
        noise = self.get_signal_noise_traces()
        wf, scale = add_scaled_signal_normal(noise, sig, self.normal_pos, self.normal_width)         

        return wf, scale
    

    def get_waveform_with_signal_with_proc(self, channels=[0, 1, 2, 3]):
        """ function to be used in evaluation, not in model training """
        sig = self.get_signal_traces()
        noise = self.get_signal_noise_traces()
        wf, scale = add_scaled_signal_normal(noise, sig, self.normal_pos, self.normal_width)
        
        snr = 0
        for channel in channels:
            idx = np.argmax(np.abs(sig[channel]))
            rms = np.std(wf[channel])
            ampl = np.amax(np.abs(wf[channel, max(idx - 10, 0):min(idx + 10, 2047)]))
            snr += ampl / rms
        snr /= len(channels)
        
        if self._normalization_func is not None:
            wf, norm =  self._normalization_func(wf)
            sig = sig / norm
            
        wf = self._waveform_processing(wf)
        sig = self._waveform_processing(sig)

        return wf, scale, sig, snr
    
    
    def _noise_preprocessing(self, wfs):
        return np.array(calibration(wfs), dtype=np.float32)


    def _add_custom_channels(self, wf, custom_functions = []):
        """
        Function to add custom channels as a function of previous channels
        e.g. an extra channel made up of the sum of the data of the four phased array channels:
             channel_extra = sum_(ch0-3))(data)
        
        Parameters:
        -----------
            wf: np.array(events, channels, samples) OR np.array(channels, samples)
                input waveform array

            custom_functions : list
                list of functions corresponding to extra custom channels
                e.g. [func1, func2] will add 2 extra custom channels
                !these functions need to be vectorized!
        Returns:
        -------- 
            wf: np.array(events, channels', ...)
                the waveform array with extra custom channels 
        """
        if not custom_functions:
            return wf
        custom_channel_array = []
        for f in custom_functions:
            custom_channel = f(wf)
            custom_channel_array.append(custom_channel)
        custom_channel_array = np.array(custom_channel_array)

        wf = np.concatenate((wf, custom_channel_array), axis = -2)

        return wf



    def _transform(self, wf):
        pass 

    def _normalization_func(self, wfs):
        pass

    def _waveform_processing(self, wfs):
        pass

    def convert_to_graph(self, wf, label, graph_definition):
        truth_dicts = {"label" : label}
        # assumes wf shape = (2048, 15) (or (1, 2048, 15))
        wf = np.squeeze(wf).T

        graph = graph_definition(input_features = wf, input_feature_names = [str(i) for i in range(nr_of_timesteps)], truth_dicts = [truth_dicts]) 
        return graph        


    def summary(self, label):
        argsort = np.flip(np.argsort(self._simulations_used))
        argsort_noise = np.flip(np.argsort(self._signal_noise_used))
        
        # with num_workers > 1 this counters are 0 ...
        signal_fraction = np.nan
        if self._num_noise + self._num_sig:
            signal_fraction = self._num_sig / (self._num_noise + self._num_sig)

        self.logger.info(f"{label}:\n"
                        f"\tTrained/validate with {self._num_noise} noise events and {self._num_sig} signal events (of {len(self._simulations_used)} unique events)"
                        f" ({signal_fraction:.3f})\n"
                        f"\t5 most used simulations (frequency): {self._simulations_used[argsort][:5]}\n"
                        f"\t5 most used noise added to signal (frequency): {self._signal_noise_used[argsort_noise][:5]}\n"
                        f"\ttotal noise events added to signals used in training: {np.sum(self._signal_noise_used)} (of {len(self._signal_noise_used)} unique events)")


class TimeTraceDataset(FractionalSignalDatasetIter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _transform(self, wf):
        return wf

    def _normalization_func(self, wfs, channel_to_use = [0, 1, 2, 3]):
        if wfs.ndim == 3:
            # assume shape (n_events, n_channel, n_sample)
            # max of max of selected channels)
            max_ampls = np.amax(np.amax(np.abs(wfs[:, channel_to_use]), axis=-1), axis=-1)
            return wfs / max_ampls[:, None, None], max_ampls
        elif wfs.ndim == 2:
            max_ampls = np.amax(np.amax(np.abs(wfs[channel_to_use]), axis=-1))
            return wfs / max_ampls, max_ampls
        else:
            self.logger.error("Undefined dimension")
            sys.exit()

    def _waveform_processing(self, wf):
        # expects wf to be shape (channel=15, sample=2048)
        # creating the array (nfilter=1, n_samples, n_channels) (will be expanded by n_events dimension in training)
        if wf.ndim == 3:
            return wf.T.reshape(-1, 1, 2048, 15)
        else:
            return wf.T.reshape(1, 2048, 15)

class STFTDataset(FractionalSignalDatasetIter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _transform(self, wf, nperseg = 128, nfft = None):
        wf_freqs, wf_times, stft = signal.stft(wf, fs=3.2e9, nperseg=nperseg, nfft=nfft)
        return np.abs(stft) # shape (channels, freq, times)


    def _normalization_func(self, wfs, channel_to_use = [0, 1, 2, 3]):
        """
        parameters
        ---------
        wfs:
            shape (events, channels, freqs, times) or (channels, freqs, times) 
        """
        # if shape (events, channels, freqs, times)
        if wfs.ndim == 4:
            self.logger.info("shape of input is (events, channels, freqs, times)")
            max_ampls = np.max(np.abs(wfs[:, channel_to_use]), axis = (1, 2, 3)) # shape (events)
            return wfs / max_ampls[:, None, None, None], max_ampls

        # if shape (channels, freqs, times)
        elif wfs.ndim == 3:
            self.logger.info("shape of input is (channels, freqs, times)")
            max_ampls = np.max(np.abs(wfs), axis = (0, 1, 2))
            return wfs/ max_ampls, max_ampls

        else:
            self.logger.error("Undefined dimension")
            sys.exit()

    # This processiing function is for use with a classic 2d kernel 
    # (this function does nothing and is a placeholder since the wfs shape is already correct)
    #-----------------------------------------------------------------------------------------
    def _waveform_processing(self, wf):
        """
        function to reshape the feature inputs and include a filter dimension
        parameters
        ----------
        wf:
            shape (events, channels, freqs, times) or (channels, freqs, times)

        returns
        -------
        wf:
            shape (events, channels, freqs, times) or (channels, freqs, times)
        """
        return wf