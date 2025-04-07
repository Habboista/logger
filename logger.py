import json
import os
import matplotlib.pyplot as plt

class Logger:
    """Store and save training and validation information for each epoch.
    
    Parameter meta_info_dict stores gives information about keys
    representing info to be stored. For example, to use a default
    configuration pass:
        
        meta_info_dict = {
            'accuracy': dict(),
            'loss': dict(),
        }

    It allows to save and load information for each epoch and plotting.
    Each epoch is a dictionary of information.
    self.info is of type list[dict[str, list[float]]].
    self.info[n] stores information logged during epoch n.
    self.info[n][key] stores a list of floats logged during epoch n
    by the name of key, e.g. self.info[n]['loss'] = [0.23, 0.21, 0.18].

    If periodic_plot is True, a plot of each key of meta_info_dict
    will be stored in the log_dir_path. If log_info is called period
    times, the plot is saved. meta_info_dict could store also parameters
    for the plot, like the scale.
    """
    def __init__(self,
        meta_info_dict,
        log_dir_path,
        periodic_plot=True,
        period=100,
    ): 
        self.meta_info_dict = meta_info_dict
        self.log_dir_path = log_dir_path
        os.makedirs(log_dir_path, exist_ok=True)

        # Each epoch info will be an element of this list
        self.info = []

        # Internal variables for plotting
        self.periodic_plot = periodic_plot
        self.period = period
        self.plot_counter = 0
        self.is_loading = False

        # Load from log_dir_path
        self.load()

    
    def new_epoch(self):
        """Start a new epoch."""
        if not self.is_loading and self.periodic_plot:
            self.plot()
            
        info_dict = {k: [] for k in self.meta_info_dict.keys()}
        self.info.append(info_dict)


    def log_info(self, info_to_log):
        """Log information for the current epoch."""

        for k in self.meta_info_dict.keys():

            v = info_to_log[k]
            assert type(v) is float

            # Append to the last epoch
            self.info[-1][k].append(v)
        
        # If requested, plot on file the new data.
        if self.periodic_plot:
            self.plot_counter = (self.plot_counter + 1) % self.period
            if self.plot_counter == 0:
                self.plot()


    def plot(self):
        """For each key, plot all the values stored gathered through
        all the epoch in one chart."""

        # Gather values
        info = {k: [] for k in self.meta_info_dict}
        for epoch_info in self.info:
            for k, v in epoch_info.items():
                info[k].extend(v)

        # Plot
        for k, v in info.items():
            plt.title(k)
            if 'yscale' in self.meta_info_dict[k]:
                plt.yscale(self.meta_info_dict[k]['yscale'])
            else:
                plt.yscale('log')
            plt.plot(v)
            plt.savefig(os.path.join(self.log_dir_path, f'{k}.pdf'))
            plt.close()


    def save_epoch(self, epoch):
        """For each info key, save a json file in the directory
        specified by log_dir_path containing the info for
        the specified epoch.
        """

        for k in self.meta_info_dict.keys():
            path = os.path.join(
                self.log_dir_path,
                f'{k}_{epoch:02d}.json',
            )
            with open(path, 'w') as f:
                json.dump(self.info[epoch][k], f, indent=4)

    
    def save_last_epoch(self):
        self.save_epoch(len(self.info) - 1)


    def load_epoch(self, epoch):
        for k in self.meta_info_dict.keys():
            path = os.path.join(
                self.log_dir_path,
                f'{k}_{epoch:02d}.json',
            )
            if os.path.exists(path):
                with open(path, 'r') as f:
                    key_info = json.load(f)
                self.info[-1][k] = key_info
            else:
                raise FileNotFoundError


    def load(self):
        self.is_loading = True
        epoch = 0
        while True:
            self.new_epoch()
            try:
                self.load_epoch(epoch)
                epoch = epoch + 1
            except FileNotFoundError:
                del self.info[-1]
                self.is_loading = False
                return


    def get_key_alias(self, key):
        """Get string alias to use instead of key for printing."""
        try:
            alias = self.meta_info_dict[k]['alias']
        except KeyError:
            alias = k
        return alias 


    def print_epoch_summary(self, epoch):
        print(f"\nEpoch {epoch:02d} summary:")
        for k, v in self.info[epoch].items():
            print(f"\t{self.get_key_alias(k)}: {sum(v) / len(v):.3f}")
        print() # Blank line


    def save_epoch_summary(self, epoch):
        summary = dict()
        for k, v in self.info[epoch].items():
            summary[k] = sum(v) / len(v)
        path = os.path.join(
            self.dir_path,
            f'__summary_{k}_{len(self.info) - epoch:02d}.json')
        with open(path, 'w') as f:
            json.dump(summary, f, indent=4)
    

    def save_last_epoch_summary(self):
        self.save_epoch_summary(-1)


    def print_last_epoch_summary(self):
        self.print_epoch_summary(-1)
