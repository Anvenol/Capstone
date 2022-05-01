import os
import json
import sys
import tqdm
import logging


class Params:
    """
    Class that loads hyperparameters from a json file as a dictionary (also support nested dicts).
    Example:
    params = Params(json_path)
    # access key-value pairs
    params.learning_rate
    params['learning_rate']
    # change the value of learning_rate in params
    params.learning_rate = 0.5
    params['learning_rate'] = 0.5
    # print params
    print(params)
    # combine two json files
    params.update(Params(json_path2))
    """

    def __init__(self, json_path=None):
        if json_path is not None and os.path.isfile(json_path):
            with open(json_path) as f:
                params = json.load(f)
                self.__dict__.update(params)
        else:
            self.__dict__ = {}

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii=False)

    def update(self, json_path=None, params=None):
        """Loads parameters from json file"""
        if json_path is not None:
            with open(json_path) as f:
                params = json.load(f)
                self.__dict__.update(params)
        elif params is not None:
            self.__dict__.update(vars(params))
        else:
            raise Exception('One of json_path and params must be provided in Params.update()!')

    def __contains__(self, item):
        return item in self.__dict__

    def __getitem__(self, key):
        return getattr(self, str(key))

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __str__(self):
        return json.dumps(self.__dict__, sort_keys=True, indent=4, ensure_ascii=False)


def set_logger(log_path):
    """
    Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    logging.info('Starting training...')
    Args:
        log_path: (string) where to log
    """
    _logger = logging.getLogger('RMD')
    _logger.setLevel(logging.INFO)

    fmt = logging.Formatter('[%(asctime)s] %(name)s: %(message)s', '%m/%d %H:%M:%S')

    class TqdmHandler(logging.StreamHandler):
        def __init__(self, formatter):
            logging.StreamHandler.__init__(self)
            self.setFormatter(formatter)
            self.setStream(tqdm)

        def emit(self, record):
            msg = self.format(record)
            tqdm.tqdm.write(msg)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    _logger.addHandler(file_handler)
    _logger.addHandler(TqdmHandler(fmt))

    # https://stackoverflow.com/questions/6234405/logging-uncaught-exceptions-in-python?noredirect=1&lq=1
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            _logger.info('=*=*=*= Keyboard interrupt =*=*=*=')
            return

        _logger.error("Exception --->", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception


def save_dict_to_json(d, json_path):
    """
    Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def model_list():
    """
    List all available models found under ./model.
    """
    files = os.listdir('./models')
    files = [name.replace('.py', '') for name in files if name.endswith('.py')]
    return files


def plot_all_loss(loss_summary, save_name, plot_title, location='./figures/'):
    gaussian_window_size = 3
    loss_cum = cum_by_axis1(loss_summary)
    loss_cum = gaussian_filter1d(loss_cum, gaussian_window_size, axis=0)
    num_loss = loss_cum.shape[1]
    color_list = ['b', 'r', 'g', 'm', 'y']
    loss_list = ['loss_1', 'loss_2', 'loss_3', 'loss_4', 'loss_5']
    f = plt.figure()
    plt.title(plot_title)
    num_batches = loss_cum.shape[0]
    if num_batches > 10000:
        pack_size = num_batches // 10000
        x = np.arange(num_batches)[0:num_batches:pack_size]
        plt.fill_between(x, 0, loss_cum[0:num_batches:pack_size, 0], color='b', alpha=0.2, label='loss_1')
        for i in range(num_loss - 1):
            plt.fill_between(x, loss_cum[0:num_batches:pack_size, i], loss_cum[0:num_batches:pack_size, i + 1],
                             color=color_list[i + 1], alpha=0.2, label=loss_list[i + 1])
    else:
        x = np.arange(num_batches)
        plt.fill_between(x, 0, loss_cum[:, 0], color='b', alpha=0.2, label='loss_1')
        for i in range(num_loss - 1):
            plt.fill_between(x, loss_cum[:, i], loss_cum[:, i + 1], color=color_list[i + 1], alpha=0.2,
                             label=loss_list[i + 1])
    plt.yscale('log')
    plt.legend()
    f.savefig(os.path.join(location, save_name + '_summary.png'))
    plt.close()


def plot_all_epoch(variable1, variable2, save_name, plot_title, location='./figures/', plot_start=0):
    num_samples = variable1.shape[0]
    if num_samples > plot_start:
        x = np.arange(start=plot_start, stop=num_samples)
        f = plt.figure()
        plt.title(plot_title)
        ax1 = plt.gca()
        line1, = ax1.plot(x, variable1[plot_start:num_samples])
        ax2 = ax1.twinx()
        line2, = ax2.plot(x, variable2[plot_start:num_samples], c='r')
        plt.legend((line1, line2), ("Test metrics", "Validation metrics"))
        ax1.set_ylabel("Test")
        ax2.set_ylabel("Validation")
        f.savefig(os.path.join(location, save_name + '_summary.png'))
        plt.close()
