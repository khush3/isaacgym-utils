from comet_ml import Experiment

import os
import yaml
import time
import torch
import numpy as np


def to_list(ip):
    if isinstance(ip, list):
        return ip
    elif isinstance(ip, np.ndarray) or isinstance(ip, torch.Tensor):
        return ip.tolist()
    else:
        return [ip]


class Logger:
    def __init__(self, config):
        self.args = config['training']
        self.log_path = self.args['log_path']
        self.log_online = self.args['log_online']
        self.log_offline = self.args['log_offline']
        self.print_stats = self.args['print_stats']

        self.episode = 0
        self.step = 0
        self.best_return = -1e10
        exp_name = time.asctime()[4:16].replace(' ', '_').replace(':', '_')

        if self.log_online:
            self.exp = Experiment(self.args['api_key'], self.args['project_name'])
            self.exp.set_name(exp_name)
            if self.args['tags'] is not None: self.exp.add_tags(to_list(self.args['tags']))
            self.exp.log_parameters(config)
        else:
            self.exp = None

        if self.log_offline:
            self.log_path = os.path.join(os.path.abspath(os.getcwd()), '..', self.log_path, exp_name)
            os.makedirs(self.log_path, exist_ok=True)
            with open('config.yml', 'w') as outfile:
                yaml.dump(config, outfile, default_flow_style=False)


    def _save(self, **kwargs):
        r'''(Variable inps) Update stats and save the log online and/or offline.'''

        for stat_label, stat in kwargs.items():
            if (self.exp is not None) and self.log_online: 
                self.exp.log_metric(stat_label, np.mean(stat))
            if self.log_offline:
                with open(os.path.join(self.log_path, stat_label + '.txt'), 'a') as f:
                    f.write(str(stat) + "\n")


    def update(self, rets, model):
        r"""Update stats and save the log online and/or offline.
    
        Args:
            pred (torch.Tensor): Network prediction
            target (torch.Tensor): Labels
            step_size (Float): log interval step size
            batch_idx (Float): Index of the current mini batch
            loss (Float): Loss at the current step
        """

        rets = to_list(rets)
        stats = {
            'returns': rets
        }

        self._save(**stats)
        print(rets)

        for ret in rets:
            print(ret)
            if ret > self.best_return:
                self.best_return = ret
            self.save_model(model, 'best_model')
    
        if self.print_stats:
            print(f'Episode {self.episode} | Best return: {self.best_return} | Current returns: {rets}')
        self.episode += 1

    
    def save_model(self, model, file_name):

        file_path = os.path.join(self.log_path, file_name)

        if isinstance(model, np.ndarray):
            if self.log_offline:
                np.save(file_path, model)
        else: 
            raise NotImplementedError
        if self.log_online:
            self.exp.log_other(file_name, model)


    def show_plots(self):
        pass


    def save_plots(self):
        pass


    def end(self):
        if self.exp is not None: self.exp.end()