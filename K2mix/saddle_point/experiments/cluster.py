import numpy as np

from ..models.diag_cov import DiagModel, BlockModel, CorrModel
from ..algorithms.saddle_point import SaddlePoint

class ClusterExperiment(object):
    '''
    Implements a cluster experiment where every point in run independently.
    '''
    def __init__(self,initialisation='uninformed', tolerance=1e-7, damping=0,
                 verbose=False, max_steps=int(1e4), max_attempt=5, problem='dense', *, sample_complexity,
                 loss, penalty, regularisation, variances, ratios, probability):


        self.alpha = sample_complexity
        self.loss = loss
        self.lamb = regularisation
        self.penalty = penalty
        self.deltas = variances
        self.ratios = ratios
        self.prob = probability
        self.problem = problem

        # Hyperparameters
        self.initialisation = initialisation
        self.tolerance = tolerance
        self.damping = damping
        self.verbose = verbose
        self.max_attempt = max_attempt
        self.max_steps = max_steps

        # Initialise the model
        self._initialise_model()

    def _initialise_model(self):
        if self.problem == 'dense':
            if 1 in self.ratios:
                i = self.ratios.index(1)
                self.model = DiagModel(sample_complexity = self.alpha,
                                        regularisation = self.lamb,
                                        loss = self.loss,
                                        penalty = self.penalty,
                                        probability = self.prob,
                                        variance = self.deltas[i])

            else:
                self.model = BlockModel(sample_complexity = self.alpha,
                                        regularisation = self.lamb,
                                        loss = self.loss,
                                        penalty = self.penalty,
                                        probability = self.prob,
                                        variances = self.deltas,
                                        ratios = self.ratios)

        elif self.problem == 'sparse':
            self.model = CorrModel(sample_complexity = self.alpha,
                                    regularisation = self.lamb,
                                    loss = self.loss,
                                    penalty = self.penalty,
                                    probability = self.prob,
                                    variances = self.deltas,
                                    ratios = self.ratios)

    def run(self, verbose=False):
        '''
        Runs saddle-point equations.
        Attemps different values of damping if running returns NaN.
        '''
        damping_vals = np.linspace(self.damping, 0.999, num=self.max_attempt)

        for damping in damping_vals:
            self.sp = SaddlePoint(model=self.model,
                                  initialisation=self.initialisation,
                                  tolerance=self.tolerance,
                                  damping=damping,
                                  verbose=False,
                                  max_steps=self.max_steps)

            self.sp.iterate()
            if not any(np.isnan(val[-1]) for val in self.sp.overlaps.values()):
                break
            else:
                print('NaN detected. Running again for higher damping.')


    def save_experiment(self, data_dir):
        '''
        Saves result of experiment in .json file with info for reproductibility.
        '''
        import json
        import os
        from datetime import datetime
        import uuid

        unique_id = uuid.uuid4().hex
        day, time = datetime.now().strftime("%d_%m_%Y"), datetime.now().strftime("%H:%M:%S")

        info = self.sp.get_info()
        info.update({
            'date': '{}_{}'.format(day,time),
        })

        sub_dir = '{}/{}'.format(data_dir, day)

        # If directory doesn't exist, create it
        if not os.path.isdir(sub_dir):
            os.mkdir(sub_dir)

        name = '{}/{}_{}.json'.format(sub_dir, info['covariance'], unique_id)
        print('Saving experiment at {}'.format(name))
        with open(name, 'w') as outfile:
            json.dump(info, outfile)
