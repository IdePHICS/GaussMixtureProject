from ..models.diag_cov import DiagModel, BlockModel, CorrModel
from ..algorithms.saddle_point import SaddlePoint
import pandas as pd

class LearningCurve(object):
    '''
    Implements experiment for generic loss and penalty.

    Note sample complexity is passed to run_experiment as an argument
    allowing for running several sample complexities for the same pre-diagonalised
    data model.
    '''
    def __init__(self,initialisation='uninformed', tolerance=1e-10, damping=0,
                 verbose=False, max_steps=1000, problem='dense', *,
                 loss, penalty, regularisation, variances, ratios, probability):

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
        self.max_steps = max_steps

    def learning_curve(self, *, sc_range):
        curve = {
            'loss': [],
            'penalty': [],
            'lambda': [],
            'probability': [],
            'sample_complexity': [],
            'V': [],
            'm': [],
            'q': [],
            'b': [],
            'test_error': [],
            'train_loss': [],
        }
        for alpha in sc_range:
            if self.verbose:
                print('Running sample complexity: {}'.format(alpha))

            self._run(sample_complexity = alpha)
            info_sp = self.sp.get_info()

            curve['loss'].append(self.loss)
            curve['lambda'].append(self.lamb)
            curve['sample_complexity'].append(alpha)
            curve['penalty'].append(self.penalty)
            curve['probability'].append(self.prob)

            curve['test_error'].append(info_sp['test_error'])
            curve['train_loss'].append(info_sp['train_loss'])
            curve['V'].append(info_sp['overlaps']['variance'])
            curve['q'].append(info_sp['overlaps']['self_overlap'])
            curve['m'].append(info_sp['overlaps']['teacher_student'])
            curve['b'].append(info_sp['overlaps']['bias'])

        self._learning_curve = pd.DataFrame.from_dict(curve)


    def get_curve(self):
        return self._learning_curve

    def _run(self, *, sample_complexity):
        '''
        Runs saddle-point equations.
        '''
        self._initialise_model(sample_complexity)

        self.sp = SaddlePoint(model=self.model,
                              initialisation=self.initialisation,
                              tolerance=self.tolerance,
                              damping=self.damping,
                              verbose=False,
                              max_steps=self.max_steps)

        self.sp.iterate()

    def _initialise_model(self, sample_complexity):
        if self.problem == 'dense':
            if 1 in self.ratios:
                i = self.ratios.index(1)
                self.model = DiagModel(sample_complexity = sample_complexity,
                                        regularisation = self.lamb,
                                        loss = self.loss,
                                        penalty = self.penalty,
                                        probability = self.prob,
                                        variance = self.deltas[i])

            else:
                self.model = BlockModel(sample_complexity = sample_complexity,
                                        regularisation = self.lamb,
                                        loss = self.loss,
                                        penalty = self.penalty,
                                        probability = self.prob,
                                        variances = self.deltas,
                                        ratios = self.ratios)

        elif self.problem == 'sparse':
            self.model = CorrModel(sample_complexity = sample_complexity,
                                    regularisation = self.lamb,
                                    loss = self.loss,
                                    penalty = self.penalty,
                                    probability = self.prob,
                                    variances = self.deltas,
                                    ratios = self.ratios)


    def save_experiment(self, date=False, unique_id=False, directory='.', *, name):
        '''
        Saves result of experiment in .json file with info for reproductibility.
        '''
        path = '{}/{}'.format(directory, name)

        if date:
            from datetime import datetime
            day, time = datetime.now().strftime("%d_%m_%Y"), datetime.now().strftime("%H:%M")
            path += '_{}_{}'.format(day, time)

        if unique_id:
            import uuid
            unique_id = uuid.uuid4().hex
            path += '_{}'.format(unique_id)

        path += '.csv'
        print('Saving experiment at {}'.format(path))
        self._learning_curve.to_csv(path, index=False)
