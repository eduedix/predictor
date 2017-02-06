import pandas as pd
import numpy as np
import xgboost as xgb


CSVFILE = 'input.csv'
CSVFILEOUTPUT = 'output.csv'


def normal_weighted_mean_squared_error(observed, predicted):
    w = np.log(observed + 1) + 1
    error = np.sum(w * (predicted - observed)**2) / np.sum(w)
    return error


class Predictor(object):
    """docstring for Predictor"""

    def __init__(self, params, target, test_size=0.3, seed=None):
        super(Predictor, self).__init__()
        self.params = params
        self.target = target
        self.test_size = test_size
        self.seed = seed

        self.initial_cv_params = {'eta': 0.1, 'seed': self.seed, 'subsample': 0.8,
                                  'colsample_bytree': 0.8, 'objective': 'reg:linear',
                                  'max_depth': 5, 'min_child_weight': 1}
        self.initial_grid_params = {'learning_rate': 0.1, 'seed': self.seed, 'subsample': 0.8,
                                    'colsample_bytree': 0.8, 'objective': 'reg:linear',
                                    'max_depth': 5, 'min_child_weight': 1}
        self.best_params = {}

    def init(self, df_train, eval_function=normal_weighted_mean_squared_error):
        self.set_training_data(df_train)
        self.set_err_func_and_scoring(eval_function)
        print('init finished')

    def set_training_data(self, df_train):
        from sklearn.model_selection import train_test_split
        X = df_train[self.params]
        Y = df_train[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, Y, test_size=self.test_size, random_state=self.seed)
        self.xgb_training_data = xgb.DMatrix(self.X_train, self.y_train)
        self.xgb_test_data = xgb.DMatrix(self.X_test, self.y_test)
        self.xgb_X_test = xgb.DMatrix(self.X_test)

    def set_test_data(self, df_test):
        X_test = df_test[self.params]
        self.xgb_X_test_target = xgb.DMatrix(X_test)

    def set_err_func_and_scoring(self, eval_function):
        from sklearn.metrics import make_scorer
        nwmse = make_scorer(eval_function, greater_is_better=False)

        def err_func(predicted, observations):
            observed = observations.get_label()
            return 'error', eval_function(observed, predicted)

        self.scoring = nwmse
        self.err_func = err_func

    def grid_search(self, ind_params, cv_params):
        from sklearn.model_selection import GridSearchCV
        optimized_GBM = GridSearchCV(xgb.XGBRegressor(**ind_params), cv_params,
                                     scoring=self.scoring, cv=5, n_jobs=-1,
                                     iid=False)
        optimized_GBM.fit(self.X_train, self.y_train)
        return optimized_GBM.best_params_

    def tune_n_estimators(self, params=None, best_n_estimators=None):
        if best_n_estimators:
            self.best_params.update({'n_estimators': best_n_estimators})
            return best_n_estimators
        if params is None:
            params = self.initial_cv_params.copy()

        params.update(self.best_params)
        if 'n_estimators' in params:
            del params['n_estimators']

        cv_xgb = xgb.cv(params=params, dtrain=self.xgb_training_data,
                        num_boost_round=10000, nfold=5, feval=self.err_func,
                        early_stopping_rounds=100)
        n_estimators = cv_xgb.shape[0]

        self.best_params.update({'n_estimators': n_estimators})
        print('tune n_estimators, n_estimators')
        print(n_estimators)
        print()
        return n_estimators

    def tune_max_depth_and_min_child_weight(self, best_params=None):
        if best_params and 'max_depth' in best_params and 'min_child_weight' in best_params:
            self.best_params.update(best_params)
            return
        cv_params = {'max_depth': [3, 5, 7], 'min_child_weight': [1, 3, 5, 7]}
        ind_params = self.initial_grid_params.copy()
        ind_params.update(self.best_params)
        if 'max_depth' in ind_params:
            del ind_params['max_depth']
        if 'min_child_weight' in ind_params:
            del ind_params['min_child_weight']

        best_params = self.grid_search(ind_params, cv_params)

        self.best_params.update(best_params)
        print('tune max_depth and min_child_weight')
        print(best_params)
        print()
        return best_params

    def tune_gamma(self, best_params=None):
        if best_params and 'gamma' in best_params:
            self.best_params.update(best_params)
            return

        cv_params = {'gamma': [i / 10.0 for i in range(0, 5)]}
        ind_params = self.initial_grid_params.copy()
        ind_params.update(self.best_params)
        if 'gamma' in ind_params:
            del ind_params['gamma']

        best_params = self.grid_search(ind_params, cv_params)

        self.best_params.update(best_params)
        print('tune gamma')
        print(best_params)
        print()
        return best_params

    def tune_subsample_and_colsample_bytree(self, best_params=None):
        if best_params and 'subsample' in best_params and 'colsample_bytree' in best_params:
            self.best_params.update(best_params)
            self.tune_n_estimators()
            return
        cv_params = {
            'subsample': [i / 10.0 for i in range(6, 10)],
            'colsample_bytree': [i / 10.0 for i in range(6, 10)]
        }
        ind_params = self.initial_grid_params.copy()
        ind_params.update(self.best_params)
        if 'subsample' in ind_params:
            del ind_params['subsample']
        if 'colsample_bytree' in ind_params:
            del ind_params['colsample_bytree']

        best_params = self.grid_search(ind_params, cv_params)
        self.best_params.update(best_params)
        print('tune subsample and colsample_bytree')
        print(best_params)
        print()

        print('calibrate n_estimators')
        params = self.initial_cv_params.copy()
        params.update(self.best_params)
        self.tune_n_estimators(params)
        return best_params

    def tune_reg_alpha(self, best_params=None):
        if best_params and 'reg_alpha' in best_params:
            self.best_params.update(best_params)
            return
        cv_params = {'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05]}
        ind_params = self.initial_grid_params.copy()
        ind_params.update(self.best_params)
        if 'reg_alpha' in ind_params:
            del ind_params['reg_alpha']

        best_params = self.grid_search(ind_params, cv_params)
        self.best_params.update(best_params)
        print('tune reg_alpha')
        print(best_params)
        print()
        return best_params

    def tune_learning_rate(self, learning_rate=0.03):
        params = self.initial_cv_params.copy()
        params.update(self.best_params)
        params['eta'] = learning_rate
        self.best_params.update({'eta': learning_rate})
        self.best_params.update({'learning_rate': learning_rate})

        print('tune learning_rate')
        print(learning_rate)
        print()

        print('calibrate n_estimators')
        self.tune_n_estimators(params)

    def find_best_model_params(self):
        print('find best model params: ')

        self.tune_n_estimators()
        self.tune_max_depth_and_min_child_weight()
        self.tune_gamma()
        self.tune_subsample_and_colsample_bytree()
        self.tune_reg_alpha()
        self.tune_learning_rate()

        print('best params: ')
        print(self.best_params)
        print()
        return self.best_params

    def fit_best_model(self, params=None):
        if params is None:
            params = self.initial_cv_params.copy()
            params.update(self.best_params)
        num_boost_round = params['n_estimators']
        if 'n_estimators' in params:
            del params['n_estimators']
        watchlist = [(self.xgb_test_data, 'eval'),
                     (self.xgb_training_data, 'train')]
        best_model = xgb.train(params, self.xgb_training_data, num_boost_round,
                               watchlist, early_stopping_rounds=100, feval=self.err_func)
        self.best_model = best_model

        return best_model

    def predict(self, xgb_test_data=None):
        if xgb_test_data is None:
            xgb_test_data = self.xgb_X_test
        model = self.best_model
        pred = model.predict(xgb_test_data)
        return pred

    def predict_test(self):
        return self.predict(self.xgb_X_test_target)


def cleanse_data(df):
    '''
    Remove unnecessary columns from df here
    Set missing data points to np.nan
    '''
    return df


def update_data(df):
    '''
    Create augmented df by adding new columns
    Example: Add mean and std of columns for clusters
    '''
    return df


def finetune_df_output(df_output):
    '''
    Finetune your output here
    '''
    return df_output


def produce_csv_output(df_test, pred):
    df_output = pd.DataFrame({'predictions': pred}, index=df_test['id'])
    df_output.index.name = 'id'
    df_output = finetune_df_output(df_output)
    df_output.to_csv(CSVFILEOUTPUT)


def main():
    df = pd.read_csv(CSVFILE, dtype={'id': np.int64})
    df = cleanse_data(df)
    df = update_data(df)

    params = ['param1', 'param2']  # enter predictor parameters from your csv
    target = 'target_parameter'  # enter your target parameter here from your csv

    df_train = df[(~np.isnan(df[target]))]
    df_test = df[(np.isnan(df[target]))]

    test_size, seed = 0.3, 7

    predictor = Predictor(params, target, test_size, seed)
    predictor.init(df_train, normal_weighted_mean_squared_error)

    best_params = {'n_estimators': 10000, 'eta': 0.1, 'subsample': 0.8,
                   'colsample_bytree': 0.8, 'objective': 'reg:linear',
                   'max_depth': 5, 'min_child_weight': 1}

    # TODO: you can uncomment next line for fine-tuning
    # best_params = predictor.find_best_model_params()

    best_model = predictor.fit_best_model(best_params)

    predictor.set_test_data(df_test)

    print()
    print('best params:')
    print(best_params)
    print()
    pred = predictor.predict_test()

    produce_csv_output(df_test, pred)
    print()
    print('predictions saved at: %s' % CSVFILEOUTPUT)

    return


if __name__ == '__main__':
    main()
