import os
import sys
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import logging

import pandas as pd
import numpy as np

import utils
from features import *

from sklearn import linear_model, ensemble, neighbors, tree, cross_validation

logger = logging.getLogger('main_program')
hdlr = logging.FileHandler('test.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)

program_description = '''

    Making Python machine learning prototype like playing Lego bricks.
                     _          _      _
                    | |        (_)    | |
                    | |__  _ __ _  ___| | _____
                    | '_ \| '__| |/ __| |/ / __|
                    | |_) | |  | | (__|   <\__ \\
                    |_.__/|_|  |_|\___|_|\_\___/


'''


def read_data(path):
    filename, file_extension = os.path.splitext(path)
    if file_extension == '.csv':
        try:
            df = pd.read_csv(path, encoding='utf-8')
            return df
        except Exception, e:
            print e
    elif file_extension in ['.xls', '.xlsx']:
        try:
            df = pd.read_excel(path, encoding='utf-8')
            return df
        except Exception, e:
            print e


def main(options):
    parser = ArgumentParser(prog='Bricks',
                            formatter_class=RawDescriptionHelpFormatter,
                            description=program_description)
    parser.add_argument('-d', '--data', dest='data_path', type=str,
                        help='The file path to the dataset')
    parser.add_argument('-l', '--label', dest='label_column', type=str,
                        help='The column name of the labels')
    parser.add_argument('-c', '--config', dest='config_path', type=str,
                        help='The file path to the config file')
    parser.add_argument('-f', '--fold', dest='fold_num', type=str,
                        help='The number of folds for cross-validation')
    parser.add_argument('-o', '--output', dest='output_path', type=str,
                        help='1: using historical data of interaction; 2: calculating interaction')
    args = parser.parse_args()

    print utils.announce(program_description)

    print utils.info('Loading data from %s' % utils.announce(args.data_path))
    print '\n'
    df = read_data(args.data_path)

    print utils.info('Columns:'), utils.announce(', '.join(df.columns))

    print utils.info('Total samples:'), utils.announce(str(len(df)))

    print utils.info('Labels distribution')
    for label, count in df[args.label_column].value_counts().iteritems():
        print utils.announce(label), count

    print '\n'

    features = GroupFeatures([
        ConstantFeature('Sepal Length'),
        ArithmeticFeature('Sepal Width', ArithmeticOperator.LOG, 2),
        ArithmeticFeature('Petal Length', ArithmeticOperator.ADD, 10),
        StackedFeatures([
            ArithmeticFeature('Petal Length', ArithmeticOperator.LOG, 3),
            ArithmeticFeature(None, ArithmeticOperator.POWER, 2),
            ArithmeticFeature(None, ArithmeticOperator.MULTIPLY, 0.15)
        ])
    ])

    print utils.info('Extracting features')
    X = features.transform(df).T
    y = df[args.label_column].tolist()

    print '\n\n'
    classifiers = [
        linear_model.LogisticRegression(penalty='l2'),
        neighbors.KNeighborsClassifier(10, weights='uniform', algorithm='auto', leaf_size=35, p=2, metric='minkowski'),
        linear_model.PassiveAggressiveClassifier(n_iter=9),
        linear_model.SGDClassifier(loss="hinge", penalty='l2', n_iter=5),
        tree.DecisionTreeClassifier(max_depth=5),
        ensemble.RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    ]
    classifiers_names = [classifier.__class__.__name__ for classifier in classifiers]
    print utils.info('Evaluating classifiers:'), utils.announce(', '.join(classifiers_names))

    models = []
    for name, classifier in zip(classifiers_names, classifiers):
        print utils.info('Training'), utils.announce(name)
        models.append(classifier.fit(X, y))
        scores = cross_validation.cross_val_score(classifier, X, y, cv=args.fold_num, n_jobs=3)
        print utils.info('\t\tAccuracy'), utils.announce(str(np.mean(scores))), utils.debug('+-' + '%.2f' % (np.std(scores)))
        print '\n'

if __name__ == '__main__':
    df = main(sys.argv[1:])
