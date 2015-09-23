import operator
import math

import pandas as pd
import numpy as np


class ConstantFeature(object):

    """docstring for ConstantFeature"""

    def __init__(self, column_name):
        self.column_name = column_name

    def __str__(self):
        return 'ConstantFeature: %s' % self.column_name

    def __repr__(self):
        return 'ConstantFeature: %s' % self.column_name

    def transform(self, data):
        if isinstance(data, pd.DataFrame):
            if self.column_name not in data:
                raise NameError('This column is not exists')
            return data[self.column_name].tolist()
        return data


class ArithmeticOperator(object):

    """docstring for ArithmeticOperator"""
    ADD = operator.add
    SUBTRACT = operator.sub
    MULTIPLY = operator.mul
    DIVIDE = operator.div
    MODULO = operator.mod
    LOG = math.log
    POWER = math.pow
    SINE = math.sin
    COSINE = math.cos
    TAN = math.tan


class ArithmeticFeature(object):

    """docstring for ArithmeticFeature"""

    def __init__(self, column_name, arithmetic_operator, operand):
        self.column_name = column_name
        self.arithmetic_operator = arithmetic_operator
        self.operand = operand

    def __str__(self):
        return 'ArithmeticFeature: %s <%s> %s' % (self.column_name, self.arithmetic_operator.__name__, self.operand)

    def __repr__(self):
        return 'ArithmeticFeature: %s <%s> %s' % (self.column_name, self.arithmetic_operator.__name__, self.operand)

    def transform(self, data):
        if isinstance(self.column_name, basestring):
            if isinstance(data, pd.DataFrame):
                if self.column_name not in data:
                    raise NameError('This column [%s] is not exists' % self.column_name)
                return data[self.column_name].apply(lambda x: self.arithmetic_operator(x, self.operand)).tolist()
        return map(lambda x: self.arithmetic_operator(x, self.operand), data)


class StackedFeatures(object):

    """docstring for StackedFeatures"""

    def __init__(self, features):
        self.features = features

    def __str__(self):
        return 'StackedFeatures (%s features): %s' % \
            (len(self.features), ' -> '.join([str(feature) for feature in self.features]))

    def __repr__(self):
        return 'StackedFeatures (%s features): %s' % \
            (len(self.features), ' -> '.join([str(feature) for feature in self.features]))

    def transform(self, data):
        for feature in self.features:
            data = feature.transform(data)
        return data


class GroupFeatures(object):

    """docstring for GroupFeatures"""

    def __init__(self, features):
        self.features = features

    def __str__(self):
        return 'GroupFeatures (%s features): [%s]' % \
            (len(self.features), ' + '.join([str(feature) for feature in self.features]))

    def __repr__(self):
        return 'GroupFeatures (%s features): [%s]' % \
            (len(self.features), ' + '.join([str(feature) for feature in self.features]))

    def transform(self, data):
        return np.vstack([feature.transform(data) for feature in self.features])
