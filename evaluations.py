#-*- coding:utf-8 -*-
"""
Name: evaluations.py
Date: 2017/1/17
usage: multi_regr.pyやrandom_forest.pyで呼び出して使用
Description:
	RMSEと自由度調整済み決定係数を計算する関数を提供する.
"""

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import math


def root_mean_squared_error(data, y_test_pred):
	"""
	RMSEを算出する。
	"""
	#print(mean_squared_error(data, y_test_pred))
	return math.sqrt(mean_squared_error(data, y_test_pred))


def r2_adj_score(data, y_test_pred, number_of_features):
	"""
	調整済み決定係数を算出する。
	"""
	#print(r2_score(data, y_test_pred))
	a = (len(data) - 1) / (len(data) - number_of_features - 1)
	r2_adj = 1 - a * (1 - r2_score(data, y_test_pred))
	return r2_adj
