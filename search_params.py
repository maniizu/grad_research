#-*- coding:utf-8 -*-
"""
Name: search_params.py
Date: 2016/12/14
usage: trees.pyで呼び出して使用
Description:
	ランダム・フォレストの最適なハイパーパラメータを、候補を総探索することで得るためのプログラム。
	Leave One Outと年ごとでクロスバリデーションする場合の2パターンをカバーする。
"""

from sklearn.cross_validation import LeaveOneLabelOut
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

import pandas as pd

class SearchParams:

	def __init__(self, data):
		"""
		parameters:

		params = {'n_estimators': [500, 600, 700, 800, 900, 1000],
				'min_samples_leaf': [i for i in range(1,51)]}
			グリッドサーチで探索するハイパーパラメータの候補

		estimator = RandomForestRegressor()
			ハイパーパラメータを必要とする実験で使用する回帰器

		data	グリッドサーチに使用するデータ
			ただし、yearという名前の変数列は除く

		labels = data['year']
			LeaveOneLabelOutで使用する分割のためのラベル
		"""
		self.params = {'n_estimators': [500, 600, 700, 800, 900, 1000],
				'min_samples_leaf': [i for i in range(1,21)]}
		self.estimator = RandomForestRegressor()
		self.data = data
		self.labels = data['year']
		if 'year' in self.data.columns:
			del self.data['year']


	def leave_one_label_out(self):
		"""
		sklearnのcross_validationにあるLeaveOneLabelOut()を使用して、
		年ごとのクロスバリデーション時に使用する。
		データをトレーニング用とテスト用に分けるオブジェクトを作成する。
		yearはpandasのデータフレーム型かシリーズ型を想定。
		"""
		lol = LeaveOneLabelOut(self.labels)
		return lol


	def search_params_LOO(self):
		"""
		Leave One Outで実行する場合の最適なハイパーパラメータを探索する。
		使用する回帰器はランダム・フォレストである。
		対象とするパラメータはn_estimatorsとmin_samples_leafの2つで、
		n_estimatorsは500,600,700,...1000まで、min_samples_leafは1から50までを探索する。
		dataはpandasのデータフレーム型で引き渡しを想定していて、第1列目に目的変数が、
		2列目以降に説明変数が格納されているものとする。
		"""
		RFR = GridSearchCV(estimator = self.estimator, param_grid = self.params, cv = len(self.data))
		RFR.fit(self.data.ix[:, 1:], self.data.ix[:, 0])

		print(RFR.best_params_)


	def search_params_LOL(self, lol):
		"""
		開催年ごとに分割するクロスバリデーションで実行する場合の最適なハイパーパラメータを探索する。
		使用する回帰器はランダム・フォレストである。
		対象とするパラメータはn_estimatorsとmin_samples_leafの2つで、
		n_estimatorsは500,600,700,...1000まで、min_samples_leafは1から50までを探索する。
		dataはpandasのデータフレーム型で引き渡しを想定していて、第1列目に目的変数が、
		2列目以降に説明変数が格納されているものとする。
		"""
		RFR = GridSearchCV(estimator = self.estimator, param_grid = self.params, cv = lol)
		RFR.fit(self.data.ix[:, 1:], self.data.ix[:, 0])
		print(RFR.best_params_)

if __name__ == '__main__':
	data_au = pd.read_csv("data_au.csv", delimiter = ',')
	data_co = pd.read_csv("data_co.csv", delimiter = ',')

	search_params_au = SearchParams(data_au)
	search_params_co = SearchParams(data_co)

	print("=" * 10 + "Leave One Outの場合" + "=" * 10)

	search_params_au.search_params_LOO()
	search_params_co.search_params_LOO()

	print("\n" + "=" * 10 + "Leave One Label Outの場合" + "=" * 10)

	lol_au = search_params_au.leave_one_label_out()
	lol_co = search_params_co.leave_one_label_out()

	search_params_au.search_params_LOL(lol_au)
	search_params_co.search_params_LOL(lol_co)
