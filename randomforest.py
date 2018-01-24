#-*- coding:utf-8 -*-
"""
Name: randomforest.py
Date: 2016/1/18
usage: python3 randomforest.py param
	param: LOLを入力. LOLなら年ごとのクロスバリデーションが行われる.
Description:
　　山形国際ドキュメンタリー映画祭のデータを使用してランダム・フォレストで各映画の観客数と会場充足率を予測する.
	ランダム・フォレストの最適な学習終了条件(オプションのmin_samples_leafの値)を
	クロスバリデーションを用いて総当りで探索し,最も決定係数が高くなるときの条件でランダム・フォレストを作成する.
	出力はデータ全てを学習データにしたときの当てはまり具合(決定係数)と,
	クロスバリデーションで算出された説明変数の重要度の平均である.
"""

import sys
import numpy as np
import pandas as pd
from sklearn import ensemble as ens
from sklearn import cross_validation as cv
from evaluations import root_mean_squared_error, r2_adj_score
from resid_plot import resid_plot
import seaborn as sns
import matplotlib.pyplot as plt

class RFCV:

	def __init__(self, data):

		self.RFR = ens.RandomForestRegressor
		self.y_test_pred = pd.DataFrame()
		self.features = []
		self.feature_importances_ = []
		self.data = data
		self.labels = data['year']
		del self.data['year']


	"""
	def mean_feature_importances(self):

		クロスバリデーションの各回で得られた説明変数の重要度を平均化する。

		self.features = np.array(self.features)
		for i in range(len(self.features.T)):
			self.feature_importances_.append([self.data.columns[i+1],self.features.T[i].mean()])
		self.feature_importances_ = dict(self.feature_importances_)
		return None
	"""
	"""
	def plot_importance(self):
		plot_data = pd.DataFrame()
		plot_data['features'] = self.data.columns[1:]
		year = 2011
		for importances in self.feature_importances_:
			plot_data['importances'] = importances
			plot_data.sort_values(by=["importances"], ascending=False)
			sns.set(style="whitegrid")
			f, ax = plt.subplots(figsize=(6, 15))
			sns.barplot(x="importances", y="features", data=plot_data)
			sns.plt.savefig("importance_"+str(year)+".pdf")
			sns.plt.show()
			year+=2
	"""


	def random_forest_cv(self, cv, estimators, min_samples):
		"""
		ランダム・フォレストでクロスバリデーションする。cvに分割のためのオブジェクトを渡し、
		estimatorsに木構造の数を、min_samplesに葉に含まれるサンプル数を設定する。
		出力はpandasのデータフレーム型になっている各目的変数の予測値と、
		クロスバリデーションの各試行で得られた説明変数の重要度の平均(リスト型)である
		"""
		RFR = self.RFR(n_estimators = estimators, min_samples_leaf = min_samples, \
			random_state = np.random.seed(10))
		year = 2011
		counter = 0

		print(self.data.columns[1:])
		for train_index, test_index in cv:
			RFR_learn = RFR.fit(self.data.ix[train_index, 1:], self.data.ix[train_index, 0])
			self.y_test_pred = pd.concat([self.y_test_pred, \
				pd.DataFrame(RFR_learn.predict(self.data.ix[test_index, 1:]))], ignore_index = True)
			predata_r2_adj = r2_adj_score(self.data.ix[train_index,0], \
				pd.DataFrame(RFR_learn.predict(self.data.ix[train_index, 1:])), \
				len(RFR_learn.feature_importances_))
			predata_rmse = root_mean_squared_error(self.data.ix[train_index, 0], \
				RFR_learn.predict(self.data.ix[train_index, 1:]))
			rmse = root_mean_squared_error(self.data.ix[test_index,0], self.y_test_pred.ix[test_index])
			r2_adj = r2_adj_score(self.data.ix[test_index,0], self.y_test_pred.ix[test_index], \
				len(RFR_learn.feature_importances_))
			print(year+counter*2,"年の訓練データへの当てはめ:", predata_r2_adj, predata_rmse)
			print(year+counter*2,"年:", r2_adj, rmse)
			print(RFR_learn.feature_importances_)
			self.features.append(RFR_learn.feature_importances_)

			#プロット用のデータを整理
			plot_data = pd.DataFrame()
			plot_data['features'] = self.data.columns[1:]
			plot_data['importances'] = RFR_learn.feature_importances_
			plot_data = plot_data.sort_values("importances", ascending=False)
			plot_sub_data = plot_data[:3]
			#ここからプロット
			"""
			sns.set(style="whitegrid", font="IPAPGothic")
			f, ax = plt.subplots(figsize=(10, 12))
			sns.barplot(x="importances", y="features", data=plot_data)
			sns.plt.xlabel("importances")
			sns.plt.xlim([0,0.4])
			sns.plt.savefig("importance_"+self.data.columns[0][:2]+str(year+counter*2)+".pdf")
			sns.plt.show()
			"""
			#プレゼン用の縮小版をプロット
			sns.set(font="IPAPGothic")
			sns.barplot(x="features", y="importances", data=plot_sub_data, palette="autumn")
			sns.plt.ylabel("重要度", fontsize=20)
			sns.plt.xlabel("")
			sns.plt.tick_params(labelsize=18)
			sns.plt.ylim([0,0.4])
			sns.plt.savefig("smallimp_"+self.data.columns[0][:2]+"_"+str(year+counter*2)+".pdf")
			sns.plt.show()

			counter += 1

		#RFCV.mean_feature_importances(self)
		#RFCV.plot_importance(self)
		return self.y_test_pred #self.feature_importances_)



if __name__ == '__main__':

	data_au = pd.read_csv("data_au.csv", delimiter = ',')
	data_co = pd.read_csv("data_co.csv", delimiter = ',')

	label = data_au['year']

	random_forest_au = RFCV(data_au)
	random_forest_co = RFCV(data_co)

	"""
	if sys.argv[1] == "LOO":
		kf = cv.KFold(n = len(data_au), n_folds = len(data_au))

		pred_au = random_forest_au.random_forest_cv(kf, 1000, 5)
		pred_co = random_forest_co.random_forest_cv(kf, 1000, 5)
	"""

	if sys.argv[1] == "LOL":
		lol = cv.LeaveOneLabelOut(label)

		print("="*10, "audience", "="*10)
		pred_au = random_forest_au.random_forest_cv(lol, 600, 6)
		print("="*10, "congestion rate", "="*10)
		pred_co = random_forest_co.random_forest_cv(lol, 500, 4)

	"""
	print("=" * 10 + "audience を回帰" + "=" * 10)
	print("RMSE : ", random_forest_au.root_mean_squared_error())
	print("adj.r2 : ", random_forest_au.r2_adj_score())

	print("=" * 10 + "congestion rate を回帰" + "=" * 10)
	print("RMSE : ", random_forest_co.root_mean_squared_error())
	print("adj.r2 : ", random_forest_co.r2_adj_score())

	#残差プロット
	resid_plot(data_au.ix[:, 0], pred_au.ix[:, 0], xmin = 0, xmax = 1000)
	resid_plot(data_co.ix[:, 0], pred_co.ix[:, 0], xmin = 0, xmax = 1.6)
	"""

"""
from randomforest import RFCV
import pandas as pd
from sklearn import cross_validation as cv
data = pd.read_csv("data_au.csv", delimiter = ',')
label = data['year']
rf = RFCV(data)
lol = cv.LeaveOneLabelOut(label)
pred_au, feature_au = rf.random_forest_cv(lol, 1000, 1)


from randomforest import RFCV
import pandas as pd
from sklearn import cross_validation as cv
data = pd.read_csv("data_au.csv", delimiter = ',')
rf = RFCV(data)
kf = cv.KFold(n = len(data), n_folds = len(data))
pred_au, feature_au = rf.random_forest_cv(kf, 1000, 5)
"""
