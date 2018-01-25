#-*- coding:utf-8 -*-
"""
Name: multi_regr.py
Date: 2016/1/18
usage: python3 multi_regr.py param
	param: LOLを入力. LOLなら年ごとのクロスバリデーションが行われる.

Description:
	映画祭について重回帰分析を行う.
"""

import sys
import pandas as pd
import statsmodels.api as sm
from sklearn import cross_validation as cv
from evaluations import root_mean_squared_error, r2_adj_score
from resid_plot import resid_plot
import matplotlib.pyplot as plt
import seaborn as sns
#from backward import backward_aic

class MRCV:

	def __init__(self, data):

		self.model = sm.OLS
		self.y_test_pred = pd.DataFrame()
		self.features = []
		self.data = data
		self.labels = data['year']
		del self.data['year']

	"""
	def multi_regr_cv(self, cv):
		year = 2011
		counter = 0

		for train_index, test_index in cv:
			#remove_columns = backward_aic(self.data.ix[train_index], self.data.columns[0])
			#for removement in remove_columns:
			#	del data[removement]

			model = self.model(self.data.ix[train_index, 0], self.data.ix[train_index, 1:]).fit()
			self.y_test_pred = pd.concat([self.y_test_pred, \
				pd.DataFrame(model.predict(self.data.ix[test_index, 1:]))], ignore_index = True)
			rmse = root_mean_squared_error(self.data.ix[test_index, 0], \
				self.y_test_pred.ix[test_index])
			r2_adj = r2_adj_score(self.data.ix[test_index, 0], \
				self.y_test_pred.ix[test_index], len(data.columns[1:]))
			print(model.summary())
			print(year+counter*2,"年:", r2_adj, rmse, end = "\n\n\n")
			counter += 1

		return self.y_test_pred
	"""

	def multi_regr_lol(self, delete):
		lol = cv.LeaveOneLabelOut(label)

		year = 2011
		counter = 0
		for train_index, test_index in lol:
			data = self.data[:]
			for remove_element in delete[counter]:
				del data[remove_element]

			model = self.model(data.ix[train_index, 0], data.ix[train_index, 1:]).fit()
			self.y_test_pred = pd.concat([self.y_test_pred, \
				pd.DataFrame(model.predict(data.ix[test_index, 1:]))], ignore_index = True)

			predata_r2_adj = r2_adj_score(data.ix[train_index,0], \
				pd.DataFrame(model.predict()), len(data.columns[1:]))
			predata_rmse = root_mean_squared_error(data.ix[train_index, 0], \
				pd.DataFrame(model.predict()))
			rmse = root_mean_squared_error(data.ix[test_index, 0], self.y_test_pred.ix[test_index])
			r2_adj = r2_adj_score(data.ix[test_index, 0], self.y_test_pred.ix[test_index], \
				len(data.columns[1:]))
			print(model.summary())
			print(year+counter*2,"年:", r2_adj, rmse, )#end = "\n\n\n")
			print(year+counter*2,"年の訓練データへのあてはまり:", predata_r2_adj, predata_rmse, end = "\n\n\n")

			#t値が大きいものから3つの変数をプロット
			#print(model.tvalues)
			tvalues = pd.DataFrame(model.tvalues, columns=['t_value'])
			tvalues = tvalues.sort_values(by='t_value', ascending=False)
			#print(tvalues, type(tvalues))
			tvalue = tvalues[:3]
			print(tvalue)
			sns.set(font='IPAPGothic')
			sns.barplot(x=tvalue.index, y='t_value', data=tvalue, palette='autumn')
			sns.plt.ylabel("t 値", fontsize=20)
			sns.plt.xlabel("")
			sns.plt.tick_params(labelsize=18)
			sns.plt.savefig(data.columns[0]+'_'+str(year+counter*2)+'.pdf')
			sns.plt.show()

			counter += 1

if __name__ == '__main__':

	data_au = pd.read_csv("data_au.csv", delimiter = ',')
	data_co = pd.read_csv("data_co.csv", delimiter = ',')

	label = data_au['year']

	del data_au['capacity'], data_co['capacity']

	multi_regr_au = MRCV(data_au)
	multi_regr_co = MRCV(data_co)

	"""
	if sys.argv[1] == "LOO":
		kf = cv.KFold(n = len(data_au), n_folds = len(data_au))

		pred_au = multi_regr_au.multi_regr_cv(kf)
		pred_co = multi_regr_co.multi_regr_cv(kf)
	"""

	if sys.argv[1] == "LOL":

		#年ごとにクロスバリデーションを行うとき、ステップワイズ法によって除外される変数列
		#RでAICを求めて選択した(別ファイル参照)
		remove_au_2011 = ['q_and_a', 'temp', 'start_time', 'precipitation', 'rerun', \
			'same_time', 'lang_support', 'AD', 'AP', 'CM', 'CU', 'DS', 'EM', 'FC', \
			'IS', 'JF', 'NAC', 'NDJ', 'PJ', 'TJ', 'TV', 'YF', 'YN', 'フォーラム3', \
			'フォーラム4', 'フォーラム5', '美術館1', '美術館2']
		remove_au_2013 = ['past_works', 'temp', 'start_time', 'sunshine', 'precipitation', \
		 	'rerun', 'same_time', 'lang_support', 'free', 'AD', 'AP', 'AS', 'CM', 'CU', \
			'DS', 'EM', 'FC', 'JF', 'LA','NDJ', 'PJ', 'SI', 'YN', 'フォーラム4', \
			'小ホール', '美術館1', '美術館2']
		remove_au_2015 = ['q_and_a', 'past_works', 'temp', 'start_time', 'pre_aud', \
			'sunshine', 'precipitation', 'same_time', 'lang_support', 'pr_yamagata', \
			'AD', 'AP', 'AS', 'CU', 'DS', 'EM', 'FC', 'IS', 'JF', 'LA', 'TJ', 'TV', \
			'YN', 'フォーラム3', 'フォーラム4', 'フォーラム5', '美術館1', '美術館2']
		remove_au = [remove_au_2011, remove_au_2013, remove_au_2015]

		remove_co_2011 = ['country', 'q_and_a', 'past_works', 'start_time', 'pre_aud', \
			'precipitation', 'rerun', 'same_time', 'AD', 'AP', 'CM', 'CU', 'IS', 'NDJ', \
			'PJ', 'SI', 'TJ', 'TV', 'YF', '小ホール', '美術館2']
		remove_co_2013 = ['country', 'past_works', 'temp', 'start_time', 'pre_aud', \
			'sunshine', 'precipitation', 'rerun', 'lang_support', 'free', 'AD', \
			'AS', 'CM', 'EM', 'FC', 'JF', 'LA', 'NAC', 'NDJ', 'TV', 'YF', 'YN', \
			'フォーラム5', '美術館2']
		remove_co_2015 = ['country', 'q_and_a', 'past_works', 'pre_aud', 'sunshine', \
			'precipitation', 'free', 'AD', 'AP', 'AS', 'DS', 'FC', 'IS', 'JF', 'LA', \
			'YN', '小ホール', '美術館2']
		remove_co = [remove_co_2011, remove_co_2013, remove_co_2015]

		pred_au = multi_regr_au.multi_regr_lol(remove_au)
		pred_co = multi_regr_co.multi_regr_lol(remove_co)
