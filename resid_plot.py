# -*- coding:utf-8 -*-
"""
Name: resid_plot.py
Date: 2016/12/16
usage: multi_regr.pyやrandomforest.pyで読み込んで使用する
Description:
	multi_regr.pyやrandomforest.pyで得られた予測値をもとに残差プロットする。
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def resid_plot(y, y_pred, xmin, xmax):
	"""
	与えられた実測値と予測を使用して、残差プロットする関数
	"""
	plt.scatter(y_pred, y_pred - y,
			c = 'blue',
			marker = 'o',
			s = 35,
			alpha = 0.5,
			label = "Plot Residuals")

	plt.xlabel("Predicted values")
	plt.ylabel("Residuals")
	plt.legend(loc = "upper left")
	plt.hlines(y = 0, xmin = xmin, xmax = xmax, color = 'red')
	plt.xlim([xmin, xmax])
	plt.savefig("plot_rf_residuals")
	plt.show()
