#-*- coding:utf-8 -*-
"""
Name: get_auth.py
Date: 2016/11/4
usage: python3 get_auth.py 指定年(4桁の整数)
Description:
	yidffの作品一覧ページから指定された年までの監督名を取得し,それをファイル出力する.
"""

import sys
import re
import urllib.request, urllib.error
from bs4 import BeautifulSoup
import pandas as pd

if __name__ == '__main__':
	#指定年に映画祭がやっていたかを判定
	year = int(sys.argv[1])
	if year % 2 != 1 or year < 1989 or year > 2015:
		print("その年に映画祭はありませんでした")
		exit()
	#映画祭の各作品の監督名のリスト(重複あり)
	auth_all = []
	#1989年から2015年まで14回分のデータを取得する
	for i in range(int((year - 1987) / 2)):
		#opener : urlを開くためのオブジェクト
		opener = urllib.request.build_opener()
		#year : 何年開催かを表す変数(89年から隔年で開催)
		url_year = 89 + 2 * i
		#2000年以降は20○○年と表す
		if url_year >= 100:
			url_year = url_year - 100 + 2000

		#urlからhtml情報を取得し,加工できるようにする
		url = "http://www.yidff.jp/" + str(url_year) + "/" + str(url_year) + "list.html"
		html = opener.open(url).read()
		shtml = html.decode('Shift_JIS', 'replace')
		soup = BeautifulSoup(shtml)
		#2009年(11回目)からhtmlの構造が変わるので,if文で分岐処理をする
		if i < 10:
			#<font size="-1">と</font>の間にある文字列をリストに格納
			data = soup.findAll("font", attrs = {"size": "-1"})
			re_tag = re.compile(r'<font.*?>|</font>')
			re_data = [re_tag.sub("", str(d)).split("<font>")[0] for d in data]
		else:
			#<span class="filmdata">と</span>の間にある文字列をリストに格納
			data = soup.findAll("span", attrs = {"class": "filmdata"})
			re_tag = re.compile(r'<span.*?>|</span>')
			re_data = [re_tag.sub("", str(d)).split("<span>")[0] for d in data]
		#記号は全角,半角が混ざっているので,全角に統一
		re_data = [data.replace("[", "［") for data in re_data]
		re_data = [data.replace("]", "］") for data in re_data]
		re_data = [data.replace("/", "／") for data in re_data]
		#監督名以外の情報を削除
		re_data = [data for data in re_data if not("href" in data)]
		re_data = [data for data in re_data if "［" in data]
		re_data = [data for data in re_data if "／" in data]
		re_data = [data.replace("［", "") for data in re_data]
		re_data = [re.sub("／.*?］", "", data) for data in re_data]
		re_data = [re.sub("（.*?）", "", data) for data in re_data]
		re_data = [re.sub("／.*?<／strong>", "", data) for data in re_data]
		re_data = [re.sub("製作：|撮影：", "", data) for data in re_data]
		re_data = [re.sub(" ", "", data) for data in re_data]
		#監督が2人以上いる場合,1人ずつに分ける
		tentative_data = re_data[:]
		for data in tentative_data:
			if "、" in data:
				re_data.remove(data)
				re_data.extend(data.split("、"))
			elif "＋" in data:
				re_data.remove(data)
				re_data.extend(data.split("＋"))

		"""
		if url_year == 2005:
			print(str(url_year) + "年での映画の本数は" + str(len(re_data)))
			re_data = pd.DataFrame(re_data, columns = ['Name'])
			re_data.to_csv("auth_data_" + str(url_year) + ".txt", index = False)
		"""

		#auth_allに各回の監督名を格納する
		auth_all.extend(re_data)

	#auth_allの各監督名をカウントしてauth_dictに{監督名: 作品数}として格納
	auth_set = set(auth_all)
	auth_dict = {}
	for name in auth_set:
		auth_dict[name] = auth_all.count(name)

	#DataFrameにしてcsvファイルとして出力
	auth_data = pd.DataFrame()
	auth_data['name'] = list(auth_dict.keys())
	auth_data['works'] = list(auth_dict.values())
	auth_data.to_csv("auth_data_to_" + str(year) + ".csv", index = False)
