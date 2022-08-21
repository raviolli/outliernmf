import basics
import outliers

import pickle

def test_tonmf_outlier(data_df, text_coln):
	from sklearn.feature_extraction.text import TfidfVectorizer
	import matplotlib.pyplot as plt
	import numpy as np

	'''
	0) clean
	1) TF-IDF text
		each document has a MAX_VOC array.
	'''

	data_df[text_coln] = data_df[text_coln].apply(basics.remove_whitespace)
	data_df[text_coln] = data_df[text_coln].apply(basics.remove_urls)
	data_df[text_coln] = data_df[text_coln].apply(basics.remove_phone_num)
	data_df[text_coln] = data_df[text_coln].apply(basics.remove_nonAscii)
	data_df[text_coln] = data_df[text_coln].apply(basics.lower_text)


	print(data_df)

	vectorize_model = TfidfVectorizer()
	result = vectorize_model.fit_transform(data_df[text_coln].to_list())

	print(result.toarray().T.shape)

	model = outliers.TONMF({'rank':4,'alpha':10,'beta':10,'wh_iter':10, 'iter':50})
	results = model.transform(result.toarray())

	print(results)
	with open("tonmf_results.pickle","wb") as fp:
		pickle.dump(results, fp)


	fig, ax = plt.subplots()
	y = np.linalg.norm(results,ord=2,axis=0)
	x = np.array(range(len(y)))
	ax.stem(x, y)
	plt.show()

if __name__ == '__main__':
	
	import pandas as pd
	df = pd.read_csv("../data_scrapers/split_results.csv")
	print(df)
	df = df.dropna()

	test_tonmf_outlier(df, 'text')