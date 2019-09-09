import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression


# 读取历史数据以及需要用作预测的数据
ori_data = pd.read_excel(r'D:\应用程序\PythonProject\project3_DataAnalysis\Data\p1-customers.xlsx')
predict_data = pd.read_excel(r'D:\应用程序\PythonProject\project3_DataAnalysis\Data\p1-mailinglist.xlsx')
# print(ori_data.columns)
# test = ori_data[['Avg Sale Amount', '# Years as Customer','Avg Num Products Purchased']]
# print(test.corr())   #根据相关性简单判断来选取特征
# exit()


# 选取自变量1
indvar1 = ori_data['Avg Num Products Purchased']

# 利用独热编码，选取自变量2
ohe = OneHotEncoder()
temp = ori_data['Customer Segment']
# print(type(temp.values.reshape(-1, 1)))
# exit()
indvar2 = ohe.fit_transform(temp.values.reshape(-1, 1)).toarray()

# 合并两个自变量
indvar = pd.concat([indvar1, pd.DataFrame(indvar2)], axis=1, ignore_index=True)
#print(indvar.head(),type(indvar))
# 选取因变量
devar = ori_data['Avg Sale Amount']

# 测试集（可以一开始把训练集和测试集合并处理?）
test_indvar1 = predict_data['Avg Num Products Purchased']
test_indvar2 = ohe.fit_transform(predict_data['Customer Segment'].values.reshape(-1, 1)).toarray()
test_indvar = pd.concat([test_indvar1, pd.DataFrame(test_indvar2)], axis=1, ignore_index=True)


# 线性回归
model = LinearRegression()
model.fit(indvar, devar)
train = {}
train['intercept'] = model.intercept_
train['coefficient'] = model.coef_
print(train, model.score(indvar, devar))

# 预测
predict_data['Avg Sale Amount'] = model.predict(test_indvar)
#print(predict_data)
predict_data['profit'] = predict_data.apply(lambda x: x['Avg Sale Amount'] * x['Score_Yes']*0.5, axis=1)
print(predict_data)
print(sum(predict_data['profit'])-250*6.5)