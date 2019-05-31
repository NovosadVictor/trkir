import xlrd, xlwt
import numpy as np

#
# sheet = xlrd.open_workbook('all_avgs_by_12_hours.xlsx')
# sheet = sheet.sheet_by_index(0)
#
# info = []
# for i in range(2, 12):
#     info.append(np.array(sheet.col_values(i)[1:]).astype(float))
#
# for i in range(len(info)):
#     info[i] = (info[i] - np.mean(info[i])) / (np.max(info[i]) - np.min(info[0]))
#
# info = np.array(info).T.tolist()
# sheet = xlwt.Workbook()
# table = sheet.add_sheet('normalized average data')
#
# for i in range(len(info)):
#     for j in range(len(info[i])):
#         table.write(i, j, float(info[i][j]))
#
# sheet.save('norm_all_avgs_by_12_hours.xls')

import xlrd, xlwt
import numpy as np
import csv

from sklearn.decomposition import PCA

sheet = xlrd.open_workbook('all_avgs_by_day.xlsx')
sheet = sheet.sheet_by_index(2)

info = []
for i in range(1, sheet.ncols):
    info.append(np.array(sheet.col_values(i)[1:]).astype(float))

info = np.array(info).T.tolist()

for i in range(len(info)):
    print(np.mean(info[i]), np.max(info[i]), np.min(info[i]), (np.max(info[i]) - np.min(info[i])))
    info[i] = (info[i] - np.mean(info[i])) / (np.max(info[i]) - np.min(info[i]))

info = np.array(info).T
pca_1 = PCA(n_components=2)

pca_1.fit(info)

Xt_1 = pca_1.transform(info)
ev_1 = pca_1.explained_variance_ratio_
print(ev_1)

with open('pca_norm.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(Xt_1)

#
# sheet = xlwt.Workbook()
# table = sheet.add_sheet('normalized average data')
#
# for i in range(len(info)):
#     for j in range(len(info[i])):
#         table.write(i, j, float(info[i][j]))
#
# sheet.save('norm_all_avgs_by_day.xls')

