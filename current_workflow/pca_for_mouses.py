# # import csv
# # import xlrd
# #
# # import numpy as np
# # from sklearn.externals import joblib
# #
# # from matplotlib import pyplot as plt
# # import matplotlib.patches as mpatches
# #
# from sklearn.decomposition import PCA, LatentDirichletAllocation
# #
# # features = ['date', 'head_body_dist', 'body_speed_x', 'body_speed_y',
# #             'head_speed_x', 'head_speed_y', 'head_body_dist_change',
# #             'head_rotation', 'body_speed', 'head_speed',
# #             'body_acceleration', 'head_acceleration',
# #             'body_coord_x', 'body_coord_y',
# #             'head_coord_x', 'head_coord_y',
# #             'temp',
# #             ]
# #
# # sheet = xlrd.open_workbook('Healthy period_mouse.xlsx')
# # sheet = sheet.sheet_by_index(0)
# #
# # info = []
# # print(sheet.ncols)
# # for i in range(1, 16 * 2 + 1):
# #     info.append(sheet.col_values(i)[1:])
# #
# # dates = sheet.col_values(0)[1:]
# #
# # info = [info[i] for i in range(len(info)) if i not in [2, 4, 18, 20]]
# #
# # splits = [i for i in range(1, len(dates) - 1) if dates[i] == '' and dates[i - 1] != '']
# # splits.insert(0, 0)
# # splits.append(len(dates) - 1)
# #
# # splitted_by_days = [[info[f][splits[i - 1]: splits[i]] for f in range(len(info))] for i in range(1, len(splits))]
# #
# # splitted_by_days = [splitted_by_days[d] for d in range(len(splitted_by_days)) if d not in [i for i in range(3, 11)]]
# #
# # for d in range(len(splitted_by_days)):
# #     print(d)
# #     splitted_by_days[d] = np.array(splitted_by_days[d]).T.tolist()
# #
# #     splitted_by_days[d] = splitted_by_days[d][2:]#true_splitted_day
# #     splitted_by_days[d] = np.array(splitted_by_days[d]).T.astype(float).tolist()
# #     splitted_by_days[d] = np.mean(splitted_by_days[d], axis=1)
# #
# #
# # splitted_by_mouse = [[splitted_by_days[d][m * len(splitted_by_days[d]) // 2: (m + 1) * len(splitted_by_days[d]) // 2]
# #                       for d in range(len(splitted_by_days))]
# #                      for m in range(2)]
# #
# # print(splitted_by_mouse)
# # print(len(splitted_by_mouse[0]))
# #
# # splitted_by_mouse[0] = [splitted_by_mouse[0][1], splitted_by_mouse[0][3], splitted_by_mouse[0][4]]
# # splitted_by_mouse[1] = [splitted_by_mouse[1][1], splitted_by_mouse[1][3], splitted_by_mouse[1][4]]
# #
# # splitted_by_mouse = np.array(splitted_by_mouse).tolist()
# #
# # pca_1 = PCA(n_components=2)
# #
# # pca_1.fit(splitted_by_mouse[0] + splitted_by_mouse[1])
# #
# # Xt_1 = pca_1.transform(splitted_by_mouse[0] + splitted_by_mouse[1])
# # ev_1 = pca_1.explained_variance_ratio_
# # print(ev_1)
# #
# # to_csv = []
# # to_csv.append(['id', 'L_PC1 %d' % int(ev_1[0] * 100), 'L_PC2 %d' % int(ev_1[1] * 100)])
# #
# # for k in range(len(splitted_by_mouse[0])):
# #     to_csv.append(['L_%d' % (k + 1), Xt_1[k, 0], Xt_1[k, 1]])
# #
# # for k in range(len(splitted_by_mouse[0])):
# #     to_csv.append(['R_%d' % (k + 1), Xt_1[k, 0], Xt_1[k, 1]])
# #
# # with open('healthy_pca.csv', 'w') as csvfile:
# #     writer = csv.writer(csvfile)
# #     writer.writerows(to_csv)
# #
# #
# # sheet = xlrd.open_workbook('Sick period_mouse.xlsx')
# # sheet = sheet.sheet_by_index(0)
# #
# # info = []
# # print(sheet.ncols)
# # for i in range(1, 16 * 2 + 1):
# #     info.append(sheet.col_values(i)[1:])
# #
# # dates = sheet.col_values(0)[1:]
# #
# # info = [info[i] for i in range(len(info)) if i not in [2, 4, 18, 20]]
# #
# # splits = [i for i in range(1, len(dates) - 1) if dates[i] == '' and dates[i - 1] != '']
# # splits.insert(0, 0)
# # splits.append(len(dates) - 1)
# #
# # splitted_by_days = [[info[f][splits[i - 1]: splits[i]] for f in range(len(info))] for i in range(1, len(splits))]
# #
# # splitted_by_days = [splitted_by_days[d] for d in range(len(splitted_by_days)) if d not in [i for i in range(0, 8)]]
# #
# # for d in range(len(splitted_by_days)):
# #     print(d)
# #     splitted_by_days[d] = np.array(splitted_by_days[d]).T.tolist()
# #
# #     splitted_by_days[d] = splitted_by_days[d][2:]#true_splitted_day
# #     splitted_by_days[d] = np.array(splitted_by_days[d]).T.astype(float).tolist()
# #     splitted_by_days[d] = np.mean(splitted_by_days[d], axis=1)
# #
# #
# # splitted_by_mouse = [[splitted_by_days[d][m * len(splitted_by_days[d]) // 2: (m + 1) * len(splitted_by_days[d]) // 2]
# #                       for d in range(len(splitted_by_days))]
# #                      for m in range(2)]
# #
# # print(splitted_by_mouse)
# # print(len(splitted_by_mouse[0]))
# #
# # splitted_by_mouse[0] = [np.mean(splitted_by_mouse[0][:8], axis=1), np.mean(splitted_by_mouse[0][8:],
# # splitted_by_mouse[1] = [splitted_by_mouse[1][1], splitted_by_mouse[1][3], splitted_by_mouse[1][4]]
# #
# # splitted_by_mouse = np.array(splitted_by_mouse).tolist()
# #
# # pca_1 = PCA(n_components=2)
# #
# # pca_1.fit(splitted_by_mouse[0] + splitted_by_mouse[1])
# #
# # Xt_1 = pca_1.transform(splitted_by_mouse[0] + splitted_by_mouse[1])
# # ev_1 = pca_1.explained_variance_ratio_
# # print(ev_1)
# #
# # to_csv = []
# # to_csv.append(['id', 'L_PC1 %d' % int(ev_1[0] * 100), 'L_PC2 %d' % int(ev_1[1] * 100)])
# #
# # for k in range(len(splitted_by_mouse[0])):
# #     to_csv.append(['L_%d' % (k + 1), Xt_1[k, 0], Xt_1[k, 1]])
# #
# # for k in range(len(splitted_by_mouse[0])):
# #     to_csv.append(['R_%d' % (k + 1), Xt_1[k, 0], Xt_1[k, 1]])
# #
# # with open('sick_pca.csv', 'w') as csvfile:
# #     writer = csv.writer(csvfile)
# #     writer.writerows(to_csv)
#
#
# import xlrd
# import numpy as np
#
# import csv
#
#
# sheet = xlrd.open_workbook('Healthy period_mouse.xlsx')
# sheet = sheet.sheet_by_index(0)
#
# info = []
# print(sheet.ncols)
# for i in range(1, 16 * 2 + 1):
#     info.append(sheet.col_values(i)[1:])
#
# info = [info[i] for i in range(len(info)) if i not in [2, 4, 18, 20, 11, 12, 13, 14, 27, 28, 29, 30]]
#
# dates = sheet.col_values(0)[1:]
#
# splits = [i for i in range(len(dates) - 1) if dates[i] == '' and dates[i - 1] != '']
# splits.insert(0, 0)
# splits.append(len(dates) - 1)
#
# print(splits)
#
# splitted = [[info[i][splits[j - 1]: splits[j]] for i in range(len(info))] for j in range(1, len(splits))]
#
# to_forget = [splitted[i] for i in range(3, 11)]
#
# splitted = [split for split in splitted if split not in to_forget]
#
# for k in range(len(splitted)):
#     splitted[k] = np.array(splitted[k]).T.tolist()
# #    splitted[k] = splitted[k]['' not in splitted[k]]
#     true_splitted_k = []
#     for i in range(len(splitted[k])):
#         if '' not in splitted[k][i][:]:
#             true_splitted_k.append(splitted[k][i])
#
#     splitted[k] = np.array(true_splitted_k).astype(float).T.tolist()
#
#
# avgs_h = [[i for i in range(len(info))] for j in range(len(splitted) * 2)]
#
# for k in range(len(splitted)):
#     for j in range(len(info)):
#         avgs_h[2 * k][j] = np.mean(splitted[k][j][:len(splitted[k][j]) // 2])
#         avgs_h[2 * k + 1][j] = np.mean(splitted[k][j][len(splitted[k][j]) // 2:])
#
# features = ['head_body_dist', 'body_speed_x',# 'body_speed_y',
#             'head_speed_x', #'head_speed_y',
#             'head_body_dist_change',
#             'head_rotation', 'body_speed', 'head_speed',
#             'body_acceleration', 'head_acceleration',
#          #   'body_coord_x', 'body_coord_y',
#         #    'head_coord_x', 'head_coord_y',
#             'temp',
#             ]
#
# features = ['experiment_id'] + ['L_' + feature + '_avg' for feature in features] + ['R_' + feature + '_avg' for feature in features]
#
# forgetted_avgs = []
# for z in range(len(to_forget)):
#     to_forget[z] = np.array(to_forget[z]).T.tolist()
#
#     true_splitted_k = []
#     for i in range(len(to_forget[z])):
#         if '' not in to_forget[z][i][len(to_forget[z][i]) // 2:]:
#             true_splitted_k.append(to_forget[z][i][len(to_forget[z][i]) // 2 :])
#
#     to_forget[z] = np.array(true_splitted_k).astype(float).T.tolist()
#
#     avg_forget = [0 for i in range(len(info) // 2)] + [np.mean(to_forget[z][i][:len(to_forget[z][i]) // 2]) for i in range(len(to_forget[z]))]
#     avg_forget_2 = [0 for i in range(len(info) // 2)] + [np.mean(to_forget[z][i][len(to_forget[z][i]) // 2:]) for i in range(len(to_forget[z]))]
#
#     forgetted_avgs.append(avg_forget)
#     forgetted_avgs.append(avg_forget_2)
#
#
# avgs_h = avgs_h[:6] + forgetted_avgs + avgs_h[6:]#.insert(1, avg_forget)
#
# sheet = xlrd.open_workbook('Sick period_mouse.xlsx')
# sheet = sheet.sheet_by_index(0)
#
# info = []
# print(sheet.ncols)
# for i in range(1, 16 * 2 + 1):
#     info.append(sheet.col_values(i)[2:])
#
# info = [info[i] for i in range(len(info)) if i not in [2, 4, 18, 20, 11, 12, 13, 14, 27, 28, 29, 30]]
#
# dates = sheet.col_values(0)[2:]
#
# splits = [i for i in range(len(dates) - 1) if dates[i] == '' and dates[i - 1] != '']
# splits.insert(0, 0)
# splits.append(len(dates) - 1)
#
# print(splits)
#
# splitted = [[info[i][splits[j - 1]: splits[j]] for i in range(len(info))] for j in range(1, len(splits))]
#
# to_forget = [splitted[i] for i in range(8)]
# print(to_forget[0])
#
# splitted = [split for split in splitted if split not in to_forget]
#
# for k in range(len(splitted)):
#     splitted[k] = np.array(splitted[k]).T.tolist()
#     true_splitted_k = []
#     for i in range(len(splitted[k])):
#         if '' not in splitted[k][i][:]:
#             true_splitted_k.append(splitted[k][i])
#
#     splitted[k] = np.array(true_splitted_k).astype(float).T.tolist()
#
# print(len(splitted[0]))
#
# avgs_s = [[i for i in range(len(info))] for j in range(len(splitted) * 2)]
# for k in range(len(splitted)):
#     for j in range(len(info)):
#         print(k, j)
#         avgs_s[2 * k][j] = np.mean(splitted[k][j][:len(splitted[k][j]) // 2])
#         avgs_s[2 * k + 1][j] = np.mean(splitted[k][j][len(splitted[k][j]) // 2:])
#
# features = ['head_body_dist', 'body_speed_x',# 'body_speed_y',
#             'head_speed_x',# 'head_speed_y',
#             'head_body_dist_change',
#             'head_rotation', 'body_speed', 'head_speed',
#             'body_acceleration', 'head_acceleration',
#    #         'body_coord_x', 'body_coord_y',
#    #         'head_coord_x', 'head_coord_y',
#             'temp',
#             ]
#
# #features = ['experiment_id'] + ['day_id'] + ['L_' + feature + '_avg' for feature in features] + ['R_' + feature + '_avg' for feature in features]
# features = ['experiment_id'] + ['mouse_id'] + [feature + '_avg' for feature in features]
#
# forgetted_avgs = []
# for z in range(len(to_forget)):
#     to_forget[z] = np.array(to_forget[z]).T.tolist()
#
#     true_splitted_k = []
#     for i in range(len(to_forget[z])):
#         if '' not in to_forget[z][i][:len(to_forget[z][i]) // 2]:
#             true_splitted_k.append(to_forget[z][i][:len(to_forget[z][i]) // 2])
#
#     to_forget[z] = np.array(true_splitted_k).astype(float).T.tolist()
#
#     avg_forget = [np.mean(to_forget[z][i][:len(to_forget[z][i]) // 2]) for i in range(len(to_forget[z]))] + [0 for i in range(len(info) // 2)]
#     avg_forget_2 = [np.mean(to_forget[z][i][len(to_forget[z]) // 2:]) for i in range(len(to_forget[z]))] + [0 for i in range(len(info) // 2)]
#
#     forgetted_avgs.append(avg_forget)
#     forgetted_avgs.append(avg_forget_2)
#
# avgs_s = forgetted_avgs + avgs_s
# avgs = avgs_h + avgs_s
# #avgs = avgs_s
#
# labels = ['1 1 1 2 2 2 2 2 2 2 2 3 3'.split(' ') * 2, '1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3'.split(' ') * 2]
# days = ['1 2 3 1 2 3 4 5 6 7 8 1 1'.split(' ') * 2, '1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 1 2 3'.split(' ') * 2]
# ccc = [features]
# #ccc += [['H_{}'.format(labels[0][i]), 'D_{}'.format(days[0][i])] + avgs[i] for i in range(len(avgs_h))]
# ccc += [['Healthy', 'Left_{}'.format(days[0][i])] + avgs[i][:len(avgs[i]) // 2] for i in range(len(avgs_h))]
# ccc += [['Healthy', 'Right_{}'.format(days[0][i])] + avgs[i][len(avgs[i]) // 2:] for i in range(len(avgs_h))]
# #ccc += [['S_{}'.format(labels[1][i - len(avgs_h)]), 'D_{}'.format(days[1][i - len(avgs_h)])] + avgs[i] for i in range(len(avgs_h), len(avgs))]
# #ccc += [['Sick', 'Left'.format(days[1][i - len(avgs_h)])] + avgs[i][:len(avgs[i]) // 2] for i in range(len(avgs_h), len(avgs))]
# ccc += [['Sick', 'PV1'] + avgs[i][:len(avgs[i]) // 2] for i in range(len(avgs_h), len(avgs))]
# ccc += [['Sick', 'PV1 + DI'] + avgs[i][len(avgs[i]) // 2:] for i in range(len(avgs_h), len(avgs))]
# #ccc += [['Sick', 'Right'.format(days[1][i - len(avgs_h)])] + avgs[i][len(avgs[i]) // 2:] for i in range(len(avgs_h), len(avgs))]
# #
# with open('all_avgs_by_day.csv', 'w') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(ccc)

# avgs_by_mouse = [[avgs[i][m * len(avgs[i]) // 2: (m + 1) * len(avgs[i]) // 2] for i in range(len(avgs))] for m in range(2)]
#
#
# print(avgs_by_mouse)
#
# pca_1 = PCA(n_components=2)
#
# pca_1.fit(avgs_by_mouse[0] + avgs_by_mouse[1])
#
# Xt_1 = pca_1.transform(avgs_by_mouse[0] + avgs_by_mouse[1])
# ev_1 = pca_1.explained_variance_ratio_
# print(ev_1)
#
# to_csv = []
# to_csv.append(['id', 'L_PC1 %d' % int(ev_1[0] * 100), 'L_PC2 %d' % int(ev_1[1] * 100)])
#
# for k in range(len(avgs_by_mouse[0])):
#     if 2 < k < 11:
#         pass
#     if k < 3 + 8 + 2:
#         to_csv.append(['healthy_%d' % (k + 1), Xt_1[k, 0], Xt_1[k, 1]])
#     else:
#         to_csv.append(['PV1_%d' % (k + 1), Xt_1[k, 0], Xt_1[k, 1]])
#
# for k in range(len(avgs_by_mouse[1])):
#     if k < 8:
#         pass
#     if k < 3 + 8 + 2:
#         to_csv.append(['healthy_%d' % (k + 1), Xt_1[k + len(avgs_by_mouse[0]), 0], Xt_1[k + len(avgs_by_mouse[0]), 1]])
#     else:
#         to_csv.append(['(PV1 + DI)_%d' % (k + 1), Xt_1[k + len(avgs_by_mouse[0]), 0], Xt_1[k + len(avgs_by_mouse[0]), 1]])
#
#
# with open('by_day_pca.csv', 'w') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(to_csv)
#
#
# lda = LatentDirichletAllocation(n_components=3, random_state=0)
# lda.fit(avgs_by_mouse[0] + avgs_by_mouse[1])
#
# XT_1 = lda.transform(avgs_by_mouse[0] + avgs_by_mouse[1])
#
#
# to_csv = []
# to_csv.append(['id', 'lda_1', 'lda_2'])
#
# for k in range(len(avgs_by_mouse[0])):
#     if 2 < k < 11:
#         pass
#     if k < 3 + 8 + 2:
#         to_csv.append(['healthy_%d' % (k + 1), Xt_1[k, 0], Xt_1[k, 1]])
#     else:
#         to_csv.append(['PV1_%d' % (k + 1), Xt_1[k, 0], Xt_1[k, 1]])
#
# for k in range(len(avgs_by_mouse[1])):
#     if k < 8:
#         pass
#     if k < 3 + 8 + 2:
#         to_csv.append(['healthy_%d' % (k + 1), Xt_1[k + len(avgs_by_mouse[0]), 0], Xt_1[k + len(avgs_by_mouse[0]), 1]])
#     else:
#         to_csv.append(['(PV1 + DI)_%d' % (k + 1), Xt_1[k + len(avgs_by_mouse[0]), 0], Xt_1[k + len(avgs_by_mouse[0]), 1]])
#
#
# with open('h_vs_s_all_lda.csv', 'w') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(to_csv)
#
#
#
#     day_1 = data[:398]
#
#     day_1 = day_1[np.where(day_1 != '')]
#     day_1 = day_1[np.where(day_1 != 'PV1')]
#
#     day_3_to_compare = data[1202 - 50: 1643 + 50]
#     day_3_to_compare = day_3_to_compare[np.where(day_3_to_compare != '')]
#     day_3_to_compare = day_3_to_compare[np.where(day_3_to_compare != 'PV1')]
#
#     day_3 = data[1202: 1643]
#     day_3 = day_3[np.where(day_3 != '')]
#     day_3 = day_3[np.where(day_3 != 'PV1')]
#
#     day_1 = day_1.astype(float).tolist()
#     day_3 = day_3.astype(float).tolist()
#     day_3_to_compare = np.array(day_3_to_compare).astype(float).tolist()
#
#     data = data[np.where(data != '')]
#     data = data[np.where(data != 'PV1')]
#     data = data.astype(float).tolist()
#
#     # plt.figure()
#     # plt.plot([i for i in range(len(data))], data)
#     # plt.ylim(top=20)
#     # plt.show()
#
#     period = 20
#     mean_data = [0 for i in range(period, len(data) - period)]
#     for i in range(period, len(data) - period):
#         mean_data[i - period] = np.mean(data[i - period: i + period])
#
#     mean_day_1 = [0 for i in range(period, len(day_1) - period)]
#     for i in range(period, len(day_1) - period):
#         mean_day_1[i - period] = np.mean(day_1[i - period: i + period])
#
#     mean_day_3 = [0 for i in range(period, len(day_3) - period)]
#     for i in range(period, len(day_3) - period):
#         mean_day_3[i - period] = np.mean(day_3[i - period: i + period])
#
#     plt.figure(figsize=(14, 8))
#     # plt.subplot(211)
#     # plt.plot([i for i in range(len(mean_day_1))], mean_day_1, [i for i in range(len(mean_day_3))], mean_day_3)
#     # #plt.ylim(top=20)
#
#     mean_day_3_to_compare = [0 for i in range(period, len(day_3_to_compare) - period)]
#     for i in range(period, len(day_3_to_compare) - period):
#         mean_day_3_to_compare[i - period] = np.mean(day_3_to_compare[i - period: i + period])
#
#     dist = metric(mean_day_1, mean_day_3_to_compare[:len(mean_day_1)])
#     best = 0
#     error = dist
#     print(error)
#
#     for i in range(1, len(mean_day_3_to_compare) - len(mean_day_1), 2):
#         cur_to_compare = mean_day_3_to_compare[i: i + len(mean_day_1)]
#         cur_dist = metric(cur_to_compare, mean_day_1)
#         if cur_dist < dist:
#             dist = cur_dist
#             best = i
#             error = cur_dist
#
#     # plt.subplot(211)
#     # plt.plot([i for i in range(len(mean_day_1))], mean_day_1, 'C0',[i for i in range(len(mean_day_1))], mean_day_3_to_compare[best: best + len(mean_day_1)], 'C1')
#
#     print(best, error)
#     # plt.ylim(top=20)
#
#     p1 = np.array(mean_day_1).reshape(-1, 1)
#     print(p1)
#     print(p1)
#     print(p1.shape)
#     p2 = np.array(mean_day_3_to_compare[best: best + len(mean_day_1)]).reshape(len(mean_day_1), 1)
#     print(p2.shape)
#
#     # reg = LinearRegression(copy_X=True).fit(p1, p2)
#     # print(reg.score(p1, p2))
#     # print(reg.coef_, reg.intercept_)
#     # a, b = reg.coef_[0][0], reg.intercept_[0]
#     # res = np.dot(p1, reg.coef_) + reg.intercept_
#     #
#     # res = res.reshape(-1, len(mean_day_1))[0].tolist()
#     # p2 = p2.reshape(1, -1)[0].tolist()
#     # #plt.subplot(212)
#     # plt.plot([i for i in range(len(mean_day_1))], mean_day_1, 'C0', [i for i in range(len(mean_day_1))], p2, 'C1', [i for i in range(len(mean_day_1))], res, 'C2')
#     # plt.title('y = {0:.2f} * x + {1:.2f}'.format(a, b))
#     # #plt.ylim(top=20)
#     # if 's' in sys.argv:
#     #     plt.show()
#
#     kernel = C(1, (1e-2, 1e2)) * RBF(27, (1e-2, 1e2))
#     reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9).fit(p1, p2)
#     res, sigma = reg.predict(p1, return_std=True)
#
#     res = res.reshape(-1, len(mean_day_1))[0].tolist()
#     p2 = p2.reshape(1, -1)[0].tolist()
#     # plt.subplot(212)
#     x = [i for i in range(len(mean_day_1))]
#     plt.plot(x, mean_day_1, 'C0', x, p2, 'C1', x, res, 'C2')
#     plt.fill(np.concatenate([x, x[::-1]]),
#              np.concatenate([res - 1.9600 * sigma,
#                              (res + 1.9600 * sigma)[::-1]]),
#              alpha=.5, fc='b', ec='None', label='95% confidence interval')
#     plt.xlabel('$x$')
#     # plt.title('y = {0:.2f} * x + {1:.2f}'.format(a, b))
#     plt.ylim(top=35, bottom=20)
#     if 's' in sys.argv:
#         plt.show()
#
#
#
#
# # TRASH
# #
# # plt.plot(mean_days[0], 'royalblue')
# # plt.plot(mean_days[1], 'navy')
# # plt.plot(mean_DIP_temp, 'yellow')
# # plt.plot(healthy, 'g')
# # # plt.plot(mean_healthy_temp)
# # plt.plot(mean_PV1_temp)
# # plt.plot(mean_DIP_temp)
# plt.axis([0, len(mean_DIP_temp), np.min(mean_DIP_temp) - 6, np.max(mean_DIP_temp) + 6])
# plt.xticks([])#[(len(mean_healthy_temp) // 72) * i for i in range(len(mean_healthy_temp) // 6 + 1)], [(start_time + timedelta(hours=i)).strftime("%d %H:%M") for i in range(24 * 3)], rotation=60)
# # plt.axis([0, len(mean_healthy_temp), np.min(mean_healthy_temp) - 2, np.max(mean_healthy_temp) + 2])
# plt.title('Average speed')
# plt.show()
#
# to_csv = [['avg_healthy'] + mean_healthy_temp[:400], ['avg_PV1'] + mean_PV1_temp[:400], ['avg_DIP'] + mean_DIP_temp[:400]]
# #tt = np.array([mean_healthy_temp[:400], mean_PV1_temp[:400], mean_DIP_temp[:400]])
# #print(tt.shape)
# #tt = tt.reshape(402, 3)
# #to_csv += tt
# #to_csv = np.array(to_csv).T.tolist()
#
# #to_csv = np.array(to_csv).T.tolist()
# with open('temp_avgs.csv', 'w') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(to_csv)
#

# import csv
# import xlrd
#
# import numpy as np
# from sklearn.externals import joblib
#
# from matplotlib import pyplot as plt
# import matplotlib.patches as mpatches
#
from sklearn.decomposition import PCA, LatentDirichletAllocation
#
# features = ['date', 'head_body_dist', 'body_speed_x', 'body_speed_y',
#             'head_speed_x', 'head_speed_y', 'head_body_dist_change',
#             'head_rotation', 'body_speed', 'head_speed',
#             'body_acceleration', 'head_acceleration',
#             'body_coord_x', 'body_coord_y',
#             'head_coord_x', 'head_coord_y',
#             'temp',
#             ]
#
# sheet = xlrd.open_workbook('Healthy period_mouse.xlsx')
# sheet = sheet.sheet_by_index(0)
#
# info = []
# print(sheet.ncols)
# for i in range(1, 16 * 2 + 1):
#     info.append(sheet.col_values(i)[1:])
#
# dates = sheet.col_values(0)[1:]
#
# info = [info[i] for i in range(len(info)) if i not in [2, 4, 18, 20]]
#
# splits = [i for i in range(1, len(dates) - 1) if dates[i] == '' and dates[i - 1] != '']
# splits.insert(0, 0)
# splits.append(len(dates) - 1)
#
# splitted_by_days = [[info[f][splits[i - 1]: splits[i]] for f in range(len(info))] for i in range(1, len(splits))]
#
# splitted_by_days = [splitted_by_days[d] for d in range(len(splitted_by_days)) if d not in [i for i in range(3, 11)]]
#
# for d in range(len(splitted_by_days)):
#     print(d)
#     splitted_by_days[d] = np.array(splitted_by_days[d]).T.tolist()
#
#     splitted_by_days[d] = splitted_by_days[d][2:]#true_splitted_day
#     splitted_by_days[d] = np.array(splitted_by_days[d]).T.astype(float).tolist()
#     splitted_by_days[d] = np.mean(splitted_by_days[d], axis=1)
#
#
# splitted_by_mouse = [[splitted_by_days[d][m * len(splitted_by_days[d]) // 2: (m + 1) * len(splitted_by_days[d]) // 2]
#                       for d in range(len(splitted_by_days))]
#                      for m in range(2)]
#
# print(splitted_by_mouse)
# print(len(splitted_by_mouse[0]))
#
# splitted_by_mouse[0] = [splitted_by_mouse[0][1], splitted_by_mouse[0][3], splitted_by_mouse[0][4]]
# splitted_by_mouse[1] = [splitted_by_mouse[1][1], splitted_by_mouse[1][3], splitted_by_mouse[1][4]]
#
# splitted_by_mouse = np.array(splitted_by_mouse).tolist()
#
# pca_1 = PCA(n_components=2)
#
# pca_1.fit(splitted_by_mouse[0] + splitted_by_mouse[1])
#
# Xt_1 = pca_1.transform(splitted_by_mouse[0] + splitted_by_mouse[1])
# ev_1 = pca_1.explained_variance_ratio_
# print(ev_1)
#
# to_csv = []
# to_csv.append(['id', 'L_PC1 %d' % int(ev_1[0] * 100), 'L_PC2 %d' % int(ev_1[1] * 100)])
#
# for k in range(len(splitted_by_mouse[0])):
#     to_csv.append(['L_%d' % (k + 1), Xt_1[k, 0], Xt_1[k, 1]])
#
# for k in range(len(splitted_by_mouse[0])):
#     to_csv.append(['R_%d' % (k + 1), Xt_1[k, 0], Xt_1[k, 1]])
#
# with open('healthy_pca.csv', 'w') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(to_csv)
#
#
# sheet = xlrd.open_workbook('Sick period_mouse.xlsx')
# sheet = sheet.sheet_by_index(0)
#
# info = []
# print(sheet.ncols)
# for i in range(1, 16 * 2 + 1):
#     info.append(sheet.col_values(i)[1:])
#
# dates = sheet.col_values(0)[1:]
#
# info = [info[i] for i in range(len(info)) if i not in [2, 4, 18, 20]]
#
# splits = [i for i in range(1, len(dates) - 1) if dates[i] == '' and dates[i - 1] != '']
# splits.insert(0, 0)
# splits.append(len(dates) - 1)
#
# splitted_by_days = [[info[f][splits[i - 1]: splits[i]] for f in range(len(info))] for i in range(1, len(splits))]
#
# splitted_by_days = [splitted_by_days[d] for d in range(len(splitted_by_days)) if d not in [i for i in range(0, 8)]]
#
# for d in range(len(splitted_by_days)):
#     print(d)
#     splitted_by_days[d] = np.array(splitted_by_days[d]).T.tolist()
#
#     splitted_by_days[d] = splitted_by_days[d][2:]#true_splitted_day
#     splitted_by_days[d] = np.array(splitted_by_days[d]).T.astype(float).tolist()
#     splitted_by_days[d] = np.mean(splitted_by_days[d], axis=1)
#
#
# splitted_by_mouse = [[splitted_by_days[d][m * len(splitted_by_days[d]) // 2: (m + 1) * len(splitted_by_days[d]) // 2]
#                       for d in range(len(splitted_by_days))]
#                      for m in range(2)]
#
# print(splitted_by_mouse)
# print(len(splitted_by_mouse[0]))
#
# splitted_by_mouse[0] = [np.mean(splitted_by_mouse[0][:8], axis=1), np.mean(splitted_by_mouse[0][8:],
# splitted_by_mouse[1] = [splitted_by_mouse[1][1], splitted_by_mouse[1][3], splitted_by_mouse[1][4]]
#
# splitted_by_mouse = np.array(splitted_by_mouse).tolist()
#
# pca_1 = PCA(n_components=2)
#
# pca_1.fit(splitted_by_mouse[0] + splitted_by_mouse[1])
#
# Xt_1 = pca_1.transform(splitted_by_mouse[0] + splitted_by_mouse[1])
# ev_1 = pca_1.explained_variance_ratio_
# print(ev_1)
#
# to_csv = []
# to_csv.append(['id', 'L_PC1 %d' % int(ev_1[0] * 100), 'L_PC2 %d' % int(ev_1[1] * 100)])
#
# for k in range(len(splitted_by_mouse[0])):
#     to_csv.append(['L_%d' % (k + 1), Xt_1[k, 0], Xt_1[k, 1]])
#
# for k in range(len(splitted_by_mouse[0])):
#     to_csv.append(['R_%d' % (k + 1), Xt_1[k, 0], Xt_1[k, 1]])
#
# with open('sick_pca.csv', 'w') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(to_csv)


import xlrd
import numpy as np

import csv


sheet = xlrd.open_workbook('Healthy period_mouse.xlsx')
sheet = sheet.sheet_by_index(0)

info = []
print(sheet.ncols)
for i in range(1, 16 * 2 + 1):
    info.append(sheet.col_values(i)[1:])

info = [info[i] for i in range(len(info)) if i not in [2, 4, 18, 20, 11, 12, 13, 14, 27, 28, 29, 30]]

dates = sheet.col_values(0)[1:]

splits = [i for i in range(len(dates) - 1) if dates[i] == '' and dates[i - 1] != '']
splits.insert(0, 0)
splits.append(len(dates) - 1)

print(splits)

splitted = [[info[i][splits[j - 1]: splits[j]] for i in range(len(info))] for j in range(1, len(splits))]

to_forget = [splitted[i] for i in range(3, 11)]

splitted = [split for split in splitted if split not in to_forget]

for k in range(len(splitted)):
    splitted[k] = np.array(splitted[k]).T.tolist()
#    splitted[k] = splitted[k]['' not in splitted[k]]
    true_splitted_k = []
    for i in range(len(splitted[k])):
        if '' not in splitted[k][i][:]:
            true_splitted_k.append(splitted[k][i])

    splitted[k] = np.array(true_splitted_k).astype(float).T.tolist()


avgs_h = [[i for i in range(len(info))] for j in range(len(splitted))]

for k in range(len(splitted)):
    for j in range(len(info)):
        avgs_h[k][j] = (np.mean(splitted[k][j]))

features = ['head_body_dist', 'body_speed_x',# 'body_speed_y',
            'head_speed_x', #'head_speed_y',
            'head_body_dist_change',
            'head_rotation', 'body_speed', 'head_speed',
            'body_acceleration', 'head_acceleration',
         #   'body_coord_x', 'body_coord_y',
        #    'head_coord_x', 'head_coord_y',
            'temp',
            ]

features = ['experiment_id'] + ['L_' + feature + '_avg' for feature in features] + ['R_' + feature + '_avg' for feature in features]

forgetted_avgs = []
for z in range(len(to_forget)):
    to_forget[z] = np.array(to_forget[z]).T.tolist()

    true_splitted_k = []
    for i in range(len(to_forget[z])):
        if '' not in to_forget[z][i][len(to_forget[z][i]) // 2:]:
            true_splitted_k.append(to_forget[z][i][len(to_forget[z][i]) // 2 :])

    to_forget[z] = np.array(true_splitted_k).astype(float).T.tolist()

    avg_forget = [0 for i in range(len(info) // 2)] + [np.mean(to_forget[z][i]) for i in range(len(to_forget[z]))]

    forgetted_avgs.append(avg_forget)


avgs_h = avgs_h[:3] + forgetted_avgs + avgs_h[3:]#.insert(1, avg_forget)

sheet = xlrd.open_workbook('Sick period_mouse.xlsx')
sheet = sheet.sheet_by_index(0)

info = []
print(sheet.ncols)
for i in range(1, 16 * 2 + 1):
    info.append(sheet.col_values(i)[2:])

info = [info[i] for i in range(len(info)) if i not in [2, 4, 18, 20, 11, 12, 13, 14, 27, 28, 29, 30]]

dates = sheet.col_values(0)[2:]

splits = [i for i in range(len(dates) - 1) if dates[i] == '' and dates[i - 1] != '']
splits.insert(0, 0)
splits.append(len(dates) - 1)

print(splits)

splitted = [[info[i][splits[j - 1]: splits[j]] for i in range(len(info))] for j in range(1, len(splits))]

to_forget = [splitted[i] for i in range(8)]
print(to_forget[0])

splitted = [split for split in splitted if split not in to_forget]

for k in range(len(splitted)):
    splitted[k] = np.array(splitted[k]).T.tolist()
    true_splitted_k = []
    for i in range(len(splitted[k])):
        if '' not in splitted[k][i][:]:
            true_splitted_k.append(splitted[k][i])

    splitted[k] = np.array(true_splitted_k).astype(float).T.tolist()

print(len(splitted[0]))

avgs_s = [[i for i in range(len(info))] for j in range(len(splitted))]
for k in range(len(splitted)):
    for j in range(len(info)):
        print(k, j)
        avgs_s[k][j] = np.mean(splitted[k][j])

features = ['head_body_dist', 'body_speed_x',# 'body_speed_y',
            'head_speed_x',# 'head_speed_y',
            'head_body_dist_change',
            'head_rotation', 'body_speed', 'head_speed',
            'body_acceleration', 'head_acceleration',
   #         'body_coord_x', 'body_coord_y',
   #         'head_coord_x', 'head_coord_y',
            'temp',
            ]

#features = ['experiment_id'] + ['day_id'] + ['L_' + feature + '_avg' for feature in features] + ['R_' + feature + '_avg' for feature in features]
features = ['experiment_id'] + ['mouse_id'] + [feature + '_avg' for feature in features]

forgetted_avgs = []
for z in range(len(to_forget)):
    to_forget[z] = np.array(to_forget[z]).T.tolist()

    true_splitted_k = []
    for i in range(len(to_forget[z])):
        if '' not in to_forget[z][i][:len(to_forget[z][i]) // 2]:
            true_splitted_k.append(to_forget[z][i][:len(to_forget[z][i]) // 2])

    to_forget[z] = np.array(true_splitted_k).astype(float).T.tolist()

    avg_forget = [np.mean(to_forget[z][i]) for i in range(len(to_forget[z]))] + [0 for i in range(len(info) // 2)]

    forgetted_avgs.append(avg_forget)

avgs_s = forgetted_avgs + avgs_s
avgs = avgs_h + avgs_s
#avgs = avgs_s

labels = ['1 1 1 2 2 2 2 2 2 2 2 3 3'.split(' '), '1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 3 3 3'.split(' ')]
days = ['1 2 3 1 2 3 4 5 6 7 8 1 1'.split(' '), '1 2 3 4 5 6 7 8 1 2 3 4 5 6 7 8 1 2 3'.split(' ')]
ccc = [features]
#ccc += [['H_{}'.format(labels[0][i]), 'D_{}'.format(days[0][i])] + avgs[i] for i in range(len(avgs_h))]
ccc += [['Healthy', 'Left'.format(days[0][i])] + avgs[i][:len(avgs[i]) // 2] for i in range(len(avgs_h))]
ccc += [['Healthy', 'Right'.format(days[0][i])] + avgs[i][len(avgs[i]) // 2:] for i in range(len(avgs_h))]
#ccc += [['S_{}'.format(labels[1][i - len(avgs_h)]), 'D_{}'.format(days[1][i - len(avgs_h)])] + avgs[i] for i in range(len(avgs_h), len(avgs))]
#ccc += [['Sick', 'Left'.format(days[1][i - len(avgs_h)])] + avgs[i][:len(avgs[i]) // 2] for i in range(len(avgs_h), len(avgs))]
ccc += [['Sick', 'PV1'] + avgs[i][:len(avgs[i]) // 2] for i in range(len(avgs_h), len(avgs))]
ccc += [['Sick', 'PV1 + DI'] + avgs[i][len(avgs[i]) // 2:] for i in range(len(avgs_h), len(avgs))]
#ccc += [['Sick', 'Right'.format(days[1][i - len(avgs_h)])] + avgs[i][len(avgs[i]) // 2:] for i in range(len(avgs_h), len(avgs))]
#
# with open('all_avgs_by_day.csv', 'w') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(ccc)

avgs_by_mouse = [[avgs[i][m * len(avgs[i]) // 2: (m + 1) * len(avgs[i]) // 2] for i in range(len(avgs))] for m in range(2)]


print(avgs_by_mouse)

pca_1 = PCA(n_components=2)

pca_1.fit(avgs_by_mouse[0] + avgs_by_mouse[1])

Xt_1 = pca_1.transform(avgs_by_mouse[0] + avgs_by_mouse[1])
ev_1 = pca_1.explained_variance_ratio_
print(ev_1)

to_csv = []
to_csv.append(['id', 'L_PC1 %d' % int(ev_1[0] * 100), 'L_PC2 %d' % int(ev_1[1] * 100)])

for k in range(len(avgs_by_mouse[0])):
    if 2 < k < 11:
        pass
    if k < 3 + 8 + 2:
        to_csv.append(['healthy_%d' % (k + 1), Xt_1[k, 0], Xt_1[k, 1]])
    else:
        to_csv.append(['PV1_%d' % (k + 1), Xt_1[k, 0], Xt_1[k, 1]])

for k in range(len(avgs_by_mouse[1])):
    if k < 8:
        pass
    if k < 3 + 8 + 2:
        to_csv.append(['healthy_%d' % (k + 1), Xt_1[k + len(avgs_by_mouse[0]), 0], Xt_1[k + len(avgs_by_mouse[0]), 1]])
    else:
        to_csv.append(['(PV1 + DI)_%d' % (k + 1), Xt_1[k + len(avgs_by_mouse[0]), 0], Xt_1[k + len(avgs_by_mouse[0]), 1]])


with open('by_day_pca.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(to_csv)


lda = LatentDirichletAllocation(n_components=3, random_state=0)
lda.fit(avgs_by_mouse[0] + avgs_by_mouse[1])

XT_1 = lda.transform(avgs_by_mouse[0] + avgs_by_mouse[1])


to_csv = []
to_csv.append(['id', 'lda_1', 'lda_2'])

for k in range(len(avgs_by_mouse[0])):
    if 2 < k < 11:
        pass
    if k < 3 + 8 + 2:
        to_csv.append(['healthy_%d' % (k + 1), Xt_1[k, 0], Xt_1[k, 1]])
    else:
        to_csv.append(['PV1_%d' % (k + 1), Xt_1[k, 0], Xt_1[k, 1]])

for k in range(len(avgs_by_mouse[1])):
    if k < 8:
        pass
    if k < 3 + 8 + 2:
        to_csv.append(['healthy_%d' % (k + 1), Xt_1[k + len(avgs_by_mouse[0]), 0], Xt_1[k + len(avgs_by_mouse[0]), 1]])
    else:
        to_csv.append(['(PV1 + DI)_%d' % (k + 1), Xt_1[k + len(avgs_by_mouse[0]), 0], Xt_1[k + len(avgs_by_mouse[0]), 1]])


with open('h_vs_s_all_lda.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(to_csv)

    import os, sys
    import csv
    import xlrd

    import numpy as np
    from numpy.linalg import norm

    from matplotlib import pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


    def metric(plot_1, plot_2):
        return norm(np.array(plot_1) - np.array(plot_2))


    sheet = xlrd.open_workbook('Healthy period_mouse.xlsx')
    sheet = sheet.sheet_by_index(0)

    data = np.array(sheet.col_values(16)[1:])
    # data_r = np.array(sheet.col_values(32))
    # indexes = np.where(data == '')[0].tolist()
    # indexes_r = np.where(data_r == '')[0].tolist()
    # indexes.insert(0, 0)
    # indexes_r.insert(0, 0)
    # print(indexes, indexes_r)
    #
    # l_first_3_days = data[1:407]
    # r_first_3_days = data_r[1:407]
    # r_8_days = data_r[407: 1464]
    # l_1_day = data[1465:1598]
    # r_1_day = data_r[1465:1598]
    # l_2_day = data[1599:]
    # r_2_day = data_r[1599:]
    #
    # days = [l_first_3_days, r_first_3_days, r_8_days, l_1_day, r_1_day, r_2_day]
    #
    # period = 20
    # mean_days = [[] for i in range(len(days))]
    #
    # plt.figure(figsize=(14, 8))
    # for i in range(len(days)):
    #     print(i)
    #     days[i] = days[i][np.where(days[i] != '')]
    #     days[i] = days[i][np.where(days[i] != 'PV1')]
    #
    #     days[i] = days[i].astype(float).tolist()
    #
    #     for j in range(period, len(days[i]) - period):
    #         mean_days[i].append(np.mean(days[i][j - period: j + period]))
    #
    #     plt.plot(mean_days[i])
    #
    #     plt.show()

    day_1 = data[:398]

    day_1 = day_1[np.where(day_1 != '')]
    day_1 = day_1[np.where(day_1 != 'PV1')]

    day_3_to_compare = data[1202 - 50: 1643 + 50]
    day_3_to_compare = day_3_to_compare[np.where(day_3_to_compare != '')]
    day_3_to_compare = day_3_to_compare[np.where(day_3_to_compare != 'PV1')]

    day_3 = data[1202: 1643]
    day_3 = day_3[np.where(day_3 != '')]
    day_3 = day_3[np.where(day_3 != 'PV1')]

    day_1 = day_1.astype(float).tolist()
    day_3 = day_3.astype(float).tolist()
    day_3_to_compare = np.array(day_3_to_compare).astype(float).tolist()

    data = data[np.where(data != '')]
    data = data[np.where(data != 'PV1')]
    data = data.astype(float).tolist()

    # plt.figure()
    # plt.plot([i for i in range(len(data))], data)
    # plt.ylim(top=20)
    # plt.show()

    period = 20
    mean_data = [0 for i in range(period, len(data) - period)]
    for i in range(period, len(data) - period):
        mean_data[i - period] = np.mean(data[i - period: i + period])

    mean_day_1 = [0 for i in range(period, len(day_1) - period)]
    for i in range(period, len(day_1) - period):
        mean_day_1[i - period] = np.mean(day_1[i - period: i + period])

    mean_day_3 = [0 for i in range(period, len(day_3) - period)]
    for i in range(period, len(day_3) - period):
        mean_day_3[i - period] = np.mean(day_3[i - period: i + period])

    plt.figure(figsize=(14, 8))
    # plt.subplot(211)
    # plt.plot([i for i in range(len(mean_day_1))], mean_day_1, [i for i in range(len(mean_day_3))], mean_day_3)
    # #plt.ylim(top=20)

    mean_day_3_to_compare = [0 for i in range(period, len(day_3_to_compare) - period)]
    for i in range(period, len(day_3_to_compare) - period):
        mean_day_3_to_compare[i - period] = np.mean(day_3_to_compare[i - period: i + period])

    dist = metric(mean_day_1, mean_day_3_to_compare[:len(mean_day_1)])
    best = 0
    error = dist
    print(error)

    for i in range(1, len(mean_day_3_to_compare) - len(mean_day_1), 2):
        cur_to_compare = mean_day_3_to_compare[i: i + len(mean_day_1)]
        cur_dist = metric(cur_to_compare, mean_day_1)
        if cur_dist < dist:
            dist = cur_dist
            best = i
            error = cur_dist

    # plt.subplot(211)
    # plt.plot([i for i in range(len(mean_day_1))], mean_day_1, 'C0',[i for i in range(len(mean_day_1))], mean_day_3_to_compare[best: best + len(mean_day_1)], 'C1')

    print(best, error)
    # plt.ylim(top=20)

    p1 = np.array(mean_day_1).reshape(-1, 1)
    print(p1)
    print(p1)
    print(p1.shape)
    p2 = np.array(mean_day_3_to_compare[best: best + len(mean_day_1)]).reshape(len(mean_day_1), 1)
    print(p2.shape)

    # reg = LinearRegression(copy_X=True).fit(p1, p2)
    # print(reg.score(p1, p2))
    # print(reg.coef_, reg.intercept_)
    # a, b = reg.coef_[0][0], reg.intercept_[0]
    # res = np.dot(p1, reg.coef_) + reg.intercept_
    #
    # res = res.reshape(-1, len(mean_day_1))[0].tolist()
    # p2 = p2.reshape(1, -1)[0].tolist()
    # #plt.subplot(212)
    # plt.plot([i for i in range(len(mean_day_1))], mean_day_1, 'C0', [i for i in range(len(mean_day_1))], p2, 'C1', [i for i in range(len(mean_day_1))], res, 'C2')
    # plt.title('y = {0:.2f} * x + {1:.2f}'.format(a, b))
    # #plt.ylim(top=20)
    # if 's' in sys.argv:
    #     plt.show()

    kernel = C(1, (1e-2, 1e2)) * RBF(27, (1e-2, 1e2))
    reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9).fit(p1, p2)
    res, sigma = reg.predict(p1, return_std=True)

    res = res.reshape(-1, len(mean_day_1))[0].tolist()
    p2 = p2.reshape(1, -1)[0].tolist()
    # plt.subplot(212)
    x = [i for i in range(len(mean_day_1))]
    plt.plot(x, mean_day_1, 'C0', x, p2, 'C1', x, res, 'C2')
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([res - 1.9600 * sigma,
                             (res + 1.9600 * sigma)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    # plt.title('y = {0:.2f} * x + {1:.2f}'.format(a, b))
    plt.ylim(top=35, bottom=20)
    if 's' in sys.argv:
        plt.show()
