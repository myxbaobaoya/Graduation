# import random
# import numpy as np
# import math
# np.set_printoptions(suppress=True)
# np.set_printoptions(precision=3)
# for i in range(200):
#     a = random.randint(1,10)
#     b = random.randint(1,10)*0.1
#     c = random.randint(1,25)*0.1
#     d = random.randint(1,10)
#     result = 2*3.14+3.14*a+2*b*b+2*math.exp(c) + 1/d
#     if i == 0:
#         t = np.array([a, b, c, d])
#         train_data = np.array([a, b, c,d])
#         r = np.array([result])
#         result_data = np.array([result])
#     else:
#         t = np.array([a, b, c, d])
#         train_data = np.vstack((train_data, t))
#         r = np.array([result])
#         result_data = np.vstack((result_data, r))
# np.savetxt("train_data.txt", train_data,fmt='%.03f')
# np.savetxt("result_data.txt", result_data,fmt='%.03f')