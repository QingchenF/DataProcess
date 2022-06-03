

from PIL import Image
import numpy as np

# # 原图
# img1 = cv2.imread('./Figure_1.png')
# img2 = cv2.imread('./Figure_2.png')
# # 灰色图
# # gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# # gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#
# # ====使用numpy的数组矩阵合并concatenate======
# # 纵向连接
# image = np.vstack((img1, img2))
# # 横向连接
# #image = np.concatenate([img1, img2], axis=1)
#
# # =============
# cv2.imwrite('./new_image.jpg', image)
