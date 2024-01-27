import numpy as np
from keras.utils import img_to_array
from PIL import Image as pil

from project.models.logs import log_errors

# def img_to_base64(full_path_of_file):
# 	"""
# 	Chuyển đổi ảnh sang định dạng Base64
# 	:param full_path_of_file: Đường dẫn đến tệp ảnh
# 	:return chuỗi Base64
# 	"""
# 	try:
# 		with pil.open(full_path_of_file) as img:
# 			bs64_str = b64encode(img).decode(encoding='utf-8')
# 			bs64_str = f'data:image/jpeg;base64,{bs64_str}'
# 			return bs64_str
# 	except Exception as ex:
# 		log_errors(ex)

def image_processing(file_path, image_size=(224, 224)):
	"""
	Xử lý hình ảnh đầu vào
	:param file_path: Đường dẫn đầy đủ của tệp ảnh (Bao gồm phần mở rộng)
	:param image_size: Tuple/List/NdArray quy định kích thước ảnh (Mặc định: 224 x 224)
	:return: NdArray tập ảnh đã được xử lý
	"""
	try:
		# Tiền xử lý ảnh
		with pil.open(file_path) as image:
			image = image.resize(image_size)
			image = image.convert('RGB')
			image = img_to_array(image) / 255
			image = np.expand_dims(image, axis=0)
			return image
	except Exception as ex:
		log_errors(ex)

# def compare(rect1, rect2):
# 	"""
# 	Sắp xếp thứ tự bbox
# 	"""
# 	if abs(rect1[1] - rect2[1]) > 10:
# 		return rect1[1] - rect2[1]
# 	else:
# 		return rect1[0] - rect2[0]
