import os

import numpy as np
import requests
import tensorflow_hub as hub
from keras.models import load_model
from roboflow import Roboflow

from project.models.logs import log_errors
from project.models.tools import image_processing

def get_results_from_result_of_pred(result_of_pred=None, class_names=None):
	"""
	Trả kết quả từ tập result_of_pred (All Default)
	"""
	try:
		top_acc = np.round(100 * np.max(result_of_pred), 2)
		top_loss = np.round(100 - top_acc, 2)
		top_result = class_names[np.argmax(result_of_pred)]
		acc_of_pred = np.round(result_of_pred * 100, 2).tolist()
		return top_result, top_acc, top_loss, acc_of_pred
	except Exception as ex:
		log_errors(ex)

class TFDefault:
	"""
	Default Constructor for Tensorflow
	Args:
		img_save_path: Đường dẫn đầy đủ của tệp ảnh (Bao gồm phần mở rộng)
		class_names: List/Tuple chứa nhãn
		tf_model_name: Tên của mô hình lưu trên ổ cứng (Định dạng .h5)
	"""

	def __init__(
			self,
			img_save_path=None,
			class_names=None,
			tf_model_name=None,
			custom_objects=False,
			img_size=(224, 224)
	):
		self.img_save_loc = f'/imgs/{tf_model_name}'
		if not os.path.exists(self.img_save_loc):
			os.mkdir(self.img_save_loc)
		self.img_save_path = f'{self.img_save_loc}/{img_save_path}'
		self.tf_model_path = f'/models_h5/{tf_model_name}_model.h5'
		self.custom_object = custom_objects
		self.class_names = class_names
		self.img_size = img_size
		self.tf_results = None

		try:
			# Nạp mô hình
			model = load_model(
				self.tf_model_path, custom_objects={'KerasLayer': hub.KerasLayer}
			) if self.custom_object else load_model(self.tf_model_path)
			# Xử lý ảnh đầu vào
			output_img = image_processing(self.img_save_path, self.img_size)
			# Dự đoán giá trị ảnh
			result_of_pred = np.array(model.predict(output_img)[0], dtype=float)
			self.tf_results = get_results_from_result_of_pred(result_of_pred, self.class_names)
		except Exception as ex:
			log_errors(ex)

class RoboflowDefault:
	"""
	Default Constructor for Roboflow
	Args:
		img_save_path: Đường dẫn đầy đủ của tệp ảnh (Bao gồm phần mở rộng)
		rb_project_name: Tên của dự án trên Roboflow
		rb_api_key: Mã API thông báo
		rb_version: Phiên bản của model
	"""

	def __init__(
			self,
			img_save_path=None,
			rb_project_name=None,
			rb_api_key=None,
			rb_version=None
	):
		self.rb_project_name = rb_project_name
		self.rb_api_key = rb_api_key
		self.rb_version = rb_version
		self.model = None
		self.img_save_path = img_save_path
		self.class_names = []
		self.rb_results = None

	def __call__(self, *args, **kwargs):
		try:
			list_of_pred = []
			self.model_download()
			du_doan = self.model.predict(self.img_save_path)
			for a in du_doan[0]['predictions']:
				list_of_pred.append(float(a['confidence']))
				self.class_names.append(str(a['class']).capitalize())
			result_of_pred = np.asarray(list_of_pred, dtype=float)
			self.rb_results = get_results_from_result_of_pred(result_of_pred, self.class_names)
		except Exception as ex:
			log_errors(ex)

	def model_download(self):
		rf = Roboflow(api_key=self.rb_api_key)
		project = rf.workspace().project(self.rb_project_name)
		model = project.version(self.rb_version).model
		self.model = model

# class LicensePlate(TFDefault):
# 	def __init__(
# 			self,
# 			full_path=None,
# 			img_save_name=None,
# 			img_save_path=None,
# 			class_names=None,
# 			tf_model_name=None,
# 			ul_model_name=None
# 	):
# 		super().__init__(img_save_path, class_names, tf_model_name)
# 		self.tf_results = None
# 		self.full_path = full_path
# 		self.img_save_name = img_save_name
# 		self.ul_model_name = ul_model_name
# 		self.img_size = (128, 128)
#
# 	def __call__(self, *args, **kwargs):
# 		# Thiết lập chung
# 		characters = []
# 		image = cv.imread(self.img_save_path)
# 		gray_image = cv.cvtColor(image, cv.IMREAD_GRAYSCALE)
# 		rgb_image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
#
# 		# Tìm biển số
# 		model = YOLO(self.ul_model_name)
# 		results = model.predict(self.img_save_path)
#
# 		for result in results:
# 			boxes = result.boxes.numpy()
# 			for box in boxes:
# 				(x, y, w, h) = box.xyxy[0].astype(int)
# 				# Cắt & chuyển ảnh xám & RGB
# 				cropped_gray_image = gray_image[y: h, x: w]
# 				# Chuyển đổi Gauss & phát hiện cạnh Canny
# 				cropped_gauss_image = cv.GaussianBlur(cropped_gray_image, (5, 5), cv.BORDER_DEFAULT)
# 				cropped_edges_image = cv.Canny(cropped_gauss_image, 80, 100)
# 				# Tìm & sắp xếp Contour kí tự
# 				contours, _ = cv.findContours(cropped_edges_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# 				boxes_characters = list([cv.boundingRect(c) for c in contours]).sort(
# 					key=functools.cmp_to_key(compare)
# 				)
# 				# Lọc bbox & thêm vào mảng để pred
# 				for i in boxes_characters:
# 					(x, y, w, h) = i
# 					# Chỉ lấy các box có chiều rộng 20 - 50 & chiều cao 90 - 100
# 					if 20 <= w <= 50 and 90 <= h <= 100:
# 						# Tách và lưu trữ ký tự đã phân đoạn
# 						character = gray_image[y:y + h + 4, x:x + w + 4]
# 						# Tiền xử lý ảnh trước khi predict
# 						character = cv.cvtColor(character, cv.COLOR_GRAY2RGB)
# 						character = cv.resize(character, self.img_size)
# 						character = keras.utils.img_to_array(character)
# 						characters.append(character)
# 				characters = np.asarray(characters, dtype=float) / 255
# 				# Dự đoán ký tự
# 				model = keras.models.load_model(self.tf_model_name)
# 				du_doan = model.predict(characters)
# 				bien_so_xe = ''
# 				for i in range(len(du_doan)):
# 					bien_so_xe += self.class_names[np.argmax(du_doan[i])]
# 				# Vẽ viền quanh biển số & đặt Text
# 				cv.rectangle(rgb_image, (x, y), (w, h), (0, 255, 0), 3)
# 				cv.putText(
# 					rgb_image,
# 					bien_so_xe,
# 					(x, y - 6),
# 					cv.FONT_HERSHEY_COMPLEX,
# 					0.4,
# 					(255, 255, 0),
# 					1
# 				)
# 		file_path = os.path.join(self.full_path, self.img_save_name)
# 		cv.imwrite(file_path, rgb_image)
# 		self.tf_results = file_path

class HuggingPageDefault:
	def __init__(
			self,
			img_save_path=None,
			hg_api_token=None,
			hg_model_url=None
	):
		self.img_save_path = img_save_path
		self.hg_api_token = hg_api_token
		self.hg_model_url = hg_model_url
		self.class_names = []
		self.hg_results = None

	def __call__(self, *args, **kwargs):
		try:
			list_of_pred = []
			hg_url = f'https://api-inference.huggingface.co/models/{self.hg_model_url}'
			headers = {"Authorization": f"Bearer {self.hg_api_token}"}
			with open(self.img_save_path, "rb") as f:
				data = f.read()
				du_doan = requests.post(hg_url, headers=headers, data=data).json()
			for a in du_doan:
				list_of_pred.append(float(a['score']))
				self.class_names.append(str(a['label']).capitalize())
			result_of_pred = np.asarray(list_of_pred, dtype=float)
			self.hg_results = get_results_from_result_of_pred(result_of_pred, self.class_names)
		except Exception as ex:
			log_errors(ex)

class TimeOfDay(TFDefault):
	"""
	Mô hình dự đoán thời gian trong ngày
	Args:
		img_save_path: Đường dẫn đầy đủ của tệp ảnh (Bao gồm phần mở rộng)
		class_names: List/Tuple chứa nhãn
		tf_model_name: Tên của mô hình lưu trên ổ cứng (Định dạng .h5)
	Returns:
		List/Tuple chứa kết quả, độ chính xác, sai số của giá trị dự đoán
	"""

	def __init__(
			self,
			img_save_path=None,
			class_names=None,
			tf_model_name=None
	):
		super().__init__(img_save_path, class_names, tf_model_name, custom_objects=True)

class BrainTumor(TFDefault):
	"""
	Mô hình dự đoán bệnh u não qua ảnh chụp cộng hưởng từ (MRI)
	Args:
		img_save_path: Đường dẫn đầy đủ của tệp ảnh (Bao gồm phần mở rộng)
		class_names: List/Tuple chứa nhãn
		tf_model_name: Tên của mô hình lưu trên ổ cứng (Định dạng .h5)
	Returns:
		List/Tuple chứa kết quả, độ chính xác, sai số của giá trị dự đoán
	"""

	def __init__(
			self,
			img_save_path=None,
			class_names=None,
			tf_model_name=None
	):
		super().__init__(img_save_path, class_names, tf_model_name)

class Covid(TFDefault):
	"""
	Mô hình dự đoán bệnh viêm phổi qua ảnh chụp X-quang
	Args:
		img_save_path: Đường dẫn đầy đủ của tệp ảnh (Bao gồm phần mở rộng)
		class_names: List/Tuple chứa nhãn
		tf_model_name: Tên của mô hình lưu trên ổ cứng (Định dạng .h5)
	Returns:
		List/Tuple chứa kết quả, độ chính xác, sai số của giá trị dự đoán
	"""

	def __init__(
			self,
			img_save_path=None,
			class_names=None,
			tf_model_name=None
	):
		super().__init__(img_save_path, class_names, tf_model_name, img_size=(128, 128))

class SkinDisease(TFDefault):
	"""
	Mô hình dự đoán bệnh về da
	Args:
		img_save_path: Đường dẫn đầy đủ của tệp ảnh (Bao gồm phần mở rộng)
		class_names: List/Tuple chứa nhãn
		tf_model_name: Tên của mô hình lưu trên ổ cứng (Định dạng .h5)
	Returns:
		List/Tuple chứa kết quả, độ chính xác, sai số của giá trị dự đoán
	"""

	def __init__(
			self,
			img_save_path=None,
			class_names=None,
			tf_model_name=None
	):
		super().__init__(img_save_path, class_names, tf_model_name, img_size=(128, 128))

class LungCancer(TFDefault):
	"""
	Mô hình dự đoán bệnh ung thư phổi qua ảnh chụp mô tế bào
	Args:
		img_save_path: Đường dẫn đầy đủ của tệp ảnh (Bao gồm phần mở rộng)
		class_names: List/Tuple chứa nhãn
		tf_model_name: Tên của mô hình lưu trên ổ cứng (Định dạng .h5)
	Returns:
		List/Tuple chứa kết quả, độ chính xác, sai số của giá trị dự đoán
	"""

	def __init__(
			self,
			img_save_path=None,
			class_names=None,
			tf_model_name=None
	):
		super().__init__(img_save_path, class_names, tf_model_name)

class Monkeys(TFDefault):
	"""
	Mô hình dự đoán khỉ
	Args:
		img_save_path: Đường dẫn đầy đủ của tệp ảnh (Bao gồm phần mở rộng)
		class_names: List/Tuple chứa nhãn
		tf_model_name: Tên của mô hình lưu trên ổ cứng (Định dạng .h5)
	Returns:
		List/Tuple chứa kết quả, độ chính xác, sai số của giá trị dự đoán
	"""

	def __init__(
			self,
			img_save_path=None,
			class_names=None,
			tf_model_name=None
	):
		super().__init__(img_save_path, class_names, tf_model_name, custom_objects=True, img_size=(160, 160))

class TyreQuality(TFDefault):
	"""
	Mô hình dự đoán chất lượng lốp xe
	Args:
		img_save_path: Đường dẫn đầy đủ của tệp ảnh (Bao gồm phần mở rộng)
		class_names: List/Tuple chứa nhãn
		tf_model_name: Tên của mô hình lưu trên ổ cứng (Định dạng .h5)
	Returns:
		List/Tuple chứa kết quả, độ chính xác, sai số của giá trị dự đoán
	"""

	def __init__(
			self,
			img_save_path=None,
			class_names=None,
			tf_model_name=None
	):
		super().__init__(img_save_path, class_names, tf_model_name, img_size=(128, 128))

# class MaskDetection(TFDefault):
# 	"""
# 	Mô hình dự đoán khỉ
# 	Args:
# 		img_save_path: Đường dẫn đầy đủ của tệp ảnh (Bao gồm phần mở rộng)
# 		class_names: List/Tuple chứa nhãn
# 		tf_model_name: Tên của mô hình lưu trên ổ cứng (Định dạng .h5)
# 	Returns:
# 		List/Tuple chứa kết quả, độ chính xác, sai số của giá trị dự đoán
# 	"""
#
# 	def __init__(
# 			self,
# 			img_save_path=None,
# 			class_names=None,
# 			tf_model_name=None
# 	):
# 		super().__init__(img_save_path, class_names, tf_model_name)
# 		self.img_size = (160, 160)
# 		self.tf_results = None
#
# 	def __call__(self):
# 		super().__call__()
