from datetime import datetime as dt
import sys
import os

def log_errors(ex):
	"""
	Ghi nhận lỗi
	:param ex: Thông báo lỗi
	"""
	with open('log_errors.txt', 'a', encoding='utf8') as f:
		e_type, e_object, e_traceback = sys.exc_info()
		e_filename = os.path.split(
			e_traceback.tb_frame.f_code.co_filename
		)[1]
		e_message = str(ex)
		e_line_number = e_traceback.tb_lineno
		f.writelines(f'\n{dt.now()}\t{e_type}\t{e_filename}:{e_line_number}\t{e_message}')
		f.close()
