import random

from flask import abort, jsonify, request
from project.models.prediction import *
from project import app

projects = {
	'age': [],
	'brain_tumor': sorted(['No', 'Yes']),
	'lung_cancer': sorted(['Adenocarcinomas', 'Benign', 'Squamous cell carcinomas']),
	'covid_19': sorted(['COVID', 'Non-COVID']),
	'monkeys': sorted(['Bald Uakari', 'Emperor Tamarin', 'Mandril', 'Proboscis Monkey', 'White face saki']),
	'time_of_days': sorted(['daytime', 'nighttime', 'sunrise']),
	'tyre_quality': sorted(['defective', 'good']),
	'weather': [],
	'vision_transformer': [],
	'skin_disease': sorted(['Chickenpox', 'Measles', 'Monkeypox', 'Normal'])
}

output = {}

@app.route('/', methods=['GET'], endpoint=str(random.getrandbits(128)))
def pred_output():
	p_key = request.args['project']
	classes = projects[p_key]
	file_path = request.args['file']
	match p_key:
		case 'time_of_days':
			pred = TimeOfDay(
				file_path,
				classes,
				p_key
			).tf_results
			output.update(
				{
					'top_result': pred[0],
					'top_acc': pred[1],
					'top_loss': pred[2],
					'class_names': classes,
					'val_acc': {
						'data': pred[3],
						'label': classes
					}
				}
			)
			return jsonify(output)
		case 'brain_tumor':
			pred = BrainTumor(
				file_path,
				classes,
				p_key
			).tf_results
			output.update(
				{
					'top_result': pred[0],
					'top_acc': pred[1],
					'top_loss': pred[2],
					'class_names': classes,
					'val_acc': {
						'data': pred[3],
						'label': classes
					}
				}
			)
			return jsonify(output)
		case 'lung_cancer':
			pred = LungCancer(
				file_path,
				classes,
				p_key
			).tf_results
			output.update(
				{
					'top_result': pred[0],
					'top_acc': pred[1],
					'top_loss': pred[2],
					'class_names': classes,
					'val_acc': {
						'data': pred[3],
						'label': classes
					}
				}
			)
			return jsonify(output)
		case 'covid_19':
			pred = Covid(
				file_path,
				classes,
				p_key
			).tf_results
			output.update(
				{
					'top_result': pred[0],
					'top_acc': pred[1],
					'top_loss': pred[2],
					'class_names': classes,
					'val_acc': {
						'data': pred[3],
						'label': classes
					}
				}
			)
			return jsonify(output)
		case 'monkeys':
			pred = Monkeys(
				file_path,
				classes,
				p_key
			).tf_results
			output.update(
				{
					'top_result': pred[0],
					'top_acc': pred[1],
					'top_loss': pred[2],
					'class_names': classes,
					'val_acc': {
						'data': pred[3],
						'label': classes
					}
				}
			)
			return jsonify(output)
		case 'tyre_quality':
			pred = TyreQuality(
				file_path,
				classes,
				p_key
			).tf_results
			output.update(
				{
					'top_result': pred[0],
					'top_acc': pred[1],
					'top_loss': pred[2],
					'class_names': classes,
					'val_acc': {
						'data': pred[3],
						'label': classes
					}
				}
			)
			return jsonify(output)
		case 'skin_disease':
			pred = SkinDisease(
				file_path,
				classes,
				p_key
			).tf_results
			output.update(
				{
					'top_result': pred[0],
					'top_acc': pred[1],
					'top_loss': pred[2],
					'class_names': classes,
					'val_acc': {
						'data': pred[3],
						'label': classes
					}
				}
			)
			return jsonify(output)
		case _:
			abort(404)
