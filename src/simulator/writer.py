import csv
import pathlib

from datetime import datetime

def writer(data, function):
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f'cs_v_ecs_{function}_{current_time}.csv'
    file_path =  pathlib.Path(__file__).cwd() / 'src' / 'simulator' /'results' / file_name

    csa, ecsa = data

    _data = [
        ['Algorithm','Mean', 'Standard Deviation', 'p-value'],
        csa,
        ecsa,
    ]

    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(_data)

    return file_path

def fmin_writer(data, function):
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f'{function}_fmin_values_{current_time}.csv'
    file_path =  pathlib.Path(__file__).cwd() / 'src' / 'simulator' /'results' / file_name

    csa, ecsa = data

    _data = [['CSA', 'ECSA']] + [[csa[i], ecsa[i]] for i in range(len(csa))]

    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(_data)

    return file_path