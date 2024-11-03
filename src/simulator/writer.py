import csv
import pathlib

from datetime import datetime

def writer(data):
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f'cs_v_ecs_{current_time}.csv'
    file_path =  pathlib.Path(__file__).cwd() / 'src' / 'simulator' /'results' / file_name

    cs, ecs = data

    _data = [
        ['Algorithm', 'Minimum', 'Maximum', 'Mean', 'Standard Deviation'],
        cs,
        ecs,
    ]

    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(_data)

    return file_path

