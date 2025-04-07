import csv
import pathlib

from datetime import datetime

def writer(data, function):
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f'sop_3_cs_v_ecs_{function}_{current_time}.csv'
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
    file_name = f'sop_3_{function}_fmin_values_{current_time}.csv'
    file_path =  pathlib.Path(__file__).cwd() / 'src' / 'simulator' /'results' / file_name

    csa, ecsa = data

    _data = [['CSA', 'ECSA']] + [[csa[i], ecsa[i]] for i in range(len(csa))]

    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(_data)

    return file_path

def std_writer(data):
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f'sop_std_{current_time}.csv'
    file_path =  pathlib.Path(__file__).cwd() / 'src' / 'simulator' /'results' / file_name

    functions, csa_std, csa_var, ecsa_std, ecsa_var = data

    _data = [['function','csa_std','csa_var','ecsa_std', 'ecsa_var']] + [[functions[i], csa_std[i], csa_var[i], ecsa_std[i], ecsa_var[i]] for i in range(len(functions))]

    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(_data)

    return file_path

def solution_writer(algo, content):
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f'{current_time}_{algo} solutions.txt'
    file_path =  pathlib.Path(__file__).cwd() / 'src' / 'simulator' /'results' / file_name

    try:
        with open(file_path, 'w') as file:
            for row in content:
                file.write("\t".join(map(str, row)) + "\n")
        print(f"File created successfully.")
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")