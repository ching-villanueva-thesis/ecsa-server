import pandas as pd
import io
import time
import base64

from flask import Flask, request, jsonify
from werkzeug.datastructures import ImmutableMultiDict
from flask_cors import CORS, cross_origin

from functions.distance import Distance
from src.cs.dcs import discrete_cuckoo_search_algorithm, initialize_nests as dcs_initialize_nests
from src.ecs.decs import discrete_enhanced_cuckoo_search_algorithm, initialize_nests as decs_initialize_nests

from src.util.scatter import plot_scatter
from src.util.cos_annealing import plot_cosine_annealing, plot_fixed_parameters
from src.util.hybrid_walk import plt_hybrid_walk, plt_levy_walk
from src.util.convergence_curve import plt_convergence

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

"""
    Request format:
    {
        multi-part form data
    }
"""
@app.route(rule='/api/simulate', methods=['POST'])
@cross_origin()
def handle_request():

    if request.method == 'POST':
        if 'demandFile' not in request.files or 'spacesFile' not in request.files:
            return jsonify({'error': 'Missing files'}), 400

        iterations = request.form.get('maxIterations', False)
        demand_file = request.files['demandFile']
        spaces_file = request.files['spacesFile']

        demand_df = pd.read_csv(io.StringIO(demand_file.stream.read().decode("utf-8")))
        spaces_df = pd.read_csv(io.StringIO(spaces_file.stream.read().decode("utf-8")))

        db_coordinates = list(zip(demand_df['Longitude'], demand_df['Latitude']))
        es_coordinates = list(zip(spaces_df['Longitude'], spaces_df['Latitude']))

        app.logger.info(f"Client request to simulate {len(db_coordinates)} district blocks and {len(es_coordinates)} evacuation spaces.")

        d = Distance(db_coordinates=db_coordinates, es_coordinates=es_coordinates)
        app.logger.info(f"Distance matrix generated.")
        
        # Run simulation
        start_time = time.time()

        runs = 5
        _dcs_best = []
        _decsa_best = []

        app.logger.info(f"Simulating {runs} runs to get the best solution.")
        for i in range(runs):
            app.logger.info(f"Run {i + 1} of {runs}")

            # Run Discrete Cuckoo Search Algorithm
            _, _, dcs_fmin, dcs_best, dcs_best_fev = discrete_cuckoo_search_algorithm(
                size=50, 
                dimensions=(len(db_coordinates), len(es_coordinates)), 
                iterations=int(iterations), 
                discovery_rate=0.25, 
                alpha_value=0.01, 
                target_function=d.fitness
            )

            _dcs_best.append((dcs_best, dcs_best_fev, dcs_fmin))
            
            # Run Discrete Enhanced Cuckoo Search Algorithm
            _, _, decsa_fmin, decsa_best, decsa_best_fev = discrete_enhanced_cuckoo_search_algorithm(
                size=50, 
                dimensions=(len(db_coordinates), len(es_coordinates)), 
                iterations=int(iterations), 
                discovery_rate=(0.5, 0.25), 
                alpha_value=(0.01, 0.05), 
                target_function=d.fitness
            )

            _decsa_best.append((decsa_best, decsa_best_fev, decsa_fmin))

        # Find the best solution from each algorithm
        app.logger.info("Simulation completed. Decoding Results...")

        dcs_best, dcs_best_fev, dcs_fmin = min(_dcs_best, key=lambda x: x[1])
        decsa_best, decsa_best_fev, decsa_fmin = min(_decsa_best, key=lambda x: x[1])

        dcs_decoded_result, dcs_allocations = d.decode_results(dcs_best)
        decsa_decoded_result, decsa_allocations = d.decode_results(decsa_best)

        app.logger.info(f"DCS: {len(dcs_decoded_result)} Blocks Allocated")
        app.logger.info(f"DECSA: {len(decsa_decoded_result)} Blocks Allocated")

        app.logger.info("Decoding Successful.")

        end_time = time.time()
        execution_time = end_time - start_time

        convergence_curve = plt_convergence(dcs_fmin, decsa_fmin, "Optimal Allocation of Evacuation Spaces")
        convergence_curve_b64 = base64.b64encode(convergence_curve.getvalue()).decode('utf-8')

        """
            Response Format
            {
                result: list,
                metrics: {
                    iterations: int,
                    convergence: list,
                    resultFitness: int
                }
            }
        """
        return jsonify({
            "dcsResult": dcs_decoded_result,
            "decsaResult": decsa_decoded_result,
            "iterations": iterations,
            "metrics": {
                "dcs" : {
                    "allocations": dcs_allocations,
                    "executionTime": execution_time,
                    "resultFitness": dcs_best_fev
                },
                "decsa" : {
                    "allocations": decsa_allocations,
                    "executionTime": execution_time,
                    "resultFitness": decsa_best_fev
                }
            },
            "convergenceCurve": f"data:image/png;base64, {convergence_curve_b64}",
        })

@app.route(rule='/api/simulate/sop1', methods=['POST'])
@cross_origin()
def handle_simulate_sop1():
    
    if request.method == 'POST':
        if 'demandFile' not in request.files or 'spacesFile' not in request.files:
            return jsonify({'error': 'Missing files'}), 400

        size = request.form.get('populationSize', False)
        demand_file = request.files['demandFile']
        spaces_file = request.files['spacesFile']

        demand_df = pd.read_csv(io.StringIO(demand_file.stream.read().decode("utf-8")))
        spaces_df = pd.read_csv(io.StringIO(spaces_file.stream.read().decode("utf-8")))

        db_coordinates = list(zip(demand_df['Longitude'], demand_df['Latitude']))
        es_coordinates = list(zip(spaces_df['Longitude'], spaces_df['Latitude']))

        app.logger.info(f"Visualizing SOP 1")

        d = Distance(db_coordinates=db_coordinates, es_coordinates=es_coordinates)
        app.logger.info(f"Distance matrix generated.")

        dcs_initial, _ = dcs_initialize_nests(size=int(size), dimensions=(len(db_coordinates), len(es_coordinates)), target_function=d.fitness)

        dcs_initial_bytes = plot_scatter(dcs_initial, title="Cuckoo Search Initial Population Scatter Plot", method="Random Initialization")
        dcs_initial_b64 = base64.b64encode(dcs_initial_bytes.getvalue()).decode('utf-8')

        decsa_initial, _ = decs_initialize_nests(size=int(size), dimensions=(len(db_coordinates), len(es_coordinates)), target_function=d.fitness)

        decsa_initial_bytes = plot_scatter(decsa_initial, title="Enhanced Cuckoo Search Initial Population Scatter Plot", method="Sobol Sequence")
        decsa_initial_b64 = base64.b64encode(decsa_initial_bytes.getvalue()).decode('utf-8')

        return jsonify({
            "dcs" : f"data:image/png;base64,{dcs_initial_b64}",
            "decsa" : f"data:image/png;base64,{decsa_initial_b64}",
        })

@app.route(rule='/api/simulate/sop2', methods=['POST'])
@cross_origin()
def handle_simulate_sop2():
    if request.method == 'POST':
        steps = request.form.get('steps', False)

        app.logger.info(f"Visualizing SOP 2")

        levy_walk = plt_levy_walk(n_steps=int(steps))
        levy_walk_b64 = base64.b64encode(levy_walk.getvalue()).decode('utf-8')

        hybrid_stochastic_walk = plt_hybrid_walk(n_steps=int(steps))
        hybrid_stochastic_walk_b64 = base64.b64encode(hybrid_stochastic_walk.getvalue()).decode('utf-8')

        return jsonify({
            "dcs" : f"data:image/png;base64, {levy_walk_b64}",
            "decsa" : f"data:image/png;base64, {hybrid_stochastic_walk_b64}"
        })

@app.route(rule='/api/simulate/sop3', methods=['POST'])
@cross_origin()
def handle_simulate_sop3():

    if request.method == 'POST':
        d_min = request.form.get('dMin', False)
        d_max = request.form.get('dMax', False)
        
        a_min = request.form.get('aMin', False)
        a_max = request.form.get('aMax', False)

        iterations = request.form.get('maxIterations', False)
        cycles = request.form.get('cycles', 4)

        app.logger.info(f"Visualizing SOP 3")
        
        plt_fixed_dr = plot_fixed_parameters(0.25, int(iterations), "Discovery Rate")
        plt_fixed_drb64 = base64.b64encode(plt_fixed_dr.getvalue()).decode('utf-8')

        plt_fixed_av = plot_fixed_parameters(0.01, int(iterations), "Alpha Value")
        plt_fixed_avb64 = base64.b64encode(plt_fixed_av.getvalue()).decode('utf-8')

        plt_cos_annealing_dr = plot_cosine_annealing(float(d_min),float(d_max), int(iterations), "Discovery Rate", int(cycles))
        plt_cos_annealing_drb64 = base64.b64encode(plt_cos_annealing_dr.getvalue()).decode('utf-8')

        plt_cos_annealing_av = plot_cosine_annealing(float(a_max),float(a_min), int(iterations), "Alpha Value", int(cycles))
        plt_cos_annealing_avb64 = base64.b64encode(plt_cos_annealing_av.getvalue()).decode('utf-8')

        return jsonify({
            "dcs": [f"data:image/png;base64, {plt_fixed_drb64}", f"data:image/png;base64, {plt_fixed_avb64}"],
            "decsa": [f"data:image/png;base64, {plt_cos_annealing_drb64}", f"data:image/png;base64, {plt_cos_annealing_avb64}"]
        })

if __name__ == '__main__':
    app.run(debug=True)