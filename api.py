import pandas as pd
import io
import time

from flask import Flask, request, jsonify
from werkzeug.datastructures import ImmutableMultiDict
from flask_cors import CORS, cross_origin

from functions.distance import Distance
from src.ecs.decs import discrete_enhanced_cuckoo_search_algorithm

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
        
        # Run simulation
        d = Distance(db_coordinates=db_coordinates, es_coordinates=es_coordinates)
        app.logger.info(f"Distance matrix generated.")

        start_time = time.time()

        app.logger.info("Running simulation...")
        nests, f_values, fmin, best, best_fev = discrete_enhanced_cuckoo_search_algorithm(size=50, dimensions=(len(db_coordinates), len(es_coordinates)), iterations=int(iterations), discovery_rate=(0.5, 0.25), alpha_value=(0.01, 0.05), target_function=d.fitness)

        app.logger.info("Simulation completed. Decoding Results...")

        decoded_result, allocations = d.decode_results(best)

        app.logger.info("Decoding Successful.")

        end_time = time.time()
        execution_time = end_time - start_time

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
            "result": decoded_result,
            "metrics": {
                "iterations": iterations,
                "convergence": fmin,
                "allocations": allocations,
                "executionTime": execution_time,
                "resultFitness": best_fev.tolist()
            }
        })

if __name__ == '__main__':
    app.run(debug=True)