from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route(rule='/api/simulate', methods=['POST'])
def handle_request():

    if request.method == 'POST':
        # Get payload
        payload = request.get_json()

        # Run simulation

        return jsonify({
            'message' : 'Placeholder; Return simulation results',
        })

if __name__ == '__main__':
    app.run(debug=True)