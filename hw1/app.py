from flask import Flask, request, jsonify

from stud.implementation import build_model

app = Flask(__name__)
model = build_model('cpu')


@app.route("/", defaults={"path": ""}, methods=["POST", "GET"])
@app.route("/<path:path>", methods=["POST", "GET"])
def annotate(path):
    try:
        json_body = request.json
        tokens_s = json_body['tokens_s']
        predictions_s = model.predict(tokens_s)
    except Exception as e:
        return {'error': 'Bad request', 'message': str(e)}, 400
    return jsonify(tokens_s=tokens_s, predictions_s=predictions_s)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12345)
