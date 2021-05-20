from flask import Flask, request
from predict import predict as func
import json
from flask_cors import cross_origin

app = Flask(__name__)


@app.route('/predict', methods=['POST', 'GET'])
@cross_origin(supports_credentials=True)
def predict():
    res = {}
    question = str(request.args.get("question"))
    print(question)
    status, answer, sub_graph = func(question)
    if not status:
        res['status'] = '500'
        res['data'] = 'illegal input'
    else:
        res['status'] = '200 OK'
        res['data'] = {'answer': answer, 'sub_graph': sub_graph}
    return json.dumps(res, ensure_ascii=False)


def main():
    app.run(debug=True, threaded=True, host='0.0.0.0', port=2333)


if __name__ == '__main__':
    main()
