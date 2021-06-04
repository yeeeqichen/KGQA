from flask import Flask, request
from predict import predict as func
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    res = {}
    question = str(request.args.get("question"))
    print(question)
    status, answer_list = func(question)
    if not status:
        res['status'] = '500'
        res['data'] = 'illegal input'
    else:
        [answer, sub_graph, head, answer_subgraph] = answer_list
        res['status'] = '200 OK'
        res['data'] = {
            'answer': answer,
            'sub_graph': sub_graph,
            'head': head,
            'answer_subgraph': answer_subgraph
            }
    return json.dumps(res, ensure_ascii=False)


def main():
    app.run(debug=True, threaded=True, host='0.0.0.0', port=2333)


if __name__ == '__main__':
    main()
