import os

from flask import Flask, request, jsonify

from check_camera import check_mail

app = Flask(__name__)


@app.route('/check_camera', methods=['POST'])
def check_cam():
    dirname = request.json.get('dirname', './')
    subdirname = request.json.get('subdirname', './')
    check_mail(dirname, subdirname)

    return jsonify({
            "response": [{
                "message": "success"
            }]
        })


if __name__ == '__main__':
    app.run(port='5000')
