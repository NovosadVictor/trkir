import os

from flask import Flask, request, jsonify

from upload import upload

app = Flask(__name__)


@app.route('/upload', methods=['POST'])
async def upload_dir():
    dirname = request.json.get('dirname', './')
    subdirname = request.json.get('subdirname', './')
    return await upload(dirname, subdirname)
    # return jsonify({
    #         "response": [{
    #             "message": "success"
    #         }]
    #     })


if __name__ == '__main__':
    app.run(port='5001')
