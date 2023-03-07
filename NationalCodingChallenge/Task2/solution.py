from flask import Flask, jsonify, request

app = Flask(__name__)
users = [
    {
        "id": 1,
        "name": "John",
        "role": "admin"
    },
    {
        "id": 2,
        "name": "Juan",
        "role": "developer"
    }
]
@app.route('/users', methods = ["GET"])
def index():
    name = request.args.get('name')
    if name is not None:
        returned_users = [user for user in users if user["name"] == name ]
        return jsonify(returned_users), 200
    else:
        return jsonify(users), 200

if __name__ == '__main__':
    app.run()