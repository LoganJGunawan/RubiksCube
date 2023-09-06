from flask import Flask, request
from flask_restful import Api, Resource

app = Flask("RubiksCubeAPI")
api = Api(app)

class Cube(Resource):
    def get(self):      #Test function to ensure connection works
        print("Hello World")
        return 200
    
    def post(self):
        print("Hereasd")
        data = request.data.decode('utf-8')
        print(data)
        #numbers = [int(num) for num in data.split(',')]
        return 201
        #return {"message":"Received Numbers: {}".format(numbers)}, 201

api.add_resource(Cube,'/move','/post')

print("Running")
app.run(host='0.0.0.0', port=5000,debug=True)