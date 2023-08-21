from flask import Flask
from flask_restful import Api, Resource, reqparse, abort
from test2 import testFunc

app = Flask("RubiksCubeAPI")
api = Api(app)
parser=reqparse.RequestParser()
parser.add_argument('moveID', type=str)

move={}     #Empty dictionary to 

class Cube(Resource):
    def get(self,moveID):      #Test function to ensure connection works
        return 200

    def getStatusMoves(self):    #Gets final move count 
        return len(move)

    def post(self,moveID):        #Relays a move for the RasbPi to tell the Arduino
        move[f"move{len(move)+1}"]=moveID
        testFunc()
        return move, 201

api.add_resource(Cube,'/move/<moveID>')

print("Running")
app.run()