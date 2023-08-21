import requests
from camera import mainCam
urlPost = "http://localhost:5000/move/"
def main():
    if requests.get(urlPost+"/500").status_code == 200:        #Checks that there is a connection to the server before continuing
        post=requests.post(urlPost+"/0")
        print(post.status_code)
    else:
        print("Something is wrong")
main()