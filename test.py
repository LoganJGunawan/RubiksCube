with open("test.txt", mode="a") as f:
    for a in range(6):
        for b in range(6):
            f.write(str(a)+str(b)+": \n")