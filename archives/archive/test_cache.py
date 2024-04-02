from alpha_zero_bayes.mcts import tuple_to_posteriror

if __name__ == "__main__":
    tupled = [((1,2,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0))]
    tupled.append(((1,2,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0)))
    tupled.append(((1,2,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0)))
    tupled.append(((1,2,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0)))
    tupled.append(((1,2,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0),(0,0,0,0)))


    for tuples in tupled:
        print(tuple_to_posteriror(0.25, tuples))
        