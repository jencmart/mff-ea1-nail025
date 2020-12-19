# Python3 program to solve
# Traveling Salesman Problem using
# Branch and Bound.
import math
import numpy as np
maxsize = float('inf')


# Function to copy temporary solution
# to the final solution
def copyToFinal(curr_path):
    final_path[:N + 1] = curr_path[:]
    final_path[N] = curr_path[0]


# Function to find the minimum edge cost
# having an end at the vertex i
def firstMin(adj, i):
    min = maxsize
    for k in range(N):
        if adj[i][k] < min and i != k:
            min = adj[i][k]

    return min


# function to find the second minimum edge
# cost having an end at the vertex i
def secondMin(adj, i):
    first, second = maxsize, maxsize
    for j in range(N):
        if i == j:
            continue
        if adj[i][j] <= first:
            second = first
            first = adj[i][j]

        elif (adj[i][j] <= second and
              adj[i][j] != first):
            second = adj[i][j]

    return second


# function that takes as arguments:
# curr_bound -> lower bound of the root node
# curr_weight-> stores the weight of the path so far
# level-> current level while moving
# in the search space tree
# curr_path[] -> where the solution is being stored
# which would later be copied to final_path[]
def TSPRec(adj, curr_bound, curr_weight,
           level, curr_path, visited):
    global final_res

    # base case is when we have reached level N
    # which means we have covered all the nodes once
    if level == N:
        print(level)
        # check if there is an edge from
        # last vertex in path back to the first vertex
        if adj[curr_path[level - 1]][curr_path[0]] != 0:

            # curr_res has the total weight
            # of the solution we got
            curr_res = curr_weight + adj[curr_path[level - 1]][curr_path[0]]
            print(curr_res)
            if curr_res < final_res:
                copyToFinal(curr_path)
                final_res = curr_res
        return

    # for any other level iterate for all vertices
    # to build the search space tree recursively
    for i in range(N):

        # Consider next vertex if it is not same
        # (diagonal entry in adjacency matrix and
        #  not visited already)
        if (adj[curr_path[level - 1]][i] != 0 and
                visited[i] == False):
            temp = curr_bound
            curr_weight += adj[curr_path[level - 1]][i]

            # different computation of curr_bound
            # for level 2 from the other levels
            if level == 1:
                curr_bound -= ((firstMin(adj, curr_path[level - 1]) +
                                firstMin(adj, i)) / 2)
            else:
                curr_bound -= ((secondMin(adj, curr_path[level - 1]) +
                                firstMin(adj, i)) / 2)

                # curr_bound + curr_weight is the actual lower bound
            # for the node that we have arrived on.
            # If current lower bound < final_res,
            # we need to explore the node further
            if curr_bound + curr_weight < final_res:
                curr_path[level] = i
                visited[i] = True

                # call TSPRec for the next level
                TSPRec(adj, curr_bound, curr_weight,
                       level + 1, curr_path, visited)

                # Else we have to prune the node by resetting
            # all changes to curr_weight and curr_bound
            curr_weight -= adj[curr_path[level - 1]][i]
            curr_bound = temp

            # Also reset the visited array
            visited = [False] * len(visited)
            for j in range(level):
                if curr_path[j] != -1:
                    visited[curr_path[j]] = True


# This function sets up final_path
def TSP(adj):
    # Calculate initial lower bound for the root node
    # using the formula 1/2 * (sum of first min +
    # second min) for all edges. Also initialize the
    # curr_path and visited array
    curr_bound = 0
    curr_path = [-1] * (N + 1)
    visited = [False] * N

    # Compute initial bound
    curr_bound = 160000


    # We start at vertex 1 so the first vertex
    # in curr_path[] is 0
    visited[0] = True
    curr_path[0] = 0

    # Call to TSPRec for curr_weight
    # equal to 0 and level 1
    TSPRec(adj, curr_bound, 0, 1, curr_path, visited)


# reads the input set of values of objects
def read_locations(filename):
    locations = []
    with open(filename) as f:
        for l in f.readlines():
            tokens = l.split(' ')
            locations.append((float(tokens[0]), float(tokens[1])))
    return locations

# Driver code
def distance(loc1, loc2):
    # based on https://stackoverflow.com/questions/15736995/how-can-i-quickly-estimate-the-distance-between-two-latitude-longitude-points
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [loc1[1], loc1[0], loc2[1], loc2[0]])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    # Radius of earth in kilometers is 6371
    km = 6371.01 * c
    return km


def distances_from():
    INPUT = 'inputs/tsp_std.in'  # the input file
    cities = read_locations(INPUT)
    city_cache = np.zeros([len(cities), len(cities)])
    for i, i_city in enumerate(cities):
        for j, j_city in enumerate(cities):
            if i != j:
                city_cache[i, j] = distance(i_city, j_city)
            else:
                city_cache[i, j] = 0
    return city_cache


if __name__ == '__main__':

    # Adjacency matrix for the given graph
    adj = distances_from()
    N = adj.shape[0]


    # final_path[] stores the final solution
    # i.e. the // path of the salesman.
    final_path = [None] * (N + 1)

    # visited[] keeps track of the already
    # visited nodes in a particular path
    visited = [False] * N

    # Stores the final minimum weight
    # of shortest tour.
    final_res = maxsize

    TSP(adj)

    print("Minimum cost :", final_res)
    print("Path Taken : ", end=' ')
    for i in range(N + 1):
        print(final_path[i], end=' ')
