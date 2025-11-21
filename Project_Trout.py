import random
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
#from Heron_Functions import UpdateHerons

np.random.seed(1)

# Define global variables for each species
water, trout, heron, fishermen = 0, 1, 2, 3

#------------------[This Stuff Seems to Work/Not Working on Currently]------------------------#

# Function to plot the spatial distribution of the environment
def plotSpatial(array, fileNumber):

    cmap = colors.ListedColormap(['#ABD9FC',"#599041", "#000000", "#CC0000"])
    bounds = [0,1,2,3,4]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(7, 6))
    plt.pcolor(array, cmap=cmap, edgecolors='k', linewidths=1, norm=norm)

    cbar = plt.colorbar(ticks=[0.5,1.5,2.5,3.5])
    cbar.ax.set_yticklabels(['water','trout', 'herons', 'fishermen'])

    plt.show()

def create_agent_domain(Xsize: int, Ysize: int, deep_water_prob: list, shallow_water_prob: list):

    # Generates initial deep water section of the domain containing water, trout, and heron
    output_array = np.random.choice([water, trout, heron], size=(Xsize, Ysize), p=deep_water_prob)

    # Generates shallow water section (edges) of the domain to include only water, trout, and fishermen
    for row in range(Xsize):
        for column in range(Ysize):
            if not (0 < row < (Xsize - 1)) or not (0 < column < (Ysize - 1)):
                output_array[row][column] = np.random.choice([water, trout, fishermen], p=shallow_water_prob)

    return output_array



#------------------[Experimental Stuff]-----------------------#

def main():

    sizeX, sizeY = 30, 30  # Define the size of the grid (30x30 cells)
    shallow_water_prob = [.75,.20,.05] # Define the probabilities for water, trout, and fishermen to appear in shallow water
    deep_water_prob = [.75,.20,.05] # Define the probabilities for water, trout, and herons to appear in deep water

    # TEST VARIABLES GET RID OF LATER
    shallow_water_prob = [.7, .2, .1]
    deep_water_prob = [.7, .3, 0]

    # Set the number of iterations and markers for 4 capturing PNGs of the domain at equal intervals
    max_iterations = 52
    iteration_markers = [1,max_iterations // 4, max_iterations // 2, max_iterations * 3 // 4, max_iterations]

    # Create and plot the initial state of the grid at time = 0
    currTime = 0
    simTime, water_pop, trout_pop, heron_pop = [],[],[],[]
    domain = create_agent_domain(sizeX, sizeY, deep_water_prob, shallow_water_prob)  # Create the initial domain grid
    plotSpatial(domain, currTime)

    # Run iterations with updates to populations and interactions
    for i in range(1,max_iterations+1):

        simTime.append(currTime)
        water_pop.append(np.count_nonzero(domain == 0))  # Count cells with water
        trout_pop.append(np.count_nonzero(domain == 1))  # Count cells with trout
        heron_pop.append(np.count_nonzero(domain == 2))  # Count cells with herons

        domain = update_trout_pop(domain)
        domain = UpdateHerons(domain)

        currTime +=1

        if i in iteration_markers:
            plotSpatial(domain, currTime)
            print(f"Trout Population: {trout_pop[i-1]}")
            print(f"Heron Population: {heron_pop[i-1]}")

        print(f"-------------- End Iteration {i} ---------------------")

    fig2 = plt.subplot()
    fig2.plot(simTime,trout_pop,heron_pop)
    fig2.set_xlabel("Time")
    fig2.set_ylabel("Population")
    plt.show()

def update_trout_pop(array):

    num_rows = array.shape[0]
    num_columns = array.shape[1]
    # trout_mortality = .50
    # trout_survival_rate = .43

    #TEST VARIABLES GET RID OF LATER
    trout_mortality = 1
    trout_survival_rate = 0

    array_updated_trout = array.copy()

    # Simulate trout population changes due to overpopulation/competition for resources
    for row in range(num_rows):
        for column in range(num_columns):
            if array[row][column] == trout:

                # print(f"Row: {row+1}, Column: {column+1}")

                crowded_survival_chance = [trout]

                if row != 0:
                    if array[row - 1][column] == trout:
                        crowded_survival_chance.append(water)

                if row != (num_rows - 1):
                    if array[row + 1][column] == trout:
                        crowded_survival_chance.append(water)

                if column != 0:
                    if array[row][column - 1] == trout:
                        crowded_survival_chance.append(water)

                if column != (num_columns - 1):
                    if array[row][column + 1] == trout:
                        crowded_survival_chance.append(water)

                if len(crowded_survival_chance) != 0:
                    array_updated_trout[row][column] = random.choice(crowded_survival_chance)

                    # print(f"Trout at row:{row+1} and column:{column+1} moved to row: {new_location[0]+1} and column {new_location[1]+1}")

    # Simulate trout population changes due to natural causes and juvenile trout survival to adulthood
    for row in range(num_rows):
        for column in range(num_columns):
            if array[row][column] == trout:
                array_updated_trout[row][column] = np.random.choice([water,trout], p = (trout_mortality, (1-trout_mortality)))
                #if array_updated_trout[row][column] == water:
                    #print(f"Trout Died @ ({row+1},{column+1})")
            if array[row][column] == water:
                array_updated_trout[row][column] = np.random.choice([water,trout], p = ((1-trout_survival_rate), (trout_survival_rate)))
                #if array_updated_trout[row][column] == trout:
                    #print(f"Trout Born @ ({row+1},{column+1})")

    # Simulate movement of trout
    for row in range(num_rows):
        for column in range(num_columns):
            if array[row][column] == trout:

                #print(f"Row: {row+1}, Column: {column+1}")

                places_to_move = []

                if row != 0:
                    if array[row-1][column] == water:
                        places_to_move.append([row-1, column])

                if row != (num_rows - 1):
                    if array[row+1][column] == water:
                        places_to_move.append([row+1, column])

                if column != 0:
                    if array[row][column-1] == water:
                        places_to_move.append([row, column-1])

                if column != (num_columns - 1):
                    if array[row][column+1] == water:
                        places_to_move.append([row, column+1])


                if len(places_to_move) != 0:
                    new_location = random.choice(places_to_move)
                    array_updated_trout[new_location[0]][new_location[1]] = trout
                    array_updated_trout[row][column] = water

                    #print(f"Trout at row:{row+1} and column:{column+1} moved to row: {new_location[0]+1} and column {new_location[1]+1}")

    return array_updated_trout

def UpdateHerons(array):

    num_rows = array.shape[0]
    num_columns = array.shape[1]
    heron_mortality = .22
    heron_survival_rate = .31

    array_updated_herons = array.copy()

    # Simulate heron population changes due to natural causes.
    for row in range(num_rows):
        for column in range(num_columns):
            if array[row][column] == heron:
                array_updated_herons[row][column] = np.random.choice([water,heron], p = (heron_mortality, (1-heron_mortality)))
                #if array_updated_herons[row][column] == water:
                    #print(f"Heron Died @ ({row+1},{column+1})")
            if array[row][column] == water:
                array_updated_herons[row][column] = np.random.choice([water,heron], p = ((1-heron_survival_rate), (heron_survival_rate)))
                #if array_updated_herons[row][column] == heron:
                    #print(f"Heron Born @ ({row+1},{column+1})")

    # Simulate herons preying on trout
        for row in range(num_rows):
            for column in range(num_columns):
                if array[row][column] == heron:

                    #print(f"Row: {row + 1}, Column: {column + 1}")

                    places_to_prey = []

                    if row != 0:
                        if array[row - 1][column] == trout:
                            places_to_prey.append([row - 1, column])

                    if row != (num_rows - 1):
                        if array[row + 1][column] == trout:
                            places_to_prey.append([row + 1, column])

                    if column != 0:
                        if array[row][column - 1] == trout:
                            places_to_prey.append([row, column - 1])

                    if column != (num_columns - 1):
                        if array[row][column + 1] == trout:
                            places_to_prey.append([row, column + 1])

                    if len(places_to_prey) != 0:
                        new_location = random.choice(places_to_prey)
                        array_updated_herons[new_location[0]][new_location[1]] = heron

                        for coordinates in places_to_prey:

                            array_updated_herons[coordinates[0]][coordinates[1]] = water

                        #print(f"Heron at row:{row + 1} and column:{column + 1} attacked row: {new_location[0] + 1} and column {new_location[1] + 1}")

    # Simulate movement of herons
    for row in range(num_rows):
        for column in range(num_columns):
            if array[row][column] == heron:

               # print(f"Row: {row+1}, Column: {column+1}")

                places_to_move = []

                if row != 0:
                    if array[row-1][column] == water:
                        places_to_move.append([row-1, column])

                if row != (num_rows - 1):
                    if array[row+1][column] == water:
                        places_to_move.append([row+1, column])

                if column != 0:
                    if array[row][column-1] == water:
                        places_to_move.append([row, column-1])

                if column != (num_columns - 1):
                    if array[row][column+1] == water:
                        places_to_move.append([row, column+1])


                if len(places_to_move) != 0:
                    new_location = random.choice(places_to_move)
                    array_updated_herons[new_location[0]][new_location[1]] = heron
                    array_updated_herons[row][column] = water

                    #print(f"Heron at row:{row+1} and column:{column+1} moved to row: {new_location[0]+1} and column {new_location[1]+1}")

    return array_updated_herons


#yo this is cool

main()