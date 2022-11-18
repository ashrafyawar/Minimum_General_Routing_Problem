# coding=utf-8
# Import packages:
import random
import math
import pygame
from pygame.locals import *
import timeit
import time
import datetime
import matplotlib.pyplot as plt
from sys import maxsize
from itertools import permutations

########################################################################
# Global settings:
SIZE = 545  # size of display window for single instance
STATUS_HEIGHT = 80  # height of status bar in display window
STATUS_HEIGHT2 = 30  # height of status bar within instance subwindows
STATUS_HEIGHT3 = 45  # height of status bar at the bottom (Github info)
DELIM_WIDTH = 5  # width of delimiter of the circles output
CITY_RADIUS = 5  # radius of circle representing city
FONTSIZE = 20  # font size for control section buttons
VERBOSE = False  # level of chattiness
SAVEPLOT = True  # save plot of tour length vs iteration (True) or only display it (False)
SLEEP = 0  # delay (in seconds) after plotting new configuration
N = 200  # initial number of cities
SEED = None  # random seed
VERSION = "1.0"  # version
COLORS = {"WHITE": (255, 255, 255), "RED": (255, 0, 0), "GREEN": (0, 255, 0), "BLUE": (0, 0, 255), "BLACK": (0, 0, 0),
          "YELLOW": (255, 255, 0),
          "LIGHT_BLUE": (0, 125, 227), "GREY1": (120, 120, 120), "GREY2": (224, 224, 224), "LIGHTBLUE": (102, 178, 255),
          "LIGHTRED": (255, 153, 153), "LIGHTYELLOW": (255, 255, 153), "PINK": (255, 51, 255), "DARKBLUE": (0, 0, 153),
          "LAVENDER": (204, 153, 255), "LIGHTGREEN": (153, 255, 204), "BROWN": (102, 51, 0), "OLIVE": (153, 153, 0),
          "DARKGREY": (105, 105, 105)}


########################################################################
# Helper functions:
def distance(x, y):
    # get Euclidean distance between two points (cities) in 2D space
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)


def tour_length(cities, N):
    # get total tour length for current sequence of cities
    assert len(cities) == N, "? " + str(len(cities)) + " vs " + str(N)
    return sum(distance(cities[k + 1], cities[k]) for k in range(N - 1)) + distance(cities[0], cities[N - 1])


def draw_text(surface, font, text, position, color):
    # draw user-defined text in pygame graphics surface
    lable = font.render(text, 1, color)
    surface.blit(lable, position)


def generate_cities(N):
    # generate positions for N cities randomly in range .025 <= x <= .975
    # and .025 <= y <= .975 (leave margins at all sides for aesthetics
    # reasons)
    random.seed(SEED)
    cities = [(.025 + random.uniform(0.0, 0.95), .025 + random.uniform(0.0, 0.95)) for i in range(N)]
    random.seed()
    print(cities)
    return cities[:]


def change_N(N):
    # change number of cities, so various variables have to be reset
    iters, siters, ex_iters = 0, 0, 0
    d_energy_min, s_energy_min = float('inf'), 10000.
    beta, n_accept = 1.0, 0
    ex_cities = generate_cities(N)
    gen_chris_cities = ex_cities[:]
    if VERBOSE: print
    "Simulating", N, "cities."
    return N, ex_cities, gen_chris_cities, iters, siters, ex_iters, d_energy_min, s_energy_min, beta, n_accept, timeit.default_timer(), {
        "exhaustive": {"iters": [], "lengths": []}, "gen_chris": {"iters": [], "lengths": []}}


def get_filename(N, iters):
    # generate file (without extension),
    # contains number of cities, iteration number and timestamp
    filename = "Images/TSP_comparism_N" + str(N) + "_iters" + str(iters) + "_" + \
               datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return filename


def save_image(surface, N, iters):
    # save current graphics surface as image (PNG format); filename
    filename = get_filename(N, iters) + "_map.png"
    pygame.image.save(surface, filename)
    return filename


def make_plot(plot_data, N, iters):
    # generate plot of minimal tour length vs iteration number for both
    # exhaustive_approach and generalized_christofides; plot has log-log-scale;
    # plot is saved to file with filename containing N, iteration and
    # timestamp
    filename = get_filename(N, iters) + "_plot.png"
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    xmax = max(plot_data["exhaustive"]["iters"][-1], plot_data["gen_chris"]["iters"][-1])
    x1 = plot_data["exhaustive"]["iters"]
    y1 = plot_data["exhaustive"]["lengths"]
    x2 = plot_data["gen_chris"]["iters"]
    y2 = plot_data["gen_chris"]["lengths"]
    if x1[-1] < x2[-1]:
        x1.append(x2[-1])
        y1.append(y1[-1])
    elif x1[-1] > x2[-1]:
        x2.append(x1[-1])
        y2.append(y2[-1])
    plt.plot(x1, y1, color="red", lw=2, label="Exhaustive Approach")
    plt.plot(x2, y2, color="blue", lw=2, label=" Generalized Christofide's Approach")
    plt.legend(loc=3, fontsize=12)
    plt.xlabel("Iteration Count", fontsize=14)
    plt.ylabel(" Min Tour Length", fontsize=14)
    plt.title("Iteration Count vs Min Tour Length", fontsize=18)
    if SAVEPLOT:
        plt.savefig(filename)
    plt.show()
    return filename


def get_length(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1.0 / 2.0)


def build_graph(data):
    graph = {}
    for this in range(len(data)):
        for another_point in range(len(data)):
            if this != another_point:
                if this not in graph:
                    graph[this] = {}

                graph[this][another_point] = get_length(data[this][0], data[this][1], data[another_point][0],
                                                        data[another_point][1])

    return graph


class UnionFind:
    def __init__(self):
        self.weights = {}
        self.parents = {}

    def __getitem__(self, object):
        if object not in self.parents:
            self.parents[object] = object
            self.weights[object] = 1
            return object

        # find path of objects leading to the root
        path = [object]
        root = self.parents[object]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]

        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def __iter__(self):
        return iter(self.parents)

    def union(self, *objects):
        roots = [self[x] for x in objects]
        heaviest = max([(self.weights[r], r) for r in roots])[1]
        for r in roots:
            if r != heaviest:
                self.weights[heaviest] += self.weights[r]
                self.parents[r] = heaviest


def minimum_spanning_tree(G):
    tree = []
    subtrees = UnionFind()
    for W, u, v in sorted((G[u][v], u, v) for u in G for v in G[u]):
        if subtrees[u] != subtrees[v]:
            tree.append((u, v, W))
            subtrees.union(u, v)

    return tree


def find_odd_vertexes(MST):
    tmp_g = {}
    vertexes = []
    for edge in MST:
        if edge[0] not in tmp_g:
            tmp_g[edge[0]] = 0

        if edge[1] not in tmp_g:
            tmp_g[edge[1]] = 0

        tmp_g[edge[0]] += 1
        tmp_g[edge[1]] += 1

    for vertex in tmp_g:
        if tmp_g[vertex] % 2 == 1:
            vertexes.append(vertex)

    return vertexes


def minimum_weight_matching(MST, G, odd_vert):
    import random
    random.shuffle(odd_vert)

    while odd_vert:
        v = odd_vert.pop()
        length = float("inf")
        u = 1
        closest = 0
        for u in odd_vert:
            if v != u and G[v][u] < length:
                length = G[v][u]
                closest = u

        MST.append((v, closest, length))
        odd_vert.remove(closest)


def find_eulerian_tour(MatchedMSTree, G):
    neighbours = {}
    for edge in MatchedMSTree:
        if edge[0] not in neighbours:
            neighbours[edge[0]] = []

        if edge[1] not in neighbours:
            neighbours[edge[1]] = []

        neighbours[edge[0]].append(edge[1])
        neighbours[edge[1]].append(edge[0])

    # finds the hamiltonian circuit
    start_vertex = MatchedMSTree[0][0]
    EP = [neighbours[start_vertex][0]]

    while len(MatchedMSTree) > 0:
        for i, v in enumerate(EP):
            if len(neighbours[v]) > 0:
                break

        while len(neighbours[v]) > 0:
            w = neighbours[v][0]

            remove_edge_from_matchedMST(MatchedMSTree, v, w)
            del neighbours[v][(neighbours[v].index(w))]
            del neighbours[w][(neighbours[w].index(v))]
            i += 1
            EP.insert(i, w)
            v = w
    return EP


def remove_edge_from_matchedMST(MatchedMST, v1, v2):
    for i, item in enumerate(MatchedMST):
        if (item[0] == v2 and item[1] == v1) or (item[0] == v1 and item[1] == v2):
            del MatchedMST[i]
    return MatchedMST


def e(cities):
    random.shuffle(cities)
    return cities


def s(N, cities, beta, n_accept, best_energy):
    energy = tour_length(cities, N)
    new_route = False
    if n_accept >= 100 * math.log(N):
        beta *= 1.005
        n_accept = 0
    p = random.uniform(0.0, 1.0)
    if p < 0.2:
        i = random.randint(0, N / 2)
        cities = cities[i:] + cities[:i]
        i = random.randint(0, N / 2)
        a = cities[:i]
        a.reverse()
        new_cities = a + cities[i:]
    elif p < 0.6:
        new_cities = cities[:]
        i = random.randint(1, N - 1)
        a = new_cities.pop(i)
        j = random.randint(1, N - 2)
        new_cities.insert(j, a)
    else:
        new_cities = cities[:]
        i = random.randint(1, N - 1)
        j = random.randint(1, N - 1)
        new_cities[i] = cities[j]
        new_cities[j] = cities[i]
    new_energy = tour_length(new_cities, N)
    if random.uniform(0.0, 1.0) < math.exp(- beta * (new_energy - energy)):
        n_accept += 1
        energy = new_energy
        cities = new_cities[:]
        if energy < best_energy:
            best_energy = energy
            best_tour = cities[:]
            new_route = True
    return cities, beta, n_accept, best_energy, new_route


########################################################################
# Initialisation:
start_timer = timeit.default_timer()
pygame.init()
helv20 = pygame.font.SysFont("Helvetica", 20)
helv24 = pygame.font.SysFont("Helvetica", 24)
# start clock:
fpsClock = pygame.time.Clock()
# set display surface for pygame:
SWIDTH = 2 * SIZE + DELIM_WIDTH
SHEIGHT = SIZE + STATUS_HEIGHT + STATUS_HEIGHT2 + STATUS_HEIGHT3 +70
print
SWIDTH, SHEIGHT
surface = pygame.display.set_mode((SWIDTH, SHEIGHT))
surface.set_alpha(None)
pygame.display.set_caption(
    "Minimum General Routing Problem: Exhaustive Approach vs Generalized Christofide's Approach ")
exhaustive_approach_cities = generate_cities(N)  # cities for exhaustive_approach
gen_chris_cities = exhaustive_approach_cities[:]  # cities for generalized_christofides


######################################################################
# Button class for control section, PyGame doesn't have ready-to-use
# buttons or similar:
class Button:

    def __init__(self, width, height, text, color, tcolor):
        self.width = width
        self.height = height
        self.text = text
        self.color = color
        self.tcolor = tcolor

    def SetText(self, text):
        self.text = text

    def PlaceButton(self, surface, x, y):
        self.x = x
        self.y = y
        surface = self.DrawButton(surface, x, y)
        surface = self.ButtonText(surface, x, y)

    def DrawButton(self, surface, x, y):
        pygame.draw.rect(surface, self.color, (x, y, self.width, self.height), 0)
        return surface

    def ButtonText(self, surface, x, y):
        font_size = int(self.width // len(self.text))
        font = pygame.font.SysFont("Arial", FONTSIZE)
        text = font.render(self.text, 1, self.tcolor)
        surface.blit(text, ((x + self.width / 2) - text.get_width() / 2, (y + self.height / 2) - text.get_height() / 2))
        return surface

    def IsPressed(self, mouse):
        return mouse[0] > self.x and \
               mouse[1] > self.y and \
               mouse[0] <= self.x + self.width and \
               mouse[1] < self.y + self.height


########################################################################
def exhaustive_approach(graph):
    # store all vertex apart from source vertex
    vertex = []
    s = 0
    for i in range(len(graph)):
        if i != s:
            vertex.append(i)

    # store minimum weight Hamiltonian Cycle
    min_path = maxsize
    next_permutation = permutations(vertex)
    for i in next_permutation:

        # store current Path weight(cost)
        current_pathweight = 0

        # compute current path weight
        k = s
        for j in i:
            current_pathweight += graph[k][j]
            k = j
        current_pathweight += graph[k][s]

        # update minimum
        min_path = min(min_path, current_pathweight)

    return min_path
########################################################################


# generalized_christofides_approach(
def generalized_christofides_approach(data):
    # build a graph
    G = build_graph(data)
    print("Graph: ", G)

    # build a minimum spanning tree
    MSTree = minimum_spanning_tree(G)
    print("MSTree: ", MSTree)

    # find odd vertexes
    odd_vertexes = find_odd_vertexes(MSTree)
    print("Odd vertexes in MSTree: ", odd_vertexes)

    # add minimum weight matching edges to MST
    minimum_weight_matching(MSTree, G, odd_vertexes)
    print("Minimum weight matching: ", MSTree)

    # find an eulerian tour
    eulerian_tour = find_eulerian_tour(MSTree, G)
    print("Eulerian tour: ", eulerian_tour)

    current = eulerian_tour[0]
    path = [current]
    visited = [False] * len(eulerian_tour)
    visited[0] = True
    length = 0

    for v in eulerian_tour[1:]:
        if not visited[v]:
            path.append(v)
            visited[v] = True

            length += G[current][v]
            current = v

    path.append(path[0])

    print("Result path: ", path)
    # print("Result length of the path: ", length)
    return length, path


########################################################################
# Main loop:
def mainloop(surface, N, ex_cities, gen_chris_cities, start_timer):
    d_energy_min = float('inf')
    s_energy_min = 10000
    running = True
    iters, siters, ex_iters = 0, 0, 0
    start = timeit.default_timer()
    speed = 0
    # parameters for generalized_christofides:
    beta = 1.0  # inverse temperature
    n_accept = 0
    # plotting data:
    plot_data = {"exhaustive": {"iters": [], "lengths": []}, "gen_chris": {"iters": [], "lengths": []}}

    # define buttons for user control:
    button_ncity_10 = Button(50, 30, "10", COLORS["LIGHTBLUE"], COLORS["BLACK"])
    button_ncity_20 = Button(50, 30, "20", COLORS["LIGHTBLUE"], COLORS["BLACK"])
    button_ncity_50 = Button(50, 30, "50", COLORS["LIGHTBLUE"], COLORS["BLACK"])
    button_ncity_100 = Button(50, 30, "100", COLORS["LIGHTBLUE"], COLORS["BLACK"])
    button_ncity_200 = Button(50, 30, "200", COLORS["LIGHTBLUE"], COLORS["BLACK"])
    button_ncity_500 = Button(50, 30, "500", COLORS["LIGHTBLUE"], COLORS["BLACK"])
    button_quit = Button(60, 30, "Quit", COLORS["RED"], COLORS["BLACK"])
    button_pause = Button(60, 30, "Pause", COLORS["LIGHTBLUE"], COLORS["BLACK"])
    button_continue = Button(65, 30, "Resume", COLORS["GREEN"], COLORS["BLACK"])
    button_save = Button(60, 30, "Save", COLORS["LAVENDER"], COLORS["BLACK"])
    button_plot = Button(60, 30, "Plot", COLORS["LAVENDER"], COLORS["BLACK"])

    # loop until user event:
    while True:

        # Event handler:
        for event in pygame.event.get():
            # pygame event handler
            if event.type == QUIT:
                # graphics window is closed
                pygame.quit()
                return
            elif event.type == KEYDOWN:
                # key is pressed
                if event.key in [K_ESCAPE, K_q]:
                    # 'q' or ESC will quit program
                    pygame.quit()
                    return
                elif event.key == K_c:
                    # 'c' continues simulation
                    running = True
                elif event.key == K_p:
                    # 'p' pauses simulation
                    running = False
            elif event.type == MOUSEBUTTONDOWN:
                # mouse button is pressed
                if button_ncity_10.IsPressed(pygame.mouse.get_pos()):
                    # N = 10 selected
                    N, ex_cities, gen_chris_cities, iters, siters, ex_iters, d_energy_min, s_energy_min, beta, n_accept, start, plot_data = change_N(
                        10)
                elif button_ncity_20.IsPressed(pygame.mouse.get_pos()):
                    # N = 20 selected
                    N, ex_cities, gen_chris_cities, iters, siters, ex_iters, d_energy_min, s_energy_min, beta, n_accept, start, plot_data = change_N(
                        20)
                elif button_ncity_50.IsPressed(pygame.mouse.get_pos()):
                    # N = 50 selected
                    N, ex_cities, gen_chris_cities, iters, siters, ex_iters, d_energy_min, s_energy_min, beta, n_accept, start, plot_data = change_N(
                        50)
                elif button_ncity_100.IsPressed(pygame.mouse.get_pos()):
                    # N = 100 selected
                    N, ex_cities, gen_chris_cities, iters, siters, ex_iters, d_energy_min, s_energy_min, beta, n_accept, start, plot_data = change_N(
                        100)
                elif button_ncity_200.IsPressed(pygame.mouse.get_pos()):
                    # N = 200 selected
                    N, ex_cities, gen_chris_cities, iters, siters, ex_iters, d_energy_min, s_energy_min, beta, n_accept, start, plot_data = change_N(
                        200)
                elif button_ncity_500.IsPressed(pygame.mouse.get_pos()):
                    # N = 500 selected
                    N, ex_cities, gen_chris_cities, iters, siters, ex_iters, d_energy_min, s_energy_min, beta, n_accept, start, plot_data = change_N(
                        500)
                elif button_quit.IsPressed(pygame.mouse.get_pos()):
                    # 'Quit' selected
                    if VERBOSE: print
                    "Quitting..."
                    pygame.quit()
                    return
                elif button_continue.IsPressed(pygame.mouse.get_pos()):
                    # 'Continue' selected, simulation continues
                    if VERBOSE: print
                    "Continuing..."
                    running = True
                elif button_pause.IsPressed(pygame.mouse.get_pos()):
                    # 'Pause' selected, simulation is halted
                    if VERBOSE: print
                    "Simulation paused."
                    running = False
                elif button_save.IsPressed(pygame.mouse.get_pos()):
                    # 'Save' selected, image will be saved
                    filename = save_image(surface, N, iters)
                    if VERBOSE: print
                    "Image saved, filename:", filename
                elif button_plot.IsPressed(pygame.mouse.get_pos()):
                    # 'Plot' selected, generate plot of tour length vs iteration
                    filename = make_plot(plot_data, N, iters)
                    if VERBOSE: print
                    "Plot generated, filename:", filename

        if not running:
            # if simulation is paused, skip rest of mainloop
            continue
        if VERBOSE and iters % 10000 == 0:
            print("N/iters/beta/s_energy_min =", N, iters, beta, round(s_energy_min, 3))
        iters += 1  # iteration counter

        change = False
        # generate new route by exhaustive_approach:
        # new_ex_cities_temp = exhaustive_approach(ex_cities[:])
        new_ex_cities = e(ex_cities[:])
        d_energy = tour_length(new_ex_cities, N)
        if d_energy < d_energy_min:
            d_energy_min = d_energy
            if VERBOSE:
                print("Tour length exhaustive_approach:", d_energy_min, "at iteration", iters)
            ex_cities = new_ex_cities[:]
            ex_iters = iters
            change = True
            plot_data["exhaustive"]["lengths"].append(d_energy_min)
            plot_data["exhaustive"]["iters"].append(iters)

        # generate new route by generalized_christofides_approach:
        # generalized_christofides_approach(gen_chris_cities[:])
        new_gen_chris_cities, beta, n_accept, s_energy_min, new_route = s(N, gen_chris_cities[:], beta, n_accept, s_energy_min)

        if new_route:
            if VERBOSE:
                print("Tour length generalized_christofides_approach:", s_energy_min, "at iteration", iters)
            gen_chris_cities = new_gen_chris_cities[:]
            siters = iters
            change = True
            plot_data["gen_chris"]["lengths"].append(s_energy_min)
            plot_data["gen_chris"]["iters"].append(iters)

        if iters % 1000 == 0:
            # every 1k iterations we will plot current configuration even if
            # it hasn't changed
            change = True

        if change:

            # calculate simulation speed:
            delta_iter = 100000 // N
            show_speed = iters % delta_iter == 0
            if show_speed:
                T = timeit.default_timer() - start
                start = timeit.default_timer()
                speed = delta_iter / T
            # only generate graphics output if new route is present for
            # exhaustive_approach and/or generalized_christofides or if iteration
            # count is divisible by 1000
            #
            # buttons and text elements:
            surface.fill(COLORS["BLACK"])
            surface.fill(COLORS["BLACK"], (0, 0, 2 * SIZE + DELIM_WIDTH, STATUS_HEIGHT))
            surface.fill(COLORS["WHITE"], (SIZE, STATUS_HEIGHT, DELIM_WIDTH, SIZE + STATUS_HEIGHT2+80))
            surface.fill(COLORS["WHITE"], (0, STATUS_HEIGHT, DELIM_WIDTH, SIZE + STATUS_HEIGHT2 +80))
            surface.fill(COLORS["WHITE"], (2 * SIZE, STATUS_HEIGHT, DELIM_WIDTH, SIZE + STATUS_HEIGHT2 + 80))
            surface.fill(COLORS["GREY1"], (0, SIZE + STATUS_HEIGHT + STATUS_HEIGHT2 + 10, 2 * SIZE + DELIM_WIDTH, DELIM_WIDTH))
            surface.fill(COLORS["GREY1"], (0, SIZE + STATUS_HEIGHT + STATUS_HEIGHT2 + 75, 2 * SIZE + DELIM_WIDTH, DELIM_WIDTH))

            surface.fill(COLORS["WHITE"], (0, STATUS_HEIGHT, 2 * SIZE + DELIM_WIDTH, DELIM_WIDTH))
            draw_text(surface, helv24, "Graduation Project 01 INSTRUCTOR: Doç. Dr. Didem GÖZÜPEK,  STUDENT: Mohammad Ashraf YAWAR 161044123 ",
                      (SIZE // 2-250, SIZE + STATUS_HEIGHT + STATUS_HEIGHT2 + DELIM_WIDTH + 80), COLORS["YELLOW"])
            draw_text(surface, helv24, "City Count:", (110, 10), COLORS["WHITE"])
            draw_text(surface, helv24, str(N), (250, 10), COLORS["YELLOW"])
            button_ncity_10.PlaceButton(surface, 10, 40)
            button_ncity_20.PlaceButton(surface, 70, 40)
            button_ncity_50.PlaceButton(surface, 130, 40)
            button_ncity_100.PlaceButton(surface, 190, 40)
            button_ncity_200.PlaceButton(surface, 250, 40)
            button_ncity_500.PlaceButton(surface, 310, 40)
            pygame.draw.line(surface, COLORS["GREY1"], (380, 0), (380, STATUS_HEIGHT), 3)
            draw_text(surface, helv24, "Iterations:", (400, 10), COLORS["WHITE"])
            draw_text(surface, helv24, str(iters // 1000) + " k", (400, 40), COLORS["YELLOW"])
            pygame.draw.line(surface, COLORS["GREY1"], (520, 0), (520, STATUS_HEIGHT), 3)
            draw_text(surface, helv24, "Tour Length Ratio:", (540, 10), COLORS["WHITE"])
            draw_text(surface, helv24, str(round(d_energy_min / s_energy_min, 3)), (540, 40), COLORS["YELLOW"])
            pygame.draw.line(surface, COLORS["GREY1"], (750, 0), (750, STATUS_HEIGHT), 3)
            draw_text(surface, helv24, "ite/sec", (770, 10), COLORS["WHITE"])
            draw_text(surface, helv24, str(int(round(speed, 0))), (770, 40), COLORS["YELLOW"])
            pygame.draw.line(surface, COLORS["GREY1"], (2 * SIZE + DELIM_WIDTH - 230, 0),
                             (2 * SIZE + DELIM_WIDTH - 230, STATUS_HEIGHT), 3)
            button_quit.PlaceButton(surface, 2 * SIZE + DELIM_WIDTH - 210, 10)
            button_pause.PlaceButton(surface, 2 * SIZE + DELIM_WIDTH - 210, 45)
            button_continue.PlaceButton(surface, 2 * SIZE + DELIM_WIDTH - 140, 45)
            button_save.PlaceButton(surface, 2 * SIZE + DELIM_WIDTH - 70, 10)
            button_plot.PlaceButton(surface, 2 * SIZE + DELIM_WIDTH - 70, 45)

            # draw cities and roads:
            # exhaustive_approach:
            for i in range(N):
                x1, y1 = ex_cities[i]
                x2, y2 = ex_cities[(i + 1) % N]
                xi1 = int(SIZE * x1)
                xi2 = int(SIZE * x2)
                yi1 = STATUS_HEIGHT + STATUS_HEIGHT2 + int(SIZE * y1)
                yi2 = STATUS_HEIGHT + STATUS_HEIGHT2 + int(SIZE * y2)
                pygame.draw.line(surface, COLORS["RED"], [xi1, yi1], [xi2, yi2], 3)

            for x, y in ex_cities:
                xi = int(SIZE * x)
                yi = STATUS_HEIGHT + STATUS_HEIGHT2 + int(SIZE * y)
                pygame.draw.circle(surface, COLORS["YELLOW"], [xi, yi], CITY_RADIUS)

            draw_text(surface, helv24, "Iterations:", (30, STATUS_HEIGHT + 10), COLORS["WHITE"])
            draw_text(surface, helv24, str(ex_iters), (150, STATUS_HEIGHT + 10), COLORS["YELLOW"])
            draw_text(surface, helv24, "Min. Tour Length:", (260, STATUS_HEIGHT + 10), COLORS["WHITE"])
            draw_text(surface, helv24, "Exhaustive Approach", (190, SIZE + STATUS_HEIGHT + STATUS_HEIGHT2 + 30), COLORS["WHITE"])

            draw_text(surface, helv24, str(round(d_energy_min, 3)), (460, STATUS_HEIGHT + 10), COLORS["YELLOW"])

            # generalized_christofides:
            for i in range(N):
                x1, y1 = gen_chris_cities[i]
                x2, y2 = gen_chris_cities[(i + 1) % N]
                xi1 = SIZE + DELIM_WIDTH + int(SIZE * x1)
                xi2 = SIZE + DELIM_WIDTH + int(SIZE * x2)
                yi1 = STATUS_HEIGHT + STATUS_HEIGHT2 + int(SIZE * y1)
                yi2 = STATUS_HEIGHT + STATUS_HEIGHT2 + int(SIZE * y2)
                pygame.draw.line(surface, COLORS["WHITE"], [xi1, yi1], [xi2, yi2], 3)

            for x, y in ex_cities:
                xi = SIZE + DELIM_WIDTH + int(SIZE * x)
                yi = STATUS_HEIGHT + STATUS_HEIGHT2 + int(SIZE * y)
                pygame.draw.circle(surface, COLORS["YELLOW"], [xi, yi], CITY_RADIUS)

            draw_text(surface, helv24, "Iterations:", (SIZE + DELIM_WIDTH + 30, STATUS_HEIGHT + 10), COLORS["WHITE"])
            draw_text(surface, helv24, str(siters), (SIZE + DELIM_WIDTH + 150, STATUS_HEIGHT + 10), COLORS["YELLOW"])
            draw_text(surface, helv24, "Min. Tour Length:", (SIZE + DELIM_WIDTH + 260, STATUS_HEIGHT + 10), COLORS["WHITE"])
            draw_text(surface, helv24, "Generalized Christofides", (SIZE + DELIM_WIDTH + 190, SIZE + STATUS_HEIGHT + STATUS_HEIGHT2 + 30), COLORS["WHITE"])

            draw_text(surface, helv24, str(round(s_energy_min, 3)), (SIZE + DELIM_WIDTH + 460, STATUS_HEIGHT + 10),
                      COLORS["YELLOW"])

            # update graphics output:
            pygame.display.flip()
            # wait a moment (SLEEP may be zero):
            time.sleep(SLEEP)


if __name__ == "__main__":
    mainloop(surface, N, exhaustive_approach_cities, gen_chris_cities, start_timer)
