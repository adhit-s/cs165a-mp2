from operator import itemgetter
import numpy as np
import heapq
import math
import sys
# import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

directions = { 'W': (0, -1), 'A': (-1, 0), 'S': (0, 1), 'D': (1, 0) }
reverse_directions = dict((v, k) for k, v in directions.items())
surround_1 = np.array([ 
    (0, 1), (1, 0), (0, -1), (-1, 0)
])
surround_2 = np.array([ 
    (0, 1), (1, 0), (0, -1), (-1, 0), 
    (-1, -1), (-1, 1), (1, 1), (1, -1),
])
surround_3 = np.array([ 
    (0, 1), (1, 0), (0, -1), (-1, 0), 
    (-1, -1), (-1, 1), (1, 1), (1, -1), 
    (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
])
surround_4 = np.array([ 
    (0, 1), (1, 0), (0, -1), (-1, 0), 
    (-1, -1), (-1, 1), (1, 1), (1, -1), 
    (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2), 
    (-1, 3), (0, 3), (1, 3), (3, 1), (3, 0), (3, -1), (-1, -3), (0, -3), (1, -3), (-3, -1), (-3, 0), (-3, 1)
])

def sim_move(pos, dir):
    return sim_move_2(pos, directions[dir])

def sim_move_2(pos, offsets):
    return (pos[0] + offsets[0], pos[1] + offsets[1])

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def dist(a, b):
    return euclidean_distance(a, b)

def sigmoid(x):
    return 1 / (1 + math.pow(math.e, -x))

class PrioritySet(object):
    def __init__(self):
        self.heap = []
        self.set = set()

    def push(self, pri, d):
        if not d in self.set:
            heapq.heappush(self.heap, (pri, d))
            self.set.add(d)

    def front(self):
        return self.heap[0]

    def pop(self):
        pri, d = heapq.heappop(self.heap)
        self.set.remove(d)
        return (pri, d)
    
    def __len__(self):
        return len(self.heap)

class Player1:
    def __init__(self, self_position, other_agent_position, params, verbosity = 1):
        self.turn = 0
        self.score = 0
        self.hunger = 50
        self.stamina = 50

        self.last_mpos = self_position
        self.last_opos = other_agent_position

        self.target_pos = None
        self.target_type = None

        self.p = params
        self.verbosity = verbosity

    def new_turn(self, dungeon_map, coins, potions, foods, self_position, other_agent_position):
        self.turn += 1
        if self.turn % 2 == 0:
            self.stamina += (1 if self.hunger > 0 else 0)
            self.hunger = max(0, self.hunger - 1)
        if self.hunger == 0:
            self.stamina = max(0, self.stamina - 1)

        self.mpos = self_position
        self.opos = other_agent_position

        self.map = np.array(dungeon_map)
        self.coins = np.array(coins)
        self.pots = np.array(potions)
        self.foods = np.array(foods)

        for coin in self.coins:
            self.edit_map(coin, 'coin')
        for pot in self.pots:
            self.edit_map(pot, 'pot')
        for food in self.foods:
            self.edit_map(food, 'food')
        self.edit_map(self.mpos, 'me')
        self.edit_map(self.opos, 'opp')

        if self.verbosity > 1:
            print(f'Turn: {self.turn} | Score: {self.score} | Hunger: {self.hunger} | Stamina: {self.stamina} | Mpos: {self.mpos} | Opos: {self.opos}')

    def read_map(self, pos):
        return self.map[pos[1]][pos[0]]
    
    def edit_map(self, pos, v):
        self.map[pos[1]][pos[0]] = v
        
    def move(self, dir):
        if dir != 'I':
            self.stamina = max(0, self.stamina - 1)
            npos = sim_move(self.mpos, dir)
            nblock = self.read_map(npos)
            if nblock == 'coin':
                p1.score += 1
            elif nblock == 'pot':
                p1.stamina = max(0, min(50, p1.stamina + 20))
            elif nblock == 'food':
                p1.hunger = max(0, min(50, p1.hunger + 30))
        self.last_mpos = self.mpos
        self.last_opos = self.opos
        if self.verbosity > 1:
            print('Move:', dir)
        return dir
    
    def is_traversable(self, pos):
        if pos[1] < 0 or pos[1] >= self.map.shape[0] or pos[0] < 0 or pos[0] >= self.map.shape[1]:
            return False
        p = self.read_map(pos)
        return p != 'wall' and p != 'opp'

    def nm(self, pos):
        return self.node_meta[pos[1]][pos[0]]
    
    def backtrace_path(self, start, goal):
        path = []
        curr_pos = goal
        curr_nm = self.nm(curr_pos)
        while curr_pos != start:
            delta = tuple(np.subtract(curr_pos, curr_nm['parent']))
            path.append(reverse_directions[delta])
            curr_pos = curr_nm['parent']
            curr_nm = self.nm(curr_pos)
        path.reverse()
        if self.verbosity > 1:
            print('Start:', start, '| Goal:', goal, '| Path:', path)
        return path

    def a_star_path(self, start, goal):
        start = tuple(start)
        goal = tuple(goal)
        self.node_meta = []
        for r in range(self.map.shape[0]):
            row = []
            for c in range(self.map.shape[1]):
                row.append({
                    'f': float('inf'),
                    'g': float('inf'),
                    'h': float('inf'),
                    'parent': (-1, -1),
                    'closed': False
                })
            self.node_meta.append(row)

        start_nm = self.nm(start)
        start_nm['f'] = 0
        start_nm['g'] = 0
        start_nm['h'] = 0

        open = PrioritySet()
        open.push(0, start)

        while len(open) > 0:
            curr = open.pop()[1]
            curr_nm = self.nm(curr)
            curr_nm['closed'] = True

            for dir in directions:
                succ = sim_move(curr, dir)
                if self.is_traversable(succ):
                    succ_nm = self.nm(succ)
                    if succ == goal:
                        succ_nm['parent'] = curr
                        return self.backtrace_path(start, goal)
                    if not succ_nm['closed']:
                        g_new = curr_nm['g'] + 1
                        h_new = self.realistic_dist(succ, goal)
                        f_new = g_new + h_new - (3 if self.read_map(succ) == 'coin' else -3)
                        if succ_nm['f'] == float('inf') or succ_nm['f'] > f_new:
                            open.push(f_new, succ)
                            succ_nm['f'] = f_new
                            succ_nm['g'] = g_new
                            succ_nm['h'] = h_new
                            succ_nm['parent'] = curr
                            succ_nm['closed'] = True
        return []
    
    def set_target(self, target_pos, target_type):
        self.target_pos = target_pos
        self.target_type = target_type
        self.t = 0
        if self.verbosity > 0:
            print('Set Target:', self.target_type, '-', self.target_pos)
        if self.target_pos != None:
            self.curr_path = self.a_star_path(self.mpos, self.target_pos)
            if len(self.curr_path) == 0:
                return False
        if self.verbosity > 0:
            print('Target Path:', self.curr_path)
        return True

    def has_valid_target(self):
        return (self.target_pos is not None) and (self.read_map(self.target_pos) == self.target_type)
    
    def next_dir_to_target(self):  
        dir = self.curr_path[self.t]
        p = sim_move(self.mpos, dir)
        if dist(p, self.opos) == 0:
            self.set_target(self.target_pos, self.target_type)
            return 'I'
        self.t += 1
        return dir
    
    def realistic_dist(self, a, b):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        theta = math.atan2(dy, dx)
        r = math.sqrt(dx ** 2 + dy ** 2)
        d = 0
        for i in range(1, math.ceil(r)):
            i_dx = i * math.cos(theta)
            i_dy = i * math.sin(theta)
            i_pos = sim_move_2(a, (round(i_dx), round(i_dy)))
            if self.read_map(i_pos) == 'wall':
                d += 2
            else:
                d += 1
        return d

    def calc_cu(self, coin):
        mrd = self.realistic_dist(self.mpos, coin)
        u = self.p['cvw_m_dist'] * -mrd + self.p['cvw_o_dist'] * self.realistic_dist(self.opos, coin)
        for off in surround_2:
            neighbor = sim_move_2(coin, off)
            if self.is_traversable(neighbor) and self.read_map(neighbor) == 'coin':
                u += self.p['cvw_surround']
        for food in self.foods:
            if self.realistic_dist(coin, food) < self.p['cvw_food_range']:
                u += self.p['cvw_food_near']
        for pot in self.pots:
            if self.realistic_dist(coin, pot) < self.p['cvw_pot_range']:
                u += self.p['cvw_pot_near']
        if mrd < 15:
            path_to_coin = self.a_star_path(self.mpos, coin)
            cpos = self.mpos
            for move in path_to_coin:
                cpos = sim_move(cpos, move)
                if self.read_map(cpos) == 'coin':
                    u += self.p['cvw_coin_in_route']
        # u = sigmoid(u)
        return u
    
    def refresh_coin_utils(self):
        self.coin_utils = PrioritySet()
        cu_map = np.zeros(self.map.shape) - 100
        for coin in self.coins:
            u = self.calc_cu(coin)
            self.coin_utils.push(-u, tuple(coin))
            cu_map[coin[1]][coin[0]] = u
#         heatmap.set_data(cu_map)
#         fig.canvas.draw()
#         fig.canvas.flush_events()
#         plt.show()

# fig, ax = plt.subplots()
# heatmap = ax.imshow(np.zeros(shape=(30, 40)), cmap='hot', interpolation='none')
# cbar = plt.colorbar(heatmap)
# plt.ion()
# plt.show()

p1 = None

params = {
    'seek_food_range': 30,
    'low_hunger_thresh': 25,

    'reroute_food_range': 5,
    'high_hunger_thresh': 40,

    'seek_pot_range': 20,
    'low_stam_thresh': 25,

    'reroute_pot_range': 5,
    'high_stam_thresh': 40,

    'cvw_m_old_dist': 5,
    'cvw_m_dist': 7,
    'cvw_o_dist': 3,
    'cvw_surround': 7,
    'cvw_coin_in_route': 7,

    'cvw_food_range': 10,
    'cvw_food_near': 100,

    'cvw_pot_range': 10,
    'cvw_pot_near': 100
}

def player1_logic(coins, potions, foods, dungeon_map, self_position, other_agent_position):
    global p1

    if p1 is None:
        p1 = Player1(self_position, other_agent_position, params, verbosity = 0)
    p1.new_turn(dungeon_map, coins, potions, foods, self_position, other_agent_position)

    pot_dists = sorted([ (p1.realistic_dist(self_position, pot), tuple(pot)) for pot in p1.pots ], key=itemgetter(0))
    food_dists = sorted([ (p1.realistic_dist(self_position, food), tuple(food)) for food in p1.foods ], key=itemgetter(0))
    p1.refresh_coin_utils()

    if p1.turn % 50 == 0:
        p1.p['cvw_m_dist'] += 1
    # if p1.turn % 20 == 0:
    #     p1.p['cvw_surround'] = max(p1.p['cvw_surround'] - 10, 5)

    # n_surround = 0
    # for off in surround_3:
    #     n_pos = sim_move_2(p1.mpos, off)
    #     if p1.is_traversable(n_pos) and p1.read_map(n_pos) == 'coin':
    #         n_surround += 1
    # n_surround /= surround_3.shape[0]
    # if n_surround > 0.5 and p1.p['cvw_m_dist'] < 50:
    #     p1.p['cvw_m_old_dist'] = p1.p['cvw_m_dist']
    #     p1.p['cvw_m_dist'] = 50
    #     if p1.verbosity > 0:
    #         print('IN HIGH DENSITY AREA')
    # if n_surround < 0.2 and p1.p['cvw_m_dist'] >= 50:
    #     p1.p['cvw_m_dist'] = p1.p['cvw_m_old_dist']
    #     if p1.verbosity > 0:
    #         print('LEFT HIGH DENSITY AREA')

    next_dir = 'I'
    if (p1.stamina > 10 and p1.hunger > 0) or p1.hunger == 0:
        if p1.has_valid_target():
            if p1.target_type == 'coin':
                if len(food_dists) > 0 and p1.hunger < p1.p['high_hunger_thresh'] and food_dists[0][0] < p1.p['reroute_food_range']:
                    t = p1.set_target(food_dists[0][1], 'food')
                    if not t:
                        p1.set_target(None, None)
                elif len(pot_dists) > 0 and p1.stamina < p1.p['high_stam_thresh'] and pot_dists[0][0] < p1.p['reroute_pot_range']:
                    t = p1.set_target(pot_dists[0][1], 'pot')
                    if not t:
                        p1.set_target(None, None)
                elif p1.coin_utils.front()[1] != p1.target_pos and p1.calc_cu(p1.target_pos) + 10 < p1.coin_utils.front()[0]:
                    t = p1.set_target(p1.coin_utils.pop()[1], 'coin')
                    if not t:
                        p1.set_target(None, None)
            if dist(p1.mpos, p1.target_pos) == 0:
                p1.set_target(None, None)
            else:
                next_dir = p1.next_dir_to_target()
        if not p1.has_valid_target():
            if len(p1.foods) > 0 and p1.hunger < p1.p['low_hunger_thresh'] and food_dists[0][0] < p1.p['seek_food_range']:
                t = p1.set_target(food_dists[0][1], 'food')
                if not t:
                    p1.set_target(None, None)
                else:
                    next_dir = p1.next_dir_to_target()
            elif len(p1.pots) > 0 and p1.stamina < p1.p['low_stam_thresh'] and pot_dists[0][0] < p1.p['seek_pot_range']:
                t = p1.set_target(pot_dists[0][1], 'pot')
                if not t:
                    p1.set_target(None, None)
                else:
                    next_dir = p1.next_dir_to_target()
            elif len(p1.coins) > 0:
                p1.refresh_coin_utils()
                t = p1.set_target(p1.coin_utils.pop()[1], 'coin')
                if not t:
                    p1.set_target(None, None)
                else:
                    next_dir = p1.next_dir_to_target()
    return p1.move(next_dir)