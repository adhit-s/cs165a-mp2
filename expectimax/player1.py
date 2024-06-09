from operator import itemgetter
import numpy as np
import heapq
import math
import sys
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)

directions = { 'W': (0, -1), 'A': (-1, 0), 'S': (0, 1), 'D': (1, 0) }
reverse_directions = dict((v, k) for k, v in directions.items())
surrounding = np.array([ 
    (0, 1), (1, 0), (0, -1), (-1, 0), 
    (-1, -1), (-1, 1), (1, 1), (1, -1), 
    # (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2), 
    # (-1, 3), (0, 3), (1, 3), (3, 1), (3, 0), (3, -1), (-1, -3), (0, -3), (1, -3), (-3, -1), (-3, 0), (-3, 1)
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
    def __init__(self, self_position, other_agent_position, verbosity = 1):
        self.turn = 0
        self.score = 0
        self.hunger = 50
        self.stamina = 50

        self.last_mpos = self_position
        self.last_opos = other_agent_position
        self.visited = []

        self.target_pos = None
        self.target_type = None

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
        self.visited.append(self.mpos)

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
        if pos[1] < 0 or pos[1] >= self.map.shape[0] or pos[0] < 0 or pos[1] >= self.map.shape[1]:
            return False
        p = self.read_map(pos)
        return p != 'wall' and p != 'opp'

    def nm(self, pos):
        return self.node_meta[pos[1]][pos[0]]
    
    def backtrace_path(self):
        path = []
        curr_pos = self.target_pos
        curr_nm = self.nm(curr_pos)
        while curr_pos != self.start_pos:
            delta = tuple(np.subtract(curr_pos, curr_nm['parent']))
            path.append(reverse_directions[delta])
            curr_pos = curr_nm['parent']
            curr_nm = self.nm(curr_pos)
        path.reverse()
        if self.verbosity > 0:
            print('Start:', self.mpos, '| Goal:', self.target_pos, '| Path:', path)
        return path

    def a_star_path(self, start, goal):
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
                        return self.backtrace_path()
                    if not succ_nm['closed']:
                        g_new = curr_nm['g'] + 1
                        h_new = dist(succ, goal)
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
        self.start_pos = self.mpos
        self.target_pos = target_pos
        self.target_type = target_type
        self.t = 0
        if self.verbosity > 0:
            print('Set Target:', self.target_type, '-', self.target_pos)
        if self.target_pos != None:
            self.curr_path = self.a_star_path(self.mpos, self.target_pos)
            if len(self.curr_path) == 0:
                return False
        return True

    def has_valid_target(self):
        return (self.target_pos is not None) and (self.read_map(self.target_pos) == self.target_type)
    
    def next_dir_to_target(self):  
        dir = self.curr_path[self.t]
        p = sim_move(self.mpos, dir)
        if dist(p, self.opos) <= 1:
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
                d += 3
            else:
                d += 1
        return d

    def calc_pu(self, pos):
        if not self.is_traversable(pos):
            return -1000
        # u = cvw_o_dist * self.realistic_dist(pos, self.opos)
        u = 0
        if self.read_map(pos) == 'coin':
            u += 100
        elif self.read_map(pos) == 'pot':
            u += 50
        elif self.read_map(pos) == 'food':
            u += 50
        # for offset in surrounding:
        #     neighbor = sim_move_2(pos, offset)
        #     if self.is_traversable(neighbor) and self.read_map(neighbor) == 'coin':
        #         u += cvw_surround
        # for food in self.foods:
        #     if self.realistic_dist(pos, food) < cvw_food_range:
        #         u += cvw_food_near
        # for pot in self.pots:
        #     if self.realistic_dist(pos, pot) < cvw_pot_range:
        #         u += cvw_pot_near
        # u = sigmoid(u)
        return u
    
    def expectimax(self, pos, dir='', is_max=True, depth=3):
        if depth == 0:
            return [ self.calc_pu(pos), dir ]
        if dir != '' and self.is_traversable(sim_move(pos, dir)):
            pos = sim_move(pos, dir)
        if is_max:
            maxes = [
                self.expectimax(pos, 'W', False, depth - 1),
                self.expectimax(pos, 'A', False, depth - 1),
                self.expectimax(pos, 'S', False, depth - 1),
                self.expectimax(pos, 'D', False, depth - 1)
            ]
            # print(maxes)
            m = maxes[0]
            for i in range(1, 4):
                if maxes[i][0] > m[0]:
                    m = maxes[i]
            return m
        else:
            p = np.random.random(4)
            p /= p.sum()
            return [(
                p[0] * self.expectimax(pos, 'W', True, depth - 1)[0] 
                + p[1] * self.expectimax(pos, 'A', True, depth - 1)[0] 
                + p[2] * self.expectimax(pos, 'S', True, depth - 1)[0] 
                + p[3] * self.expectimax(pos, 'D', True, depth - 1)[0]
            ) / 4, dir]

p1 = None

seek_food_range = 15
low_hunger_thresh = 20

reroute_food_range = 5
high_hunger_thresh = 30

seek_pot_range = 15
low_stam_thresh = 25

reroute_pot_range = 5
high_stam_thresh = 40

cvw_m_dist = 10
cvw_o_dist = 5
cvw_surround = 5

cvw_food_range = 10
cvw_food_near = 30

cvw_pot_range = 10
cvw_pot_near = 30

def player1_logic(coins, potions, foods, dungeon_map, self_position, other_agent_position):
    global p1

    if p1 is None:
        p1 = Player1(self_position, other_agent_position, verbosity = 1)
    p1.new_turn(dungeon_map, coins, potions, foods, self_position, other_agent_position)

    pot_dists = sorted([ (p1.realistic_dist(self_position, pot), tuple(pot)) for pot in p1.pots ], key=itemgetter(0))
    food_dists = sorted([ (p1.realistic_dist(self_position, food), tuple(food)) for food in p1.foods ], key=itemgetter(0))

    next_dir = 'I'
    if (p1.stamina > 10 and p1.hunger > 0) or p1.hunger == 0:
        if p1.has_valid_target():
            if dist(p1.mpos, p1.target_pos) == 0:
                p1.set_target(None, None)
            else:
                next_dir = p1.next_dir_to_target()
        if not p1.has_valid_target():
            if len(p1.foods) > 0 and p1.hunger < low_hunger_thresh and food_dists[0][0] < seek_food_range:
                t = p1.set_target(food_dists[0][1], 'food')
                if not t:
                    p1.set_target(None, None)
                else:
                    next_dir = p1.next_dir_to_target()
            elif len(p1.pots) > 0 and p1.stamina < low_stam_thresh and pot_dists[0][0] < seek_pot_range:
                t = p1.set_target(pot_dists[0][1], 'pot')
                if not t:
                    p1.set_target(None, None)
                else:
                    next_dir = p1.next_dir_to_target()
            elif len(p1.coins) > 0:
                em_move = p1.expectimax(pos=p1.mpos, depth=7)
                if p1.verbosity > 0:
                    print('Expectimax Move:', em_move)
                next_dir = em_move[1]
    return p1.move(next_dir)