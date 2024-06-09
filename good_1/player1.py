from operator import itemgetter
import numpy as np
import heapq

class PrioritySet(object):
    def __init__(self):
        self.heap = []
        self.set = set()

    def push(self, pri, d):
        if not d in self.set:
            heapq.heappush(self.heap, (pri, d))
            self.set.add(d)

    def pop(self):
        pri, d = heapq.heappop(self.heap)
        self.set.remove(d)
        return d
    
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

        if self.verbosity > 0:
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
        if self.verbosity > 0:
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
        while curr_nm['parent'] != (-1, -1):
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
            curr = open.pop()
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
                        g_new = curr_nm['g'] + 1.0
                        h_new = dist(succ, goal)
                        f_new = g_new + h_new
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
        self.curr_path = self.a_star_path(self.mpos, self.target_pos)
        self.t = 0

    def has_valid_target(self):
        return (self.target_pos is not None) and (self.read_map(self.target_pos) == self.target_type)
    
    def next_dir_to_target(self):
        dir = self.curr_path[self.t]
        p = sim_move(self.mpos, dir)
        if dist(p, self.opos) <= 1:
            return 'I'
        self.t += 1
        return dir
    
directions = { 'W': (0, -1), 'A': (-1, 0), 'S': (0, 1), 'D': (1, 0) }
reverse_directions = dict((v, k) for k, v in directions.items())

def sim_move(pos, dir):
    return (pos[0] + directions[dir][0], pos[1] + directions[dir][1])

def dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

p1 = None

hunger_threshold = 25
food_range = 25
stam_threshold = 25
pot_range = 25

def player1_logic(coins, potions, foods, dungeon_map, self_position, other_agent_position):
    global p1

    if p1 is None:
        p1 = Player1(self_position, other_agent_position, verbosity = 0)
    p1.new_turn(dungeon_map, coins, potions, foods, self_position, other_agent_position)

    next_dir = 'I'
    if (p1.stamina > 10 and p1.hunger > 0) or p1.hunger == 0:
        if p1.has_valid_target():
            if dist(p1.mpos, p1.target_pos) == 0:
                p1.set_target(None, None)
            else:
                next_dir = p1.next_dir_to_target()
        if not p1.has_valid_target():
            coin_dists = sorted([ (dist(self_position, coin), tuple(coin)) for coin in p1.coins ], key=itemgetter(0))
            pot_dists = sorted([ (dist(self_position, pot), tuple(pot)) for pot in p1.pots ], key=itemgetter(0))
            food_dists = sorted([ (dist(self_position, food), tuple(food)) for food in p1.foods ], key=itemgetter(0))
            if p1.hunger < hunger_threshold and len(food_dists) > 0 and food_dists[0][0] < food_range:
                p1.set_target(food_dists[0][1], 'food')
                next_dir = p1.next_dir_to_target()
            elif p1.stamina < stam_threshold and len(pot_dists) > 0 and pot_dists[0][0] < pot_range:
                p1.set_target(pot_dists[0][1], 'pot')
                next_dir = p1.next_dir_to_target()
            elif len(coin_dists) > 0:
                p1.set_target(coin_dists[0][1], 'coin')
                next_dir = p1.next_dir_to_target()
    return p1.move(next_dir)

# Idea: gradient descent on a heatmap of the dungeon map, where the hottest zones are the coin
# temperature decreases with distance