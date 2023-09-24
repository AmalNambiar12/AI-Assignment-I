import numpy as np
import matplotlib.pyplot as plt
  

class Bandit:
    def __init__(self, arms):
        # Q is the array containing mean values 
        # for each of the actions
        # of the multi-armed bandit.
        self.arms = arms
        self.Q = np.random.normal(0.0, 1.0, arms)

    def choose_action(self, num):
        if num > self.arms or num < 1:
            return 0
        x = np.random.normal(self.Q[num-1], 1.0)
        return x
    
    def get_random_action(self):
        num = np.random.randint(0, self.arms)
        return num+1
    
    def get_optimal_action(self):
        max = np.amax(self.Q)
        index = np.where(self.Q == max)
        return index[0][0] + 1
    
    def get_optimal_reward(self):
        return np.amax(self.Q)
   
class MRP:
    def __init__(self, num):
        # Here, num is the number of non-terminal states
        self.state_cnt = num
        self.state = (num // 2) + 1
        self.terminated = False
    
    def get_state(self):
        return self.state
    
    def is_terminated(self):
        return self.terminated
    
    def move(self):
        # No action is needed in an MRP. Returns the reward.
        if self.terminated:
            return 0
        x = np.random.randint(0, 2)
        x = 2*x - 1
        self.state += x
        if self.state == (self.state_cnt + 1):
            self.terminated = True
            return 1
        else:
            if self.state == 0:
                self.terminated = True
            return 0
        

class BanditBot:
    def __init__(self, arms, type, e=0.1, alpha=0.1, Q1=5.0, c=2.0):
        self.t = 0
        self.Q = np.zeros(arms)
        self.N = np.zeros(arms)
        self.extra_term = np.zeros(arms)
        self.type = type
        self.e = e
        if type == 2: # optimistic initial-values   
            self.alpha = alpha
            self.Q1 = Q1
        elif type == 3: # upper confidence bound
            self.c = c

    
    def argmax(self):
        if self.type == 3:
            arr = self.Q + self.extra_term
            max = np.amax(arr)
            index = np.where(arr == max)
            return index[0][0]+1
        max = np.amax(self.Q)
        index = np.where(self.Q == max)
        return np.random.choice(index[0]) + 1
    
    def updateN(self, index):
        index -= 1
        self.N[index] += 1
        if self.type == 3:
            self.extra_term[index] = self.c * np.sqrt(np.log(self.t)/self.N[index])                
    
    def updateQ(self, index, reward):
        index -= 1
        if self.type == 1 or self.type == 3:
            N = self.N[index]
            self.Q[index] += (reward - self.Q[index])/N
        elif self.type == 2:
            self.Q[index] += (reward - self.Q[index])*self.alpha


    def step(self, bandit, graph=None):
        self.t += 1
        choice = np.random.uniform(0, 1)
        if choice <= self.e:
            x = bandit.get_random_action()
        else:
            x = self.argmax()
        reward = bandit.choose_action(x)
        self.updateN(x)
        self.updateQ(x, reward)
        if not (graph == None):
            graph.add_point(x, reward) 

    def trial(self, bandit, n, graph=None):
        for i in range(n):
            self.step(bandit, graph)        

class MRPBot:
    def __init__(self, num, alpha=0.1):
        self.V = np.random.normal(0.0, 1.0, num+2)
        #self.V = np.empty(num+2)
        self.V.fill(0.5)
        self.V[0] = 0
        self.V[num+1] = 0
        self.S = 0 
        self.alpha = alpha
        self.reward = 0

    def walk(self, mrp):
        states = ["T", "A", "B", "C", "D", "E", "T"]
        self.S = mrp.get_state()
        while (not mrp.is_terminated()):
            s = mrp.get_state()
            r = mrp.move()
            self.S = mrp.get_state()
            self.reward += r
            self.V[s] += (r + self.V[self.S] - self.V[s])*self.alpha
            #print(states[s], " ", r, end=" ")

class BanditGraph:
    def __init__(self, size=1, opti=1, opti_r=0):
        self.X = np.arange(1, size+1)
        self.Y = np.empty(size)
        self.N = 0
        self.o_action = opti
        self.o_reward = opti_r
        self.O = np.empty(size)

    def add_point(self, a, r):
        if self.N > 0:
            self.O[self.N] = self.O[self.N-1] + int(self.o_action==a)
            self.Y[self.N] = self.Y[self.N-1] + r
        else:
            self.O[self.N] = int(self.o_action==a)
            self.Y[self.N] = r
        self.N += 1

    def draw_graph(self, graph_type=1, type=1, e=0.1, alpha=0.1, Q1=5.0, c=2.0):
        # Type 1 is average reward graph
        # Type 2 is % optimal action graph  
        plt.title(get_title(type, e, alpha, Q1, c))  
        if graph_type == 1:    
            plt.plot(self.X, np.divide(self.Y, self.X))
            plt.axhline(y = self.o_reward, color = 'r', linestyle = 'dashed', label = "optimal move")
            plt.ylabel('Average Reward')
        else:
            plt.plot(self.X, np.divide(self.O, self.X)*100)
            plt.ylabel('% Optimal Action')
        plt.xlabel('Steps')
        
        
        #plt.show()

class MRPGraph():
    def __init__(self, size=1, trials=100):
        self.X = np.arange(1, trials+1)
        self.Y = np.empty(trials)
        self.size = size
        self.trials = trials
        self.n = 0
        self.error_data = np.zeros((size, trials))

    def add_error(self, V):
        x = self.size+1
        for i in range(self.size):
            if self.n == 0:
                self.error_data[i][self.n] = (V[i+1] - ((i+1)/x))**2
            self.error_data[i][self.n] = self.error_data[i][self.n-1] + (V[i+1] - ((i+1)/x))**2
        self.n += 1
    
    def draw_graph(self, alpha):
        #print(self.error_data)
        #self.Y = np.average(np.sqrt(np.divide(self.error_data, self.X)), axis=0)
        self.Y = np.divide(np.average(np.sqrt(self.error_data), axis=0), np.sqrt(self.X))
        plt.plot(self.X, self.Y)
        plt.title('Performance (MRP, alpha={})'.format(alpha))
        plt.show()


def bandit_test():
    bandit = Bandit(10)
    #m = MRP()

    opti = bandit.get_optimal_action()
    opti_r = bandit.get_optimal_reward()
    print(bandit.Q)
    print()

    N = 40000
    graph_tool = BanditGraph(N, opti, opti_r)

    bot = BanditBot(10, 1, 0.1)
    #bot = BanditBot(10, 2, 0.0)
    bot.trial(bandit, N, graph_tool)

    graph_tool.draw_graph()

def graph_test():
    g = BanditGraph(10000)
    g.draw_graph()

def MRP_test():
    m = MRP(5)
    mbot = MRPBot(5, 0.15)
    N = 40000

    g = MRPGraph(5, N)
    
    for i in range(N):
        mbot.walk(m)
        m.__init__(5)
        g.add_error(mbot.V)
    g.draw_graph()

def bandit_program():
    arms = 10
    print("\nA ",arms,"-armed bandit has been set for RL-BOT to interact with.", sep="")
    type, e, alpha, Qinit, c = 1, 0.1, 0.1, 5.0, 2.0
    print("Would you like to set the RL-BOT's parameters yourself?")
    choice = input("Your choice (Y/N)>> ").upper()
    if choice == "Y":
        print("Choose from the following:")
        print("1) e-greedy algorithm")
        print("2) Optimistic Initial Values algorithm")
        print("3) Upper-Confidence-Bound algorithm")
        type = int(input("Your choice (1/2/3)>> "))
        if type == 1:
            e = float(input("Enter a value for 'e' between 0 and 1: "))
        elif type == 2:
            e = 0.0
            alpha = float(input("Enter a value for 'alpha': "))
            Qinit = float(input("Enter a value for 'Q1': "))
        elif type == 3:
            e = float(input("Enter a value for 'e' between 0 and 1: "))
            c = float(input("Enter a value for 'c': "))
        else: quit()
    elif choice == "N":
        print("Default parameters selected.")
    else: quit()

    bot = BanditBot(arms, type, e, alpha, Qinit, c)

    if type == 1:
        print("RL-BOT will use e-greedy algorithm (e=",e,")", sep="")
    elif type == 2:
        print("RL-BOT will use Optimistic Initial Values algorithm (alpha=",alpha,", Q1=",Qinit,")", sep="")
    elif type == 3:
        print("RL-BOT will use Upper-Confidence-Bound algorithm (e=",e," c=",c,")", sep="")
    N = int(input("\nEnter the number of steps you want RL-BOT to take: "))
    
    bandit = Bandit(arms)

    opti = bandit.get_optimal_action()
    opti_r = bandit.get_optimal_reward()

    graph_tool = BanditGraph(N, opti, opti_r)

    bot.trial(bandit, N, graph_tool)

    print("\nThe RL-BOT has finished ", N, "steps. See the results?")
    while True:        
        choice = input("Your choice (Y/N)>> ").upper()
        if choice == "Y":
            graph_type = input("Which graph? (1 for Avg. Reward, 2 for % Optimal action): ")
            if not graph_type in ["1", "2"]:
                quit()
            plt.clf()
            graph_type = int(graph_type)
            graph_tool.draw_graph(graph_type, type, e, alpha, Qinit, c)
            print("Original testbed:\n",bandit.Q)
            print("Final value estimation:\n",bot.Q)
            plt.show()
            print("See more results?")
        else:
            break
    
def MRP_program():
    print("\nA Markov Reward Process has been set for RL-BOT to interact with.", sep="")
    alpha = 0.1
    print("Would you like to set the RL-BOT's parameters yourself?")
    choice = input("Your choice (Y/N)>> ").upper()
    if choice == "Y":
        alpha = float(input("Enter a value for 'alpha': "))
    elif choice == "N":
        print("Default parameters selected.")
    else: quit()
    N = int(input("\nEnter the number of steps you want RL-BOT to take: "))

    mrp = MRP(5)
    mrp_bot = MRPBot(5, 0.15)

    graph_tool = MRPGraph(5, N)
    
    for i in range(N):
        mrp_bot.walk(mrp)
        mrp.__init__(5)
        graph_tool.add_error(mrp_bot.V)
    
    print("\nThe RL-BOT has finished ", N, "steps. See the results?")
    while True:        
        choice = input("Your choice (Y/N)>> ").upper()
        if choice == "Y":
            plt.clf()
            graph_tool.draw_graph(alpha)
            plt.show()
            print("See more results?")
        else:
            break

def get_title(type=1, e=0.1, alpha=0.1, Q1=5.0, c=2.0):
    if type == 1:
        return "Performance (e-greedy, e={})".format(e)
    elif type == 2:
        return "Performance (OIV, alpha={}, Q1={})".format(alpha, Q1)
    else:
        return "Performance (UCB, e={}, c={})".format(e, c)
    
def main():
    while True:
        print("--- RL-BOT ---")   
        print("CS F407 Assignment I Submission")
        print("Made by Amal Nambiar") 
        print("Any invalid input will automatically quit the program.\n")
        print("Which project would you like to try?")
        print("1) Multi-Armed Bandit")
        print("2) Markov Reward Process")
        choice = input("Your choice (1/2)>> ").strip()
        if choice == "1":
            bandit_program()
        elif choice == "2":
            MRP_program()
        else:
            quit()
        print()  

if __name__=='__main__':
    main()