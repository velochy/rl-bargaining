import gym
from gym import Env, spaces
import numpy as np

class Bargaining(Env):
    metadata = {'render.modes': ['human']}
    reward_range = (-np.inf, np.inf)
    action_space = spaces.Box(low=0.0, high=2.0, shape=(1,), dtype=np.float16) # Bid
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float16) # [ Step num, Opponent Bid, Own price  ]

    discount = 0.99 
    no_deal_rate = 0.05 # Proportion of cases where no deal exists
    max_steps = 16 # Max steps (both players together)
    time_cost = 0.0 #0.005

    def __init__(self):
        pass

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        self.prices = [np.random.uniform(0,1), np.random.uniform(0,1)]
        if self.prices[0]>self.prices[1]: # Make sure they are in right order
            self.prices = [self.prices[1],self.prices[0]]

        if np.random.uniform(0,1)<self.no_deal_rate: # Reverse if we want a no-deal condition
            self.prices = [self.prices[1],self.prices[0]]

        self.prices[0] = 1.0 - self.prices[0]

        self.bids = [0.0, 0.0]
        self.cur_player = 0 # 0 for seller, 1 for buyer
        self.steps = 0
        self.rewards = [0.0, 0.0]
        self.ended, self.done = False, False
        self.info = {}
        
        return [self.steps, self.bids[1-self.cur_player], self.prices[self.cur_player]]

    def step(self, actions):
        """Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (observation, reward, done, info).
        # Arguments
            action (object): An action provided by the environment.
        # Returns
            observation (object): Agent's observation of the current environment.
            reward (float) : Amount of reward returned after previous action.
            done (boolean): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """

        action = actions[0]

        #print "\nAction: %.4f" % action

        # If ended, inform the other party of the result and call it done
        if self.ended or self.done: 
            self.done = True

        # Stopping before a deal was reached
        elif (action > self.prices[self.cur_player] or # Unprofitable bid
            self.steps > self.max_steps): # Ran out of bids

            cost = self.time_cost * self.steps
            self.rewards = [-cost,-cost] # As time was wasted by both

            self.ended = True

        # Otherwise, update bid and check if we have a deal
        else:
            self.bids[self.cur_player] = action
            if 1.0-self.bids[0]<=self.bids[1]: # We have a deal

                # Current bid is just the "accept" signal, so actual price is the one before
                sale_price = self.bids[1] if self.cur_player == 0 else 1.0-self.bids[0] 

                surplus = abs(self.prices[1]-(1.0-self.prices[0]))

                # Discount the final value by the time spent and scale with max possible score
                dmult = (self.discount**self.steps)/surplus

                # Set rewards equal to surplus beyond reservation price for both parties
                self.rewards = [dmult*( sale_price - (1.0-self.prices[0]) ), 
                                dmult*( self.prices[1] - sale_price )]

                #print "   Step: %i, split %.0f/%.0f" % (self.steps, 100*(sale_price - self.prices[0])/surplus, 100*(self.prices[1] - sale_price)/surplus)

                self.ended = True

        self.steps += 1

        resp = [ [self.steps, self.bids[self.cur_player], self.prices[1-self.cur_player]], # Observation
                    self.rewards[self.cur_player], self.done, self.info ]

        self.cur_player = 1-self.cur_player

        return resp

    def render(self, mode='human', close=False):
        """Renders the environment.
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.)
        # Arguments
            mode (str): The mode to render with.
            close (bool): Close all open renderings.
        """
        #return
        if self.done: return

        if self.ended:
            if 1.0-self.bids[0]<self.bids[1]: 
                print "Deal at %.3f, rewards %.3f-%.3f" % (self.bids[self.cur_player-1], self.rewards[0], self.rewards[1])
            else:
                print "No deal!"
            return

        print "Step: %i Gap: %.3f" % (self.steps, max((1.0-self.bids[0])-self.bids[1],0.0))
        print "Seller: bid %.3f surplus %.3f" % (1.0-self.bids[0],self.prices[0]-self.bids[0])
        print "Buyer:  bid %.3f surplus %.3f" % (self.bids[1],self.prices[1]-self.bids[1])
        

    def close(self):
        """Override in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        # Returns
            Returns the list of seeds used in this env's random number generators
        """
        np.random.seed(seed)


if __name__ == '__main__':
    env = Bargaining()

    obs = env.reset()
    print "Seller price: %.3f, Buyer price: %.3f" % (1.0-env.prices[0], env.prices[1])

    while True:
        bid = input('Bid for %s: ' % ('Seller' if env.cur_player == 0 else 'Buyer'))
        if env.cur_player==0: bid = 1.0-bid
        obs, reward, done, info = env.step([bid])
        if reward: 
            print "Got reward %.3f" % reward
            _, reward, _, _ = env.step([0])
            print "Other player reward %.3f" % reward
            break
        elif env.ended:
            print "No deal!"
            break 