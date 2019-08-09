The end goal of these experiments is to try to learn a superhuman level AI for a bargaining game with incomplete information.

**Bargaining game**
The game models bargaining between two parties (Seller and Buyer), based on [Rubenstein Bargaining](https://en.wikipedia.org/wiki/Rubinstein_bargaining_model), but with incomplete information and finite number of steps. 

Both players have a "reservation price", where a seller benefits from selling at any price above his, and buyer from any price below his.
Players know their own reservation price, but not that of the other player. Both are drawn from a uniform distribution between 0 and 1.

The game starts with the Seller making a bid, and then has players making bids in turn until either
 - A deal is reached, i.e. the bid of buyer is above that of the seller
 - One of the sides quits the game by making an unprofitable bid
 - The game runs longer than N steps

 The reward for a player is the percentage of surplus (difference between the reservation prices) he gets in the bidding, so a 50-50 split nets 0.5 reward to both.
 The reward is then further discounted by discount rate to the power of the number of steps, to dis-incentivize the game going on in perpetuity.

 To facilitate self-play, the game is symmetrized by having the seller price and bid be inverted so they are both y=1.0-x. In this setting, bids for both should increase, and for both, the quitting step is making a bid above their reservation price.

**Project layout:**
 - bargaining.py - logic of the game environment (in the OpenAI Gym format). Can be called directly to play manually
 - interleaved.py - code for an "agent" that actually plays two agents interleaved with one another
 - learn.py - code for actual learning

 **Current state:**
DDPG learns to play the game, but not particularily well. It reaches a deal only 75% of the time and usually by just accepting the first or second bid.

**Ideas for future**
Try to apply more game-specific algorithms like AlphaZero (i.e. MCTS + DL) or Counterfactual Regret Minimization. The main hurdle with both is to figure out how to handle the continuous action space. 