# Reinforcement-Learning

## Objectives

### Objective 1 ###
* Read through an [introduction to reinforcement learning](https://www.analyticsvidhya.com/blog/2016/12/getting-ready-for-ai-based-gaming-agents-overview-of-open-source-reinforcement-learning-platforms/) to get an idea of how reinforcement learning works

    * **Comments** <br>
    Was not particularly helpful, gave an extremely simple overview of reinforcement learning. (i.e agent gets an observation from the environment -> agent peforms and action -> may or may not get a reward -> feedback to agent -> cycle restarts)

### Objective 2 ###
* Read through a [beginners guide to reinforcement learning](https://www.analyticsvidhya.com/blog/2017/01/introduction-to-reinforcement-learning-implementation/) to go through some examples of reinforcement learning

    * **Comments** <br>
    Website contained code example of implementing a Deep Q-learning algorithm to train an agent to play cartpole.
    My own implementation in the carpole_deepQlearning.py file. The implementation was straightfoward, however I feel as if some of the important concepts were really explained. For example the article did not really explain how rewards were being given to the agent.
    <br>
    I probably need to take a look at the keras documenation to see exactly what I have implemented.
    <br>
    Overall this article is too high level for me to be able to really learn anything concrete from it. It gave me a very hazy high-level understanding of reinforcement learning. All I did was use a library.

    * **Unanswered Questions/Tasks** <br>
        - [ ] Look at Keras documentation
        - [ ] Take a look at multiagent reinforcement learning. This [paper](https://link.springer.com/chapter/10.1007/978-3-642-27645-3_14) titled: Game theory and multi-agent reinforcement learning may be interesting.
        - [ ] Need to improve fundamental understanding of how reinforcement learning works


    * **Interesting Insights!** <br>
        * Game theory and multiagent reinforcement learning (the idea of multiple agents being trained in the same environment) may be an interesting topic to learn more about.
        * Other areas RL is being applied to
            * Game theory and multi-agent interaction
            * Robotics
            * Computer networks
            * Vehicular navigation
            * Medicine
            * Industrial Logic

### Objective 3 ###
* Work through [Introduction to reinforcement learning](https://github.com/dennybritz/reinforcement-learning)
    * Start reading through [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/bookdraft2017nov5.pdf) to improve basic understanding on reinforcement learning.

        * Notes from reading **Chapter 1: Introduction**
            * Gives an introduction to reinforcement learning, gives motivation on its importance, some basic history.
            * Describes the _policy_, _reward signal_, _value function_, and a _model_ of the environment.
                * _Policy_: learning agents way of behaving at a given time. A mapping of perceived states of the environment to actions to be taken in those states. In general, policies may be stochastic.
                * _Reward signal_: The goal in a reinforcement learning problem. On each time step, the environment sends a single number to the agent. This number is the _reward_. The agents objective is to maximize the total _reward_ it receives over time. The _reward signal_ is the primary basis for changing the _policy_; if an action selected by the policy results in a low reward, then the policy may be changed to select some other action in that situation in the future.
                * _Value_: The _value_ of a state is the total _reward_ an agent can expect to accumulate over the future, starting from that state. Whereas _reward_ immediate, intrinsic desirability of environmental states, _values_ indicate the long-term desirability of states that are likely to follow, and the rewards available in those states.
                * _Model_: Mimics the behavior of the environment, and allows inferences to be made about how the environment will behave. For example, give a state and an action, a model may predict the resultant next state, and next reward. 
            * Describes the differences between _evolutionary methods_, and methods that learn _value functions_.
                * To evaluate a policy an _evolutionary method_ holds the policy fixed and plays many against the opponent. The frequency of wins gives an unbiased estimate of the probability of winning with that policy, and can be used to direct the next policy selection. However, each policy change is only made after many games, and only the final outcome of the game is used. What happens during the game is ignored. E.g. If a player wins then all of its behavior during the game is given credit, independently of how speicific moves might have been critical to the win. 
                * Value functions allow individual states to be measured - learning while interacting with the environment. 
            * Tic-tac-toe example was helpful in understanding how exploitory and exploratory moves differ, and how they affected the policy.
        * Looked at OpenAI Gym tutorial - it goes through the basic syntax, and commands you can call to interface with the environment.

    * Notes from reading **Chapter 3: Finite Markov Decision Process**
        * Mathematically formalizes the general concepts in reinforcement learning
            * agent performs action -> actions influences environment -> repeat
        * Rewards are one number, and should be given based on _what_ goal you want the agent to achieve, not based on _how_ you want the agent to achieve a goal. For example, giving rewards to a chess playing agent for taking more of the opponents pieces with a goal to win the game is bad, because it could find a way to take more of the oppponents pieces while still losing the game.
        * Use a discount rate to value immediate rewards more than future rewards - allows us to model continuous games where the reward could go to infinity if not bounded. 0 <= _reward value_ <= 1. A reward closer to 1 makes future rewards more important. The agent becomes more far-sighted. A reward of zero makes the agent only concered with immediate rewards. i.e it chooses an action so as to only maximize the reward at time _t+1_.
        * _Value functions_ estimate how good it is for the agent to be in a given state.
        * A _policy_ is a mapping from states to probabilities of selecting each possible action.
        * Reinforcement learning methods specify how the agent's policy is changed as a result of its experience.
        * The value of a state _s_ under a policy _pi_, denoted _v<sub>&pi;</sub>(s)_ is the expected return when starting in _s_, and following _pi_ thereafter. We call _v<sub>&pi;</sub>(s)_ the _state-value function for policy <sub>&pi;<sub>.
        * The value of taking action _a_ in state _s_ under a policy _pi_, denoted _q<sub>&pi;</sub>(s,a)_ is the expected return starting from s, taking the action _a_, and thereafter following policy _pi_.
        We call _q<sub>&pi;</sub>(s,a)_ the action-value function for policy <sub>&pi;</sub>.
        * The Bellman equation expresses the relationship between the value of a state and the values of its successor state. Bellman equations exist for both the value function, and the action value function.
        * Optimal policies - a policy whose expected return is greater than or equal to the expected return of all other policies in every state. Could be more than one. All optimal policies share the same _optimal state-value function_, denoted _v<sub>*</sub>_. And the same _optimal action-value function_, denoted _q<sub>*</sub>.
        * The Bellman optimality equation defines how the optimal value of a state is related to the optimal value of successor states.
        * Usually have to use approximation methods and heuristics to represent these equations, otherwise it would be computationally inefficient, and take forever to solve.



