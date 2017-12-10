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



