# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        Foodlist=successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        mdistances=map(lambda  x:manhattanDistance(newPos, x),Foodlist)
        gdistances=map(lambda x:manhattanDistance(newPos, x.configuration.pos),newGhostStates)
        no_caps=successorGameState.getCapsules()
        if len(mdistances)==0:return successorGameState.getScore()**5
        "*** YOUR CODE HERE ***"
        if len(Foodlist)==0:successorGameState.getScore()**5
        score=(20*(len(Foodlist))**2+min(mdistances))
        score+=1
        for i in range(len(newScaredTimes)):
            if newScaredTimes[i]>0:
                if gdistances[i]==0:
                    score*=0.2
            if newScaredTimes[i]==0:
                if gdistances[i]==0:
                    return -float('Inf')

        scor1=35*len(no_caps)
        if action=='Stop':return -float('Inf')
        return ((successorGameState.getScore())**3/score)-scor1
def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '4'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        def minimize(agent, gamestate, depth):
            if gamestate.isWin() or gamestate.isLose() or depth==self.depth:
                return self.evaluationFunction(gamestate)
            Value=float('Inf')
            Legalactions = gamestate.getLegalActions(agent)
            for action in Legalactions:
                Value=min(Value, minimax(agent, gamestate.generateSuccessor(agent, action), depth))
            return  Value

        def maximize(agent, gamestate, depth):
            if gamestate.isWin() or gamestate.isLose() or depth==self.depth:
                return self.evaluationFunction(gamestate)
            Value = -float('Inf')
            Legalactions = gamestate.getLegalActions(agent)
            for action in Legalactions:
                Value = max(Value, minimax(agent, gamestate.generateSuccessor(agent, action),depth))
            return Value

        def minimax(agent, gamestate,depth):
            nagent=(agent+1)%gamestate.getNumAgents()
            if nagent==0:
                Value=maximize(nagent, gamestate, depth+1)
            else:
                Value=minimize(nagent, gamestate, depth)
            return Value
        legalMoves = gameState.getLegalActions()
        v=-float('Inf')
        bestmove=''
        for a in legalMoves:
            s=gameState.generateSuccessor(0, a)
            B=max(v,minimax(0, s, 0))
            if B>v:
                v=B
                bestmove=a
        return bestmove

        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        def minimize(agent, gamestate, alpha, beta, depth):
            if gamestate.isWin() or gamestate.isLose() or depth==self.depth:
                return self.evaluationFunction(gamestate)
            Value=float('Inf')
            Legalactions = gamestate.getLegalActions(agent)
            for action in Legalactions:
                Value=min(Value, minimax(agent, gamestate.generateSuccessor(agent, action), alpha, beta,depth))
                if alpha>=Value:
                    return Value
                beta= min(beta, Value)
            return  Value

        def maximize(agent, gamestate, alpha, beta, depth):
            if gamestate.isWin() or gamestate.isLose() or depth==self.depth:
                return self.evaluationFunction(gamestate)
            Value = -float('Inf')
            Legalactions = gamestate.getLegalActions(agent)
            for action in Legalactions:
                Value = max(Value, minimax(agent, gamestate.generateSuccessor(agent, action), alpha, beta, depth))
                if beta>=Value:
                    return Value
                alpha=max(Value, alpha)
            return Value

        def minimax(agent, gamestate, alpha, beta, depth):
            nagent=(agent+1)%gamestate.getNumAgents()
            if nagent==0:
                Value=maximize(nagent, gamestate, alpha, beta,depth+1)
            else:
                Value=minimize(nagent, gamestate, alpha, beta, depth)
            return Value
        legalMoves = gameState.getLegalActions()
        v=-float('Inf')
        alpha = -(float("Inf"))
        beta = float("Inf")
        bestmove=''
        for a in legalMoves:
            s=gameState.generateSuccessor(0, a)
            B=max(v,minimax(0, s, alpha, beta, 0))
            if B>v:
                v=B
                bestmove=a
            if B >=beta:
                return bestmove
            alpha = max(alpha, B)
        return bestmove


        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def minimize(agent, gamestate, depth):
            if gamestate.isWin() or gamestate.isLose() or depth==self.depth:
                return self.evaluationFunction(gamestate)
            Value = 0
            Legalactions = gamestate.getLegalActions(agent)
            for action in Legalactions:
                Value+= minimax(agent, gamestate.generateSuccessor(agent, action), depth)*(1./len(Legalactions))
            return  Value

        def maximize(agent, gamestate, depth):
            if gamestate.isWin() or gamestate.isLose() or depth==self.depth:
                return self.evaluationFunction(gamestate)
            Value = -float('Inf')
            Legalactions = gamestate.getLegalActions(agent)
            for action in Legalactions:
                Value = max(Value, minimax(agent, gamestate.generateSuccessor(agent, action),depth))
            return Value

        def minimax(agent, gamestate,depth):
            nagent=(agent+1)%gamestate.getNumAgents()
            if nagent==0:
                Value=maximize(nagent, gamestate, depth+1)
            else:
                Value=minimize(nagent, gamestate, depth)
            return Value
        legalMoves = gameState.getLegalActions()
        v=-float('Inf')
        bestmove=''
        for a in legalMoves:
            s=gameState.generateSuccessor(0, a)
            B=max(v,minimax(0, s, 0))
            if B>v:
                v=B
                bestmove=a
        return bestmove

        util.raiseNotDefined()
SavedMaps={}
SavedMaps['start']=0
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      I considered the true path of the pacman to the nearrest food to give an importance and more prority to the number
      of dots present,I didnt see a reason to remove the current score as it looks perefect and the number of capsules present were considered
      also as eating the ghosts gave a good score
    """
    "*** YOUR CODE HERE ***"
    def Transeverse(point):
        Walls=currentGameState.getWalls()
        maxx,maxy=Walls.asList()[-1]
        Map=[[999 for i in range(maxy+1)]for i in range(maxx+1)]
        frointer=util.Queue()
        Map[point[0]][point[1]]=0
        frointer.push((point,1))
        while not(frointer.isEmpty()):
            (x,y),c=frointer.pop()
            for dx,dy in [(0,1), (1,0), (-1,0), (0,-1)]:
                nextx,nexty=x+dx,y+dy
                if -1<nextx<maxx and -1<nexty<maxy:
                    if Map[nextx][nexty]==999 and Walls[nextx][nexty]==False:
                        Map[nextx][nexty]=c
                        frointer.push(((nextx, nexty), c+1))
        return Map
    global SavedMaps

    Foodnow = currentGameState.getFood()
    Foodlist = Foodnow.asList()
    if SavedMaps['start']==0:



        for i in Foodlist:
            SavedMaps[i]=Transeverse(i)
        SavedMaps['start'] = 1
    SavedMaps['start'] += 1
    LargeNumb = 999999999999999
    Current_score = currentGameState.getScore()

    if currentGameState.isWin() :
        return Current_score**7
    if currentGameState.isLose():
        return -LargeNumb
    Pos=currentGameState.getPacmanPosition()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    minimum_distance = LargeNumb
    for x in Foodlist:
        try:
            minimum_distance = min(minimum_distance,SavedMaps[x][Pos[0]][Pos[1]])
        except KeyError:
            minimum_distance = minimum_distance
    if len(Foodlist) == 0: minimum_distance=1./20
    gdistances = map(lambda x: manhattanDistance(Pos, x.configuration.pos), GhostStates)
    Number_dots = len(Foodlist)
    if Number_dots == 0: return Current_score**7
    proxmimityscore = 0
    scaredscore=0
    Current_score = currentGameState.getScore()
    Pos_capsules = currentGameState.getCapsules()
    for i,s in enumerate(ScaredTimes):
        if s==0:
            if gdistances[i]<5:
                if gdistances[i]==0:
                    return -LargeNumb
                proxmimityscore+= 100./gdistances[i]
        else:
            if gdistances[i]==0:
                scaredscore+=2500
            else:
                scaredscore+=(100./(gdistances[i]**2))
    score = Current_score**5+2*1./minimum_distance

    return (score)-35*(len(Pos_capsules))-proxmimityscore+scaredscore

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

