package de.raimannma.reinforce4j;

import net.jafama.FastMath;

import java.util.ArrayList;
import java.util.Arrays;

public class DQN {
    private final int numStates;
    private final int numActions;
    private final double gamma;
    private final double epsilon;
    private final double alpha;
    private final int experienceAddEvery;
    private final int experienceSize;
    private final int learningStepsPerIteration;
    private final double tdErrorClamp;
    private final int numHiddenUnits;
    private Mat W1;
    private Mat B1;
    private Mat W2;
    private Mat B2;
    private ArrayList<Experience> experience;
    private int experienceIndex;
    private int t;
    private double lastReward;
    private Mat lastState;
    private Mat currentState;
    private int lastAction;
    private int currentAction;
    private Graph lastG;
    private boolean isFirstRun;

    public DQN(final int numActions, final int numStates, final Configuration config) {
        this.numActions = numActions;
        this.numStates = numStates;

        this.gamma = config.getOrDefault(Option.GAMMA, 0.75);
        this.epsilon = config.getOrDefault(Option.EPSILON, 0.1);
        this.alpha = config.getOrDefault(Option.ALPHA, 0.01);

        this.experienceAddEvery = DQN.toInteger(config.getOrDefault(Option.EXPERIENCE_ADD_EVERY, 25.0));
        this.experienceSize = DQN.toInteger(config.getOrDefault(Option.EXPERIENCE_SIZE, 5000.0));
        this.learningStepsPerIteration = DQN.toInteger(config.getOrDefault(Option.LEARNING_STEPS_PER_ITERATION, 10.0));
        this.tdErrorClamp = config.getOrDefault(Option.TD_ERROR_CLAMP, 1.0);
        this.numHiddenUnits = DQN.toInteger(config.getOrDefault(Option.NUM_HIDDEN_UNITS, 100.0));

        this.reset();
    }

    private static Mat createRandMat(final int n, final int d) {
        final Mat mat = new Mat(n, d);
        Arrays.setAll(mat.w, i -> Utils.randN(0, 0.01));
        return mat;
    }

    private static int toInteger(final Double val) {
        return (int) FastMath.round(val);
    }

    private static int maxIndex(final double[] arr) {
        int maxIndex = 0;
        double maxVal = arr[0];
        for (int i = 1; i < arr.length; i++) {
            if (arr[i] > maxVal) {
                maxIndex = i;
                maxVal = arr[i];
            }
        }
        return maxIndex;
    }

    private void reset() {
        this.W1 = DQN.createRandMat(this.numHiddenUnits, this.numStates);
        this.B1 = new Mat(this.numHiddenUnits, 1);
        this.W2 = DQN.createRandMat(this.numActions, this.numHiddenUnits);
        this.B2 = new Mat(this.numActions, 1);

        this.experience = new ArrayList<>();
        this.experienceIndex = 0;

        this.t = 0;

        this.lastReward = 0;
        this.lastState = null;
        this.currentState = null;
        this.lastAction = 0;
        this.currentAction = 0;
        this.isFirstRun = true;
    }

    private Mat forwardQ(final Mat state, final boolean needsBackprop) {
        final Graph graph = new Graph(needsBackprop);

        final Mat a1Mat = graph.add(graph.mul(this.W1, state), this.B1);
        final Mat h1Mat = graph.tanh(a1Mat);
        final Mat a2Mat = graph.add(graph.mul(this.W2, h1Mat), this.B2);
        this.lastG = graph;
        return a2Mat;
    }

    public int act(final double[] stateArr) {
        final Mat state = new Mat(this.numStates, 1);
        state.setFrom(stateArr);

        final int action;
        if (FastMath.random() < this.epsilon) {
            action = Utils.randI(this.numActions);
        } else {
            action = DQN.maxIndex(this.forwardQ(state, false).w);
        }
        this.lastState = this.currentState;
        this.lastAction = this.currentAction;
        this.currentState = state;
        this.currentAction = action;
        return action;
    }

    public void learn(final double reward) {
        if (!this.isFirstRun && this.alpha > 0) {
            this.learnFromTuple(new Experience(this.lastState, this.lastAction, this.lastReward, this.currentState));
            if (this.t % this.experienceAddEvery == 0) {
                if (this.experience.size() > this.experienceIndex) {
                    this.experience.set(this.experienceIndex, new Experience(this.lastState, this.lastAction, this.lastReward, this.currentState));
                } else {
                    this.experience.add(this.experienceIndex, new Experience(this.lastState, this.lastAction, this.lastReward, this.currentState));
                }
                this.experienceIndex++;
                if (this.experienceIndex > this.experienceSize) {
                    this.experienceIndex = 0;
                }
            }
            this.t++;

            for (int i = 0; i < this.learningStepsPerIteration; i++) {
                final int rand = Utils.randI(this.experience.size());
                this.learnFromTuple(this.experience.get(rand));
            }
        } else {
            this.isFirstRun = false;
        }
        this.lastReward = reward;
    }

    private void learnFromTuple(final Experience exp) {
        final Mat tMat = this.forwardQ(exp.getCurrentState(), false);
        final double qMax = exp.getLastReward() + this.gamma * Arrays.stream(tMat.w).max().orElseThrow();

        final Mat pred = this.forwardQ(exp.getLastState(), true);
        double tdError = pred.w[exp.getLastAction()] - qMax;
        if (FastMath.abs(tdError) > this.tdErrorClamp) {
            tdError = tdError > this.tdErrorClamp ?
                    this.tdErrorClamp :
                    -this.tdErrorClamp;
        }
        pred.dw[exp.getLastAction()] = tdError;
        this.lastG.backward();

        this.W1.update(this.alpha);
        this.W2.update(this.alpha);
        this.B1.update(this.alpha);
        this.B2.update(this.alpha);
    }
}
