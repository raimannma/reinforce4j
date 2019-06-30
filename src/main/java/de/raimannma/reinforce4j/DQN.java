package de.raimannma.reinforce4j;

import com.google.gson.Gson;
import net.jafama.FastMath;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class DQN {
    private static final Random rand = new Random();
    private final int numStates;
    private final int numActions;
    private final double gamma;
    private final int experienceAddEvery;
    private final int experienceSize;
    private final int learningStepsPerIteration;
    private final double tdErrorClamp;
    private final int numHiddenUnits;
    private final int saveInterval;
    private final double alpha;
    private final double epsilon;
    private final Mat W1;
    private final Mat B1;
    private final Mat W2;
    private final Mat B2;
    private final ArrayList<Experience> experience;
    private int experienceIndex;
    private int t;
    private double lastReward;
    private Mat lastState;
    private Mat currentState;
    private int lastAction;
    private int currentAction;
    private Graph lastG;
    private boolean isFirstRun;

    public DQN(final int numActions, final int numStates, final HashMap<Option, Double> config) {
        this.numActions = numActions;
        this.numStates = numStates;

        this.gamma = config.getOrDefault(Option.GAMMA, 0.75);
        this.epsilon = config.getOrDefault(Option.EPSILON, 0.1);
        this.alpha = config.getOrDefault(Option.ALPHA, 0.05);

        this.experienceAddEvery = DQN.toInteger(config.getOrDefault(Option.EXPERIENCE_ADD_EVERY, 25.0));
        this.experienceSize = DQN.toInteger(config.getOrDefault(Option.EXPERIENCE_SIZE, 5000.0));
        this.learningStepsPerIteration = DQN.toInteger(config.getOrDefault(Option.LEARNING_STEPS_PER_ITERATION, 10.0));
        this.tdErrorClamp = config.getOrDefault(Option.TD_ERROR_CLAMP, 1.0);
        this.numHiddenUnits = DQN.toInteger(config.getOrDefault(Option.NUM_HIDDEN_UNITS, 100.0));

        this.saveInterval = DQN.toInteger(config.getOrDefault(Option.SAVE_INTERVAL, 100.0));

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

    private static Mat createRandMat(final int n, final int d) {
        final Mat mat = new Mat(n, d);
        Arrays.setAll(mat.w, i -> DQN.rand.nextGaussian() / 100);
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

    private Mat calcQ(final Mat state, final boolean needsBackprop) {
        this.lastG = new Graph(needsBackprop);
        return this.lastG.add(this.lastG.mul(this.W2, this.lastG.tanh(this.lastG.add(this.lastG.mul(this.W1, state), this.B1))), this.B2);
    }

    public int act(final double[] stateArr) {
        final Mat state = new Mat(this.numStates, 1, stateArr);

        final int action = FastMath.random() < this.epsilon ?
                DQN.rand.nextInt(this.numActions) :
                DQN.maxIndex(this.calcQ(state, false).w);
        this.lastState = this.currentState;
        this.lastAction = this.currentAction;
        this.currentState = state;
        this.currentAction = action;
        return action;
    }

    public void learn(final double reward) {
        if (this.isFirstRun) {
            this.isFirstRun = false;
            this.lastReward = reward;
            return;
        }

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

        if (this.t % this.saveInterval == 0) {
            this.saveModel();
        }

        IntStream.range(0, this.learningStepsPerIteration)
                .mapToObj(i -> this.experience.get(DQN.rand.nextInt(this.experience.size())))
                .forEach(this::learnFromTuple);
        this.lastReward = reward;
    }

    private void learnFromTuple(final Experience exp) {
        final Mat tMat = this.calcQ(exp.getCurrentState(), false);
        final double qMax = exp.getLastReward() + this.gamma * Arrays.stream(tMat.w).max().orElseThrow();

        final Mat pred = this.calcQ(exp.getLastState(), true);
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

    private void saveModel() {
        final File file = new File("dqnAgentW1.json");
        final File file1 = new File("dqnAgentW2.json");
        final File file2 = new File("dqnAgentB1.json");
        final File file3 = new File("dqnAgentB2.json");
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(file));
            writer.write(new Gson().toJson(this.W1));

            writer = new BufferedWriter(new FileWriter(file1));
            writer.write(new Gson().toJson(this.W2));

            writer = new BufferedWriter(new FileWriter(file2));
            writer.write(new Gson().toJson(this.B1));

            writer = new BufferedWriter(new FileWriter(file3));
            writer.write(new Gson().toJson(this.B2));
        } catch (final IOException e) {
            e.printStackTrace();
        }
    }

    public Mat[] loadModel() {
        final File file = new File("dqnAgentW1.json");
        final File file1 = new File("dqnAgentW2.json");
        final File file2 = new File("dqnAgentB1.json");
        final File file3 = new File("dqnAgentB2.json");
        if (!file.exists() || !file1.exists() || !file2.exists() || !file3.exists()) {
            return null;
        }
        try {
            final Gson gson = new Gson();
            BufferedReader reader = new BufferedReader(new FileReader(file));
            final Mat w1 = gson.fromJson(reader.lines().collect(Collectors.joining()), this.W1.getClass());

            reader = new BufferedReader(new FileReader(file1));
            final Mat w2 = gson.fromJson(reader.lines().collect(Collectors.joining()), this.W2.getClass());

            reader = new BufferedReader(new FileReader(file2));
            final Mat b1 = gson.fromJson(reader.lines().collect(Collectors.joining()), this.B1.getClass());

            reader = new BufferedReader(new FileReader(file3));
            final Mat b2 = gson.fromJson(reader.lines().collect(Collectors.joining()), this.B2.getClass());

            return new Mat[]{w1, w2, b1, b2};
        } catch (final FileNotFoundException e) {
            e.printStackTrace();
        }
        throw new RuntimeException();
    }
}
