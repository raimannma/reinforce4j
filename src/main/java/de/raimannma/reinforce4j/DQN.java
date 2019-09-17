package de.raimannma.reinforce4j;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import net.jafama.FastMath;

import java.io.*;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class DQN {
    final int numStates;
    final int numActions;
    private final double gamma;
    private final int expAddEvery;
    private final int expSize;
    private final int learningStepsPerIteration;
    private final double tdErrorClamp;
    private final int saveInterval;
    private final double alpha;
    private final double epsilon;
    private final Experience[] exp;
    private final boolean isSaving;
    private final int agentIndex;
    public Mat W1;
    Mat B1;
    Mat W2;
    Mat B2;
    private int expIndex;
    private int t;
    private double lastReward;
    private Mat lastState;
    private Mat currentState;
    private int lastAction;
    private int currentAction;
    private Graph lastG;
    private boolean isFirstRun;

    DQN(final int numActions, final int numStates, final Map<Option, Double> config) {
        this(numActions, numStates, config, 0, null);
    }

    public DQN(final int numActions, final int numStates, final Map<Option, Double> config, final int agentIndex, final Mat[] nets) {
        this.numActions = numActions;
        this.numStates = numStates;
        this.agentIndex = agentIndex;

        this.gamma = config.getOrDefault(Option.GAMMA, 0.3);
        this.epsilon = config.getOrDefault(Option.EPSILON, 0.01);
        this.alpha = config.getOrDefault(Option.ALPHA, 0.05);

        this.expAddEvery = DQN.toInteger(config.getOrDefault(Option.EXPERIENCE_ADD_EVERY, 25.0));
        this.expSize = DQN.toInteger(config.getOrDefault(Option.EXPERIENCE_SIZE, 5000.0));
        this.learningStepsPerIteration = DQN.toInteger(config.getOrDefault(Option.LEARNING_STEPS_PER_ITERATION, 10.0));
        this.tdErrorClamp = config.getOrDefault(Option.TD_ERROR_CLAMP, 1.0);
        final int numHiddenUnits = DQN.toInteger(config.getOrDefault(Option.NUM_HIDDEN_UNITS, 100.0));

        this.saveInterval = DQN.toInteger(config.getOrDefault(Option.SAVE_INTERVAL, 100.0));

        this.isSaving = this.saveInterval != -1;

        if (nets == null) {
            this.W1 = DQN.createRandMat(numHiddenUnits, this.numStates);
            this.W2 = DQN.createRandMat(this.numActions, numHiddenUnits);
            this.B1 = new Mat(numHiddenUnits, 1);
            this.B2 = new Mat(this.numActions, 1);
        } else {
            this.W1 = nets[0];
            this.W2 = nets[1];
            this.B1 = nets[2];
            this.B2 = nets[3];
        }

        this.exp = new Experience[this.expSize];
        this.expIndex = 0;

        this.t = 0;

        this.lastReward = 0;
        this.lastState = null;
        this.currentState = null;
        this.lastAction = 0;
        this.currentAction = 0;
        this.isFirstRun = true;
    }

    static int toInteger(final Double val) {
        return (int) FastMath.round(val);
    }

    static Mat createRandMat(final int n, final int d) {
        final Mat mat = new Mat(n, d);
        Arrays.parallelSetAll(mat.w, i -> ThreadLocalRandom.current().nextGaussian() / 100);
        return mat;
    }

    public int act(final double[] stateArr) {
        final Mat state = new Mat(this.numStates, 1, stateArr);

        final int action = FastMath.random() < this.epsilon ?
                ThreadLocalRandom.current().nextInt(this.numActions) :
                DQN.maxIndex(this.calcQ(state, false).w);
        this.lastState = this.currentState;
        this.lastAction = this.currentAction;
        this.currentState = state;
        this.currentAction = action;
        return action;
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

    public void learn(final double reward) {
        if (this.isFirstRun) {
            this.isFirstRun = false;
            this.lastReward = reward;
            return;
        }

        final Experience exp = new Experience(this.lastState, this.lastAction, this.lastReward, this.currentState);
        this.learnFromTuple(exp);
        this.calcTDError(exp);
        if (this.t % this.expAddEvery == 0) {
            this.exp[this.expIndex] = exp;
            this.expIndex++;
            if (this.expIndex > this.expSize) {
                this.expIndex = 0;
            }
        }
        this.t++;

        if (this.isSaving && this.t % this.saveInterval == 0) {
            this.saveModel();
        }
        final Experience[] sorted = Arrays.stream(this.exp).filter(Objects::nonNull).sorted(Comparator.comparingDouble(Experience::getTdError).reversed()).toArray(Experience[]::new);
        for (int i = 0; i < this.learningStepsPerIteration; i++) {
            this.learnFromTuple(sorted[IntStream.range(0, sorted.length).filter(j -> ThreadLocalRandom.current().nextDouble() < Math.pow(0.5, j + 1)).findFirst().orElseThrow()]);
        }
        this.lastReward = reward;
    }

    private void learnFromTuple(final Experience exp) {
        this.lastG.backward();

        this.W1.update(this.alpha);
        this.W2.update(this.alpha);
        this.B1.update(this.alpha);
        this.B2.update(this.alpha);
    }

    private void calcTDError(final Experience exp) {
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
        exp.setTDError(tdError);
    }

    public void saveModel() {
        this.saveModel(new File("agent.json"));
    }

    public void saveModel(final File file) {
        try {
            if (!file.exists()) {
                file.createNewFile();
            }
            if (file.canWrite()) {
                final BufferedWriter writer = new BufferedWriter(new FileWriter(file), 10 * 1024);

                final JsonObject jsonObject = new JsonObject();
                jsonObject.addProperty("W1", this.W1.toJson());
                jsonObject.addProperty("W2", this.W2.toJson());
                jsonObject.addProperty("B1", this.B1.toJson());
                jsonObject.addProperty("B2", this.B2.toJson());

                writer.write(jsonObject.toString());
                writer.close();
                System.out.println("SAVED");
            } else {
                System.out.println("Can't write on file!");
            }
        } catch (final IOException e) {
            e.printStackTrace();
        }
    }

    public void loadModel() {
        final File file = new File("agent.json");
        if (!file.exists()) {
            return;
        }
        try {
            final BufferedReader reader = new BufferedReader(new FileReader(file));

            final String json = reader.lines().collect(Collectors.joining());

            final JsonParser parser = new JsonParser();
            final JsonObject jsonObject = parser.parse(json).getAsJsonObject();

            this.W1 = Mat.fromJson(jsonObject.get("W1").getAsString());
            this.W2 = Mat.fromJson(jsonObject.get("W2").getAsString());
            this.B1 = Mat.fromJson(jsonObject.get("B1").getAsString());
            this.B2 = Mat.fromJson(jsonObject.get("B2").getAsString());

            reader.close();
            System.out.println("LOADED");
        } catch (final IOException e) {
            e.printStackTrace();
        }
    }

}
