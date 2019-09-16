package de.raimannma.reinforce4j;

import java.util.Arrays;
import java.util.Map;
import java.util.stream.IntStream;

public class MultiAgents {
    public final DQN[] agents;
    private final Map<Option, Double> config;
    Mat W1, W2, B1, B2;

    public MultiAgents(final int numAgents, final int numActions, final int numStates, final Map<Option, Double> config) {
        this.config = config;
        final int numHiddenUnits = DQN.toInteger(config.getOrDefault(Option.NUM_HIDDEN_UNITS, 100.0));
        this.W1 = DQN.createRandMat(numHiddenUnits, numStates);
        this.W2 = DQN.createRandMat(numActions, numHiddenUnits);
        this.B1 = new Mat(numHiddenUnits, 1);
        this.B2 = new Mat(numActions, 1);


        this.agents = new DQN[numAgents];
        Arrays.setAll(this.agents, i -> new DQN(numActions, numStates, config, i, new Mat[]{this.W1, this.W2, this.B1, this.B2}));
    }

    public int[] act(final double[]... states) {
        if (states.length != this.agents.length) {
            throw new ArrayIndexOutOfBoundsException("Num_States != Num_Agents");
        }
        return IntStream.range(0, this.agents.length).parallel().map(i -> this.agents[i].act(states[i])).toArray();
    }

    public void learn(final double... rewards) {
        if (rewards.length != this.agents.length) {
            throw new ArrayIndexOutOfBoundsException("Num_Rewards != Num_Agents");
        }
        IntStream.range(0, this.agents.length).parallel().forEach(i -> this.agents[i].learn(rewards[i]));
    }

    public void saveAgents() {
        this.agents[0].saveModel();
    }

    public void loadAgents() {
        this.agents[0].loadModel();
        this.W1 = this.agents[0].W1;
        this.W2 = this.agents[0].W2;
        this.B1 = this.agents[0].B1;
        this.B2 = this.agents[0].B2;
        Arrays.setAll(this.agents, i -> new DQN(this.agents[0].numActions, this.agents[0].numStates, this.config, i, new Mat[]{this.W1, this.W2, this.B1, this.B2}));
    }

    public int getNumAgents() {
        return this.agents.length;
    }

    public DQN getAgent(final int index) {
        return this.agents[index];
    }

    @Override
    public String toString() {
        return "MultiAgents{" +
                ", W1=" + this.W1.toString() +
                ", W2=" + this.W2.toString() +
                ", B1=" + this.B1.toString() +
                ", B2=" + this.B2.toString() +
                '}';
    }
}
