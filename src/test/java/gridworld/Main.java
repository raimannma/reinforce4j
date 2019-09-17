package gridworld;

import de.raimannma.reinforce4j.DQN;
import de.raimannma.reinforce4j.MultiAgents;
import de.raimannma.reinforce4j.Option;

import java.util.*;
import java.util.stream.IntStream;

public enum Main {
    ;

    private static final int NUM_AGENTS = 1;
    static Iterator<DQN> agentsIterator;
    private static MultiAgents agents;

    public static void main(final String[] args) {
        final Map<Option, Double> config = new HashMap<>();
        config.put(Option.EXPERIENCE_SIZE, 100e2);
        config.put(Option.EXPERIENCE_ADD_EVERY, 100.0);
        config.put(Option.LEARNING_STEPS_PER_ITERATION, 50.0);
        config.put(Option.NUM_HIDDEN_UNITS, 100.0);

        Main.agents = new MultiAgents(Main.NUM_AGENTS, 5, 27, config);
        Main.agents.loadAgents();
        Main.agentsIterator = new HashSet<>(Arrays.asList(Main.agents.agents)).iterator();

        IntStream.range(0, Main.NUM_AGENTS).parallel().forEach(i -> {
            new GridWorld().init(args);
        });
        Main.agents.saveAgents();
    }
}
