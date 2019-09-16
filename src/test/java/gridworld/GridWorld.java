package gridworld;

import de.raimannma.reinforce4j.DQN;
import org.knowm.xchart.QuickChart;
import org.knowm.xchart.SwingWrapper;
import org.knowm.xchart.XYChart;
import processing.core.PApplet;

import java.awt.*;
import java.util.*;

public class GridWorld extends PApplet {
    private final Random rand = new Random();
    private final Tile[][] grid = new Tile[5][5];
    private final Deque<Double> xData = new ArrayDeque<>(200);
    private final Deque<Double> yData = new ArrayDeque<>(200);
    private final Deque<Double> rewardWindow = new ArrayDeque<>(200);
    DQN agent;
    private int epoch = 0;
    private XYChart chart;
    private SwingWrapper<XYChart> sw;

    public GridWorld() {
    }

    public void init(final String[] args) {
        PApplet.main(GridWorld.class, args);
    }

    @Override
    public void settings() {
        this.size(900, 900);
    }

    @Override
    public void setup() {
        this.agent = Main.agentsIterator.next();
        this.chart = QuickChart.getChart("Simple XChart Real-time", "Epoch", "Reward", "reward", new double[]{0}, new double[]{0});

        this.sw = new SwingWrapper<>(this.chart);
        this.sw.displayChart();
        this.generateGrid();

        this.placeAgent();
        this.surface.setResizable(true);
    }

    private void generateGrid() {
        for (int i = 0; i < this.grid.length; i++) {
            for (int j = 0; j < this.grid[0].length; j++) {
                this.grid[i][j] = new Tile(i, j);
            }
        }
    }

    private void placeAgent() {
        final int x = this.rand.nextInt(this.grid.length);
        final int y = this.rand.nextInt(this.grid[0].length);
        this.grid[x][y].setAgent();
    }

    @Override
    public void draw() {
        this.background(255);
        this.doAction(this.agent.act(this.getState()));
        final double reward = this.getReward();
        this.agent.learn(reward);

        if (this.rand.nextDouble() < 0.1) {
            this.randomSpawnGreen();
            this.randomSpawnRed();
        }

        final int tileSize = this.width / this.grid.length;
        for (int i = 0; i < this.grid.length; i++) {
            this.line(0, tileSize * i + tileSize, this.width, tileSize * i + tileSize);
        }
        for (int j = 0; j < this.grid[0].length; j++) {
            this.line(tileSize * j + tileSize, 0, tileSize * j + tileSize, this.height);
        }


        for (int i = 0; i < this.grid.length; i++) {
            for (int j = 0; j < this.grid[0].length; j++) {
                this.color(0);
                this.fill(0);
                if (!this.grid[i][j].isEmpty()) {
                    this.fill(Color.ORANGE.getRGB());
                    this.rect(i * tileSize + tileSize * 0.4f, j * tileSize + tileSize * 0.4f, tileSize * 0.2f, tileSize * 0.2f);
                } else if (this.grid[i][j].isGreen) {
                    this.fill(Color.GREEN.getRGB());
                    this.rect(i * tileSize, j * tileSize, tileSize, tileSize);
                } else if (this.grid[i][j].isRed) {
                    this.fill(Color.RED.getRGB());
                    this.rect(i * tileSize, j * tileSize, tileSize, tileSize);
                }
            }
        }
        System.out.println();
        System.out.println("Epoch #" + this.epoch);
        System.out.println("Reward: " + reward);
        this.epoch++;
        this.rewardWindow.add(reward);
        this.xData.add((double) this.epoch);
        this.yData.add(this.rewardWindow.stream().mapToDouble(val -> val).average().orElse(0));
        this.chart.updateXYSeries("reward", new ArrayList<>(this.xData), new ArrayList<>(this.yData), null);
        this.sw.repaintChart();

        if (this.epoch > 10000) {
            this.noLoop();
            this.frame.dispose();
            this.frame.setVisible(false);
        }
    }

    private void doAction(final int action) {
        final Point agentPos = this.getAgentPos();
        if (action == 0) {
            if (agentPos.y - 1 >= 0) {
                this.grid[agentPos.x][agentPos.y].removeAgent();
                this.grid[agentPos.x][agentPos.y - 1].setAgent();
            }
        } else if (action == 1) {
            if (agentPos.y + 1 < this.grid[0].length) {
                this.grid[agentPos.x][agentPos.y].removeAgent();
                this.grid[agentPos.x][agentPos.y + 1].setAgent();
            }
        } else if (action == 2) {
            if (agentPos.x + 1 < this.grid.length) {
                this.grid[agentPos.x][agentPos.y].removeAgent();
                this.grid[agentPos.x + 1][agentPos.y].setAgent();
            }
        } else if (action == 3 && agentPos.x - 1 >= 0) {
            this.grid[agentPos.x][agentPos.y].removeAgent();
            this.grid[agentPos.x - 1][agentPos.y].setAgent();
        }
    }

    private double[] getState() {
        final double[] state = new double[this.grid.length * this.grid[0].length + 2];
        Arrays.fill(state, 0);
        for (int i = 0; i < this.grid.length; i++) {
            for (int j = 0; j < this.grid[0].length; j++) {
                if (!this.grid[i][j].isEmpty()) {
                    state[i * this.grid[0].length + j] = this.grid[i][j].toINT();
                }
            }
        }
        final Point agentPos = this.getAgentPos();
        state[state.length - 2] = agentPos.x;
        state[state.length - 1] = agentPos.y;
        return state;
    }

    private double getReward() {
        for (final Tile[] tiles : this.grid) {
            for (int j = 0; j < this.grid[0].length; j++) {
                if (tiles[j].hasAgent()) {
                    return tiles[j].getReward();
                }
            }
        }
        return 0;
    }

    private void randomSpawnGreen() {
        int x, y;
        do {
            x = this.rand.nextInt(this.grid.length);
            y = this.rand.nextInt(this.grid[0].length);
        } while (!this.grid[x][y].isEmpty() || this.grid[x][y].isRed);
        this.grid[x][y].setGreen();
    }

    private void randomSpawnRed() {
        int x, y;
        do {
            x = this.rand.nextInt(this.grid.length);
            y = this.rand.nextInt(this.grid[0].length);
        } while (!this.grid[x][y].isEmpty() || this.grid[x][y].isGreen);
        this.grid[x][y].setRed();
    }

    private Point getAgentPos() {
        for (int i = 0; i < this.grid.length; i++) {
            for (int j = 0; j < this.grid[0].length; j++) {
                if (this.grid[i][j].hasAgent()) {
                    return new Point(i, j);
                }
            }
        }
        return new Point(0, 0);
    }
}
