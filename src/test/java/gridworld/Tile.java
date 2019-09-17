package gridworld;

public class Tile {
    private final int x;
    private final int y;
    boolean isGreen;
    boolean isRed;
    private boolean agent;

    public Tile(final int x, final int y) {
        this.x = x;
        this.y = y;
        this.isGreen = false;
        this.isRed = false;
    }

    public boolean hasAgent() {
        return this.agent;
    }

    public void removeAgent() {
        this.agent = false;
    }

    public void setGreen() {
        this.isGreen = true;
        new Thread(() -> {
            try {
                Thread.sleep(5000);
            } catch (final InterruptedException e) {
                e.printStackTrace();
            }
            Tile.this.isGreen = false;
        }).start();
    }

    public double toINT() {
        if (this.isRed) {
            return 0;
        } else if (this.isGreen) {
            return 1;
        } else {
            return 0.5;
        }
    }

    public boolean isEmpty() {
        return !this.agent;
    }

    public double getReward() {
        if (this.isGreen) {
            this.isGreen = false;
            return 1;
        } else if (this.isRed) {
            this.isRed = false;
            return -1;
        } else {
            return 0;
        }
    }

    public void setRed() {
        this.isRed = true;
        new Thread(() -> {
            try {
                Thread.sleep(10000);
            } catch (final InterruptedException e) {
                e.printStackTrace();
            }
            Tile.this.isRed = false;
        }).start();
    }

    public void setAgent() {
        this.agent = true;
    }
}
