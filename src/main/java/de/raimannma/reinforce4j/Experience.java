package de.raimannma.reinforce4j;

class Experience {
    private final Mat lastState;
    private final int lastAction;
    private final double lastReward;
    private final Mat currentState;

    Experience(final Mat lastState, final int lastAction, final double lastReward, final Mat currentState) {
        this.lastState = lastState;
        this.lastAction = lastAction;
        this.lastReward = lastReward;
        this.currentState = currentState;
    }

    Mat getLastState() {
        return this.lastState;
    }

    int getLastAction() {
        return this.lastAction;
    }

    double getLastReward() {
        return this.lastReward;
    }

    Mat getCurrentState() {
        return this.currentState;
    }
}
