package de.raimannma.reinforce4j;

import java.util.Random;

enum Utils {
    ;

    private static final Random rand = new Random();

    public static double randN(final double mu, final double std) {
        return mu + Utils.rand.nextGaussian() * std;
    }

}
