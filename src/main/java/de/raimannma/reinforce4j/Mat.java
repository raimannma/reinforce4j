package de.raimannma.reinforce4j;

import java.util.Arrays;
import java.util.stream.IntStream;

class Mat {
    final double[] w;
    final double[] dw;
    final int d;
    final int n;

    Mat(final int n, final int d) {
        this.n = n;
        this.d = d;
        this.w = Mat.zeros(n * d);
        this.dw = Mat.zeros(n * d);
    }

    Mat(final int n, final int d, final double[] arr) {
        this.n = n;
        this.d = d;
        this.w = arr;
        this.dw = Mat.zeros(this.n * this.d);
    }

    private static double[] zeros(final int size) {
        final double[] arr = new double[size];
        Arrays.fill(arr, 0);
        return arr;
    }

    void update(final double val) {
        IntStream.range(0, this.w.length).forEach(i -> this.w[i] -= val * this.dw[i]);
        Arrays.fill(this.dw, 0);
    }
}
