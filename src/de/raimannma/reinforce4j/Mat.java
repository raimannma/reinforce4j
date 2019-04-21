package de.raimannma.reinforce4j;

import java.util.Arrays;

class Mat {
    final double[] w;
    final int d;
    final int n;
    final double[] dw;

    Mat(final int n, final int d) {
        this.n = n;
        this.d = d;
        this.w = Mat.zeros(n * d);
        this.dw = Mat.zeros(n * d);
    }

    private static double[] zeros(final int size) {
        final double[] arr = new double[size];
        Arrays.fill(arr, 0);
        return arr;
    }

    void setFrom(final double[] arr) {
        System.arraycopy(arr, 0, this.w, 0, arr.length);
    }

    void update(final double val) {
        for (int i = 0; i < this.n * this.d; i++) {
            this.w[i] += -val * this.dw[i];
            this.dw[i] = 0;
        }
    }
}
