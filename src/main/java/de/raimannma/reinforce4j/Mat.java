package de.raimannma.reinforce4j;

import com.google.gson.GsonBuilder;

import java.util.Arrays;
import java.util.stream.IntStream;

class Mat {
    final int d;
    final int n;
    double[] w;
    double[] dw;

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

    Mat(final int n, final int d, final double[] arr) {
        this.n = n;
        this.d = d;
        this.w = arr;
        this.dw = Mat.zeros(this.n * this.d);
    }

    public static Mat fromJson(final String json) {
        return new GsonBuilder().setPrettyPrinting().create().fromJson(json, Mat.class);
    }

    void update(final double val) {
        IntStream.range(0, this.w.length).forEach(i -> this.w[i] -= val * this.dw[i]);
        Arrays.fill(this.dw, 0);
    }

    public String toJson() {
        return new GsonBuilder().setPrettyPrinting().create().toJson(this);
    }

    @Override
    public String toString() {
        return "Mat{" +
                "d=" + this.d +
                ", n=" + this.n +
                ", w=" + Arrays.toString(this.w) +
                ", dw=" + Arrays.toString(this.dw) +
                '}';
    }
}
