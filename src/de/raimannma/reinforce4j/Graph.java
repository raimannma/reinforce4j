package de.raimannma.reinforce4j;

import net.jafama.FastMath;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Queue;
import java.util.stream.IntStream;

class Graph {
    private final boolean needsBackprop;
    private final Queue<Backprop> backpropQueue;

    Graph(final boolean needsBackprop) {
        this.needsBackprop = needsBackprop;
        this.backpropQueue = new ArrayDeque<>();
    }

    void backward() {
        while (!this.backpropQueue.isEmpty()) {
            this.backpropQueue.poll().run();
        }
    }

    Mat tanh(final Mat mat) {
        final Mat out = new Mat(mat.n, mat.d);
        Arrays.setAll(out.w, i -> FastMath.tanh(mat.w[i]));
        if (this.needsBackprop) {
            this.backpropQueue.add(new Backprop(BackpropMethod.TANH, mat, out));
        }
        return out;
    }

    Mat mul(final Mat mat1, final Mat mat2) {
        final int m1d = mat1.d;
        assert m1d == mat2.n;

        final int n = mat1.n;
        final int m2d = mat2.d;
        final Mat out = new Mat(n, m2d);
        IntStream.range(0, n).parallel().forEach(i -> {
            final int m1i = m1d * i;
            final int m2di = m2d * i;
            IntStream.range(0, m2d)
                    .parallel()
                    .forEach(finalJ -> out.w[m2di + finalJ] = IntStream.range(0, m1d)
                            .parallel()
                            .mapToDouble(value -> mat1.w[m1i + value] * mat2.w[m2d * value + finalJ])
                            .sum());
        });
        if (this.needsBackprop) {
            this.backpropQueue.add(new Backprop(BackpropMethod.MUL, mat1, mat2, out));
        }
        return out;
    }

    Mat add(final Mat mat1, final Mat mat2) {
        assert mat1.w.length == mat2.w.length;

        final Mat out = new Mat(mat1.n, mat1.d);
        IntStream.range(0, mat1.w.length).parallel().forEach(i -> out.w[i] = mat1.w[i] + mat2.w[i]);
        if (this.needsBackprop) {
            this.backpropQueue.add(new Backprop(BackpropMethod.ADD, mat1, mat2, out));
        }
        return out;
    }

    private enum BackpropMethod {
        ADD, MUL, TANH
    }

    private class Backprop {
        private final BackpropMethod backpropMethod;
        private final Mat[] args;

        private Backprop(final BackpropMethod backpropMethod, final Mat... args) {
            this.backpropMethod = backpropMethod;
            this.args = args;
        }


        void run() {
            if (this.backpropMethod == BackpropMethod.ADD) {
                this.addBack(this.args[0], this.args[1], this.args[2]);
            } else if (this.backpropMethod == BackpropMethod.MUL) {
                this.mulBack(this.args[0], this.args[1], this.args[2]);
            } else if (this.backpropMethod == BackpropMethod.TANH) {
                this.tanhBack(this.args[0], this.args[1]);
            }
        }

        private void mulBack(final Mat mat1, final Mat mat2, final Mat out) {
            final int n = mat1.n;
            final int m2d = mat2.d;
            final int m1d = mat1.d;
            for (int i = 0; i < n; i++) {
                final int m2di = m2d * i;
                final int m1di = m1d * i;
                for (int j = 0; j < m2d; j++) {
                    final double b = out.dw[m2di + j];
                    for (int k = 0; k < m1d; k++) {
                        final int mm1 = m1di + k;
                        final int mm2 = m2d * k + j;
                        mat1.dw[mm1] += mat2.w[mm2] * b;
                        mat2.dw[mm2] += mat1.w[mm1] * b;
                    }
                }
            }
        }

        private void addBack(final Mat mat1, final Mat mat2, final Mat out) {
            IntStream.range(0, mat1.w.length).parallel().forEach(i -> {
                mat1.dw[i] += out.dw[i];
                mat2.dw[i] += out.dw[i];
            });
        }

        private void tanhBack(final Mat mat, final Mat out) {
            IntStream.range(0, mat.w.length).parallel().forEach(i -> mat.dw[i] += (1 - out.w[i] * out.w[i]) * out.dw[i]);
        }
    }
}
