package de.raimannma.reinforce4j;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

class Mat {
    final int d;
    final int n;
    INDArray w;
    INDArray dw;

    Mat(final int n, final int d) {
        this.n = n;
        this.d = d;
        this.w = Nd4j.zeros(n * d);
        this.dw = Nd4j.zeros(n * d);
    }

    Mat(final int n, final int d, final INDArray arr) {
        this.n = n;
        this.d = d;
        this.w = arr;
        this.dw = Nd4j.zeros(n * d);
    }

    void update(final double val) {
        this.w = this.w.sub(this.dw.mul(val));
        this.dw = Nd4j.zeros(this.n * this.d);
    }
}
