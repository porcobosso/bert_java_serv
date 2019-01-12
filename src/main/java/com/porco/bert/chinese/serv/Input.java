package com.porco.bert.chinese.serv;

import org.tensorflow.Tensor;

import java.nio.IntBuffer;

/**
 * bert model input
 */
public class Input {
    /**
     * tensor input_ids
     */
    Tensor inputIds;
    /**
     * tensor input_mask
     */
    Tensor inputMask;
    /**
     * tensor segment_ids
     */
    Tensor segmentIds;

    public Input(){}

    public Input(IntBuffer inputIds,
                 IntBuffer inputMask,
                 IntBuffer segmentIds, long[] shape){
        this.inputIds = tensor(inputIds, shape);
        this.inputMask = tensor(inputMask, shape);
        this.segmentIds = tensor(segmentIds, shape);
    }

    /**
     * create tensor from int array
     * @param input int array
     * @return tensor
     */
    private Tensor tensor(IntBuffer input, long[] shape){
        input.flip();
        return Tensor.create(shape, input);
    }
}
