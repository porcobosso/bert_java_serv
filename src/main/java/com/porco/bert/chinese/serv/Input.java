package com.porco.bert.chinese.serv;

import org.tensorflow.Tensor;

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

    public Input(int[][] inputIds, int[][] inputMask, int[][] segmentIds){
        this.inputIds = tensor(inputIds);
        this.inputMask = tensor(inputMask);
        this.segmentIds = tensor(segmentIds);
    }

    /**
     * create tensor from int array
     * @param input int array
     * @return tensor
     */
    private Tensor tensor(int[][] input){
        return Tensor.create(input, Integer.class);
    }
}
