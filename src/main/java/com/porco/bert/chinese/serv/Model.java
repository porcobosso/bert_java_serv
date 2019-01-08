package com.porco.bert.chinese.serv;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class Model {

    private Preprocessor preprocessor;

    private Session session;

    private int maxLength;

    private int numClasses;

    public Model(String modelPath, int maxLength, int numClasses, Preprocessor preprocessor){
        loadModel(modelPath);
        this.maxLength = maxLength;
        this.numClasses = numClasses;
        this.preprocessor = preprocessor;
    }

    /**
     * load model from path
     * @param modelPath model path
     */
    private void loadModel(String modelPath){
        SavedModelBundle bundle = SavedModelBundle.load(
                modelPath, "serve"
        );
        this.session = bundle.session();
    }

    private int[] labelOfPrediction(float[][] prediction){
        int sampleSize = prediction.length;
        int[] labels = new int[sampleSize];

        for(int i=0;i<sampleSize;i++){
            float[] probability = prediction[i];
            int label = 0;
            float maxProb = probability[0];
            float left = 1 - maxProb;
            for(int j=1;j<numClasses && maxProb < left;j++){
                float prob = probability[j];
                if(prob>maxProb){
                    label = j;
                    maxProb = prob;
                }
                left -= prob;
            }
            labels[i] = label;
        }

        return labels;
    }

    /**
     * input sample with one sentence，get probability of each class
     * @param sentences samples
     * @return probability
     */
    public float[][] predict(String[] sentences){
        Input input = preprocessor.preprocess(sentences, maxLength);
        Tensor t =session.runner()
                .feed("input_ids_1:0", input.inputIds)
                .feed("input_mask_1:0", input.inputMask)
                .feed("segment_ids_1:0", input.segmentIds)
                .fetch("loss/Softmax:0").run().get(0);
        float[][] prediction = new float[sentences.length][numClasses];
        t.copyTo(prediction);
        return prediction;
    }

    /**
     * input sample with one sentence，get predicted label of each sample
     * @param sentences samples
     * @return probability
     */
    public int[] predictLabel(String[] sentences){
        float[][] prediction = predict(sentences);
        return labelOfPrediction(prediction);
    }

    /**
     * input sample with one sentence，get probability of each class
     * @param questions sentence one of samples
     * @param answers sentence two of samples
     * @return probability
     */
    public float[][] predict(String[] questions, String[] answers){
        Input input = preprocessor.preprocess(questions, answers, maxLength);
        Tensor t =session.runner()
                .feed("input_ids_1:0", input.inputIds)
                .feed("input_mask_1:0", input.inputMask)
                .feed("segment_ids_1:0", input.segmentIds)
                .fetch("loss/Softmax:0").run().get(0);
        float[][] prediction = new float[questions.length][numClasses];
        t.copyTo(prediction);
        return prediction;
    }

    /**
     * input sample with one sentence，get predicted label of each sample
     * @param questions sentence one of samples
     * @param answers sentence two of samples
     * @return probability
     */
    public int[] predictLabel(String[] questions, String[] answers){
        float[][] prediction = predict(questions, answers);
        return labelOfPrediction(prediction);
    }
}
