package com.porco.bert.chinese.serv;

import javax.annotation.PostConstruct;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.IntBuffer;
import java.util.HashMap;
import java.util.Map;


public class Preprocessor {

    private static String CLS = "[CLS]";
    private static String SEP = "[SEP]";

    private Map<String, Integer> vocab;

    /**
     * init vocabulary from bert.vocab
     * used to turn character into index
     * @throws IOException usually not happen
     */
    @PostConstruct
    public void initVocab() throws IOException{
        this.vocab = new HashMap<String, Integer>();
        BufferedReader reader =new BufferedReader(new InputStreamReader(
                Preprocessor.class.getResourceAsStream("/bert.vocab")
        ));
        int index = 0;
        String line;
        while((line=reader.readLine())!=null){
            line = line.trim();
            if("".equals(line)) continue;
            this.vocab.put(line, index++);
        }
    }

    /**
     * process one sentence
     * @param pair is sentence pair
     * @param t start index
     * @param maxLength max length
     * @param sentence sentence
     * @param inputIds input ids
     * @param inputMask input mask
     * @param segmentIds segment ids
     * @return current start index
     */
    private int process(boolean pair, int t, int maxLength, String sentence, IntBuffer inputIds, IntBuffer inputMask, IntBuffer segmentIds){
        boolean isAnswer = t!=0;
        if(!isAnswer){
            // not first sentence, add CLS
            inputIds.put(vocab.get(CLS));
            inputMask.put(1);
            segmentIds.put(0);
            t++;
        }
        for(int j=0;j<sentence.length();j++){
            String c = String.valueOf(sentence.charAt(j));
            if(vocab.containsKey(c)){
                inputIds.put(vocab.get(c));
                segmentIds.put(isAnswer? 1:0);
                inputMask.put(1);
                t++;
            }
        }
        inputIds.put(vocab.get(SEP));
        segmentIds.put(isAnswer? 1:0);
        inputMask.put(1);
        t++;
        if(!pair || isAnswer){
            while(t<maxLength){
                inputIds.put(0);
                segmentIds.put(0);
                inputMask.put(0);
                t++;
            }
        }

        return t;
    }

    /**
     * turn sample with one sentence into model input
     * @param sentences samples
     * @param maxLength max length of sequence
     * @return model input
     */
    public Input preprocess(String[] sentences, int maxLength){
        int sampleSize = sentences.length;

        // The create(Object) method call involves use of reflection to determine the shape and copy things over one array at a time,
        // so it is pretty slow, especially as you add dimensions.
        // The create(shape, FloatBuffer) method would be an order-of-magnitude faster
        long[] shape = new long[]{sampleSize, maxLength};
        IntBuffer inputIds = IntBuffer.allocate(sampleSize*maxLength);
        IntBuffer inputMask = IntBuffer.allocate(sampleSize*maxLength);
        IntBuffer segmentIds = IntBuffer.allocate(sampleSize*maxLength);

        for(int i=0;i<sampleSize;i++){
            String sentence = sentences[i];
            process(false, 0 , maxLength, sentence, inputIds, inputMask, segmentIds);
        }
        return new Input(inputIds, inputMask, segmentIds, shape);
    }


    /**
     * turn sample with (question, answer) pair into model input
     * @param questions question or first sentence
     * @param answers answer or first sentence
     * @param maxLength max length of concatenated sentence
     * @return model input
     */
    public Input preprocess(String[] questions, String[] answers, int maxLength){
        int sampleSize = questions.length;
        long[] shape = new long[]{sampleSize, maxLength};
        IntBuffer inputIds = IntBuffer.allocate(sampleSize*maxLength);
        IntBuffer inputMask = IntBuffer.allocate(sampleSize*maxLength);
        IntBuffer segmentIds = IntBuffer.allocate(sampleSize*maxLength);

        for(int i=0;i<sampleSize;i++){
            String question = questions[i];
            String answer = answers[i];
            int t = process(true, 0 , maxLength, question, inputIds, inputMask, segmentIds);
            process(true, t , maxLength, answer, inputIds, inputMask, segmentIds);
        }

        return new Input(inputIds, inputMask, segmentIds, shape);
    }
}
