package com.porco.bert.chinese.serv;

import javax.annotation.PostConstruct;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
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
        this.vocab = new HashMap<>();
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
     * @param isAnswer is the answer sentence(or seconded sentence)
     * @param t start index
     * @param sampleIndex sample index
     * @param sentence sentence
     * @param inputIds input ids
     * @param inputMask input mask
     * @param segmentIds segment ids
     * @return current start index
     */
    private int process(boolean isAnswer,int t, int sampleIndex, String sentence, int[][] inputIds, int[][] inputMask, int[][] segmentIds){
        if(!isAnswer){
            // 不是第二句，才加上 CLS
            inputIds[sampleIndex][t] = vocab.get(CLS);
            inputMask[sampleIndex][t++] = 1;
        }
        for(int j=0;j<sentence.length();j++){
            String c = String.valueOf(sentence.charAt(j));
            if(vocab.containsKey(c)){
                inputIds[sampleIndex][t] = vocab.get(c);
                if(isAnswer) segmentIds[sampleIndex][t] = 1;
                inputMask[sampleIndex][t++] = 1;
            }
        }
        inputIds[sampleIndex][t] = vocab.get(SEP);
        if(isAnswer) segmentIds[sampleIndex][t] = 1;
        inputMask[sampleIndex][t++] = 1;
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
        int[][] inputIds = new int[sampleSize][maxLength];
        int[][] inputMask = new int[sampleSize][maxLength];
        int[][] segmentIds = new int[sampleSize][maxLength];

        for(int i=0;i<sampleSize;i++){
            String sentence = sentences[i];
            process(false, 0 , i, sentence, inputIds, inputMask, segmentIds);
        }
        return new Input(inputIds, inputMask, segmentIds);
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
        int[][] inputIds = new int[sampleSize][maxLength];
        int[][] inputMask = new int[sampleSize][maxLength];
        int[][] segmentIds = new int[sampleSize][maxLength];

        for(int i=0;i<sampleSize;i++){
            String question = questions[i];
            String answer = answers[i];
            int t = process(false, 0 , i, question, inputIds, inputMask, segmentIds);
            process(true, t , i, answer, inputIds, inputMask, segmentIds);
        }

        return new Input(inputIds, inputMask, segmentIds);
    }
}
