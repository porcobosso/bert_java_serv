package com.porco.bert.chinese.serv;

public class ModelTest {

    public static void main(String[] args) throws Exception{
        Preprocessor preprocessor = new Preprocessor();
        preprocessor.initVocab();
        Model model = new Model(
                "/Users/future/github/bert_java_serv/models/sample0",
                48,12, preprocessor
        );
        int[] labels = model.predictLabel(new String[]{"我丈夫是军人，我想离婚，怎么办？"});
        assert labels[0] == 3;
    }

}
