OUTDIR=./data/CommonsenseQA
mkdir -p $OUTDIR

wget -O $OUTDIR/train.jsonl https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl
wget -O $OUTDIR/valid.jsonl https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl
wget -O $OUTDIR/test.jsonl https://s3.amazonaws.com/commensenseqa/test_rand_split_no_answers.jsonl
wget -O $OUTDIR/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt

cd ./data

wget https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1.zip
unzip winogrande_1.1.zip; rm -rf __MACOSX
mv winogrande_1.1 WinoGrande
rm winogrande_1.1.zip