for I in `seq 0 29`
do
    python eval_adversarial.py mlp_$I >> mlp.txt
done 
