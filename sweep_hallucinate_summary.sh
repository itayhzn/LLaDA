cd opencompass

# Baseline (no guidance)
PYTHONPATH=/scratch200/itaytuviah/LLaDA:$PYTHONPATH python run.py examples/llada_instruct_gen_gsm8k_length512_block512_confidence.py -w outputs/llada_instruct_gsm8k_baseline

# Hallucinate sweep: w_hallucinate x hallucinate_t
for w in 0p2 0p4 0p6 0p8; do
    for t in 1 2; do
        PYTHONPATH=/scratch200/itaytuviah/LLaDA:$PYTHONPATH python run.py \
            examples/llada_instruct_gen_gsm8k_length512_block512_confidence_hal_w${w}_t${t}.py \
            -w outputs/llada_instruct_gsm8k_hal_w${w}_t${t}
    done
done

# Summary sweep: w_summary x summary_t
for w in 0p2 0p4 0p6 0p8; do
    for t in 1 2; do
        PYTHONPATH=/scratch200/itaytuviah/LLaDA:$PYTHONPATH python run.py \
            examples/llada_instruct_gen_gsm8k_length512_block512_confidence_sum_w${w}_t${t}.py \
            -w outputs/llada_instruct_gsm8k_sum_w${w}_t${t}
    done
done
