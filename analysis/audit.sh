#!/bin/bash
# One-shot completeness audit across all three experiments. Prints what exists, what is
# still training, and what is missing -- so nothing silently drops out of the paper.
cd $CHARDIFF_ROOT
S=$CHARDIFF_DATA/parser_data/runs
I=$CHARDIFF_DATA/insc_data/runs
R=$CHARDIFF_DATA/runs
echo "==================== COMPLETENESS AUDIT $(date +%H:%M) ===================="
echo "queue: $(squeue -u $USER -h | wc -l) jobs ($(squeue -u $USER -h -t R | wc -l) running)"
echo
echo "-- EXP1 documentary --"
for pre in whole_v3 whole_v3_randinit; do
  tr=$(grep -l "FINETUNE DONE" logs/ftwhole-*.out 2>/dev/null | xargs -r grep -ho "${pre}_t[0-9]v[0-9]\b" 2>/dev/null | sort -u | wc -l)
  ev=$(ls .scratch/evals/${pre}_t*_iphi_whole_unk.json 2>/dev/null | wc -l)
  [ "$pre" = whole_v3 ] && ev=$(ls .scratch/evals/v3_t*_iphi_whole_unk.json 2>/dev/null | wc -l)
  [ "$pre" = whole_v3_randinit ] && ev=$(ls .scratch/evals/v3_randinit_t*_iphi_whole_unk.json 2>/dev/null | wc -l)
  printf "  %-22s trained %2d/10   evaluated %2d/10\n" "$pre" "$tr" "$ev"
done
sf=$(ls $CHARDIFF_DATA/strict_f3/ours_v3_shard*.json 2>/dev/null | wc -l)
echo "  strict Ithaca (v3): $sf/4 shards"
echo
echo "-- EXP2 parsing --"
for tag in joint_docclean joint_randinit joint_randinit_long joint_docclean_long joint_greberta joint_philberta joint_logion joint_agbert joint_xlmr_base joint_xlmr_large joint_mbert joint_tune; do
  t=$(ls -d $S/${tag}_f*_s* 2>/dev/null | wc -l)
  e=$(ls $S/${tag}_f*_s*/test_scores_greedy.json 2>/dev/null | wc -l)
  [ "$t" -gt 0 ] && printf "  %-24s runs %2d   scored %2d\n" "${tag#joint_}" "$t" "$e"
done
echo
echo "-- EXP3 meter --"
for tag in meter_mac_v2 meter_mac_v2_randinit meter_joint_docclean meter_joint_randinit; do
  t=$(ls -d $R/${tag}* 2>/dev/null | wc -l)
  e=$(grep -l "model=.*/runs/${tag}" logs/eval-*.out 2>/dev/null | wc -l)
  printf "  %-24s runs %2d   scored %2d\n" "$tag" "$t" "$e"
done
