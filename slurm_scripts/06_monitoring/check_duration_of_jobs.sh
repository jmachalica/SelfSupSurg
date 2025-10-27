# funkcja: policz czas trwania pojedynczego joba na podstawie log.txt
job_duration() {
  dir="$1"
  f="$dir/log.txt"
  [[ -f "$f" ]] || { echo "brak $f"; return; }

  start=$(grep -oE '20[0-9]{2}-[01][0-9]-[0-3][0-9] [0-2][0-9]:[0-5][0-9]:[0-5][0-9]' "$f" | head -1)
  end=$(grep  -oE '20[0-9]{2}-[01][0-9]-[0-3][0-9] [0-2][0-9]:[0-5][0-9]:[0-5][0-9]' "$f" | tail -1)
  [[ -n "$start" && -n "$end" ]] || { echo "$dir: brak timestampów w $f"; return; }

  start_s=$(date -d "$start" +%s)
  end_s=$(date   -d "$end"   +%s)
  dur=$(( end_s - start_s ))
  echo "$dir: ${dur}s ($(date -ud "@$dur" +%H:%M:%S))  [od $start do $end]"
}

job_duration /net/tscratch/people/plgjmachali/surgvu_results/finetuning/moco_to_surgvu/12/job_1948590_20251014_0404
job_duration /net/tscratch/people/plgjmachali/surgvu_results/finetuning/moco_to_surgvu/25/job_1948591_20251014_0443
job_duration /net/tscratch/people/plgjmachali/surgvu_results/finetuning/moco_to_surgvu/100/job_1948592_20251014_0444

job_duration /net/tscratch/people/plgjmachali/surgvu_results/finetuning/simclr_to_surgvu/25/job_1952104_20251015_0322/
job_duration /net/tscratch/people/plgjmachali/surgvu_results/finetuning/simclr_to_surgvu/100/job_1952105_20251015_0322/

