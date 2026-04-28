#!/usr/bin/env bash
# End-to-end driver: territorial agent does explore, Qwen does eval.
#
# Strategy:
#   1. Run our agent through env's exploration to a /tmp/ dir
#   2. Locate the existing Qwen baseline sample dir
#      (results/baseline_qwen/qwen2-5-vl-7b/<hash>/text/active/think/)
#   3. Copy our messages.json + exploration_turn_logs.json + history_state.json
#      OVER the Qwen ones (so the path is reused, eval phase reads our trajectory)
#   4. Re-run spatial_run.py --phase eval+cogmap+aggregate with --eval-override
#      --cogmap-override; Qwen answers eval questions using OUR cogmap.
#   5. Diff metrics.json against the original Qwen baseline.
#
# Usage:
#   ./run_territorial_pipeline.sh [seed] [mode]
#   defaults: seed=0  mode=dual

set -euo pipefail

SEED="${1:-0}"
MODE="${2:-dual}"
MAX_STEPS=15
TOS_ROOT=/root/autodl-tmp/tos
DATA_DIR="$TOS_ROOT/room_data/3-room"

# Where the territorial-only run will write
TERR_OUT="/tmp/territorial_seed${SEED}_${MODE}"
rm -rf "$TERR_OUT"
mkdir -p "$TERR_OUT"

# Where Qwen's baseline sample dir lives — must exist already
QWEN_ROOT="$TOS_ROOT/results/baseline_qwen"
QWEN_SAMPLE_DIR=$(find "$QWEN_ROOT/qwen2-5-vl-7b" -type d -name think | head -1)
if [[ -z "$QWEN_SAMPLE_DIR" ]]; then
    echo "ERROR: cannot find Qwen sample dir under $QWEN_ROOT"
    echo "Run the Qwen baseline first (results/baseline_qwen exists)."
    exit 1
fi
echo "[pipeline] Qwen sample dir: $QWEN_SAMPLE_DIR"

# Backup Qwen's originals (only on first run)
BACKUP_DIR="$QWEN_SAMPLE_DIR/_qwen_backup"
if [[ ! -d "$BACKUP_DIR" ]]; then
    mkdir -p "$BACKUP_DIR"
    cp "$QWEN_SAMPLE_DIR"/messages.json "$BACKUP_DIR/" 2>/dev/null || true
    cp "$QWEN_SAMPLE_DIR"/exploration_turn_logs.json "$BACKUP_DIR/" 2>/dev/null || true
    cp "$QWEN_SAMPLE_DIR"/history_state.json "$BACKUP_DIR/" 2>/dev/null || true
    cp "$QWEN_SAMPLE_DIR"/metrics.json "$BACKUP_DIR/" 2>/dev/null || true
    echo "[pipeline] Qwen originals backed up → $BACKUP_DIR"
else
    echo "[pipeline] Qwen backup already exists at $BACKUP_DIR (skipping)"
fi

# ---------- Step 1: run territorial agent through env ----------
echo
echo "============================================================"
echo "Step 1/4: Territorial agent explore (seed=$SEED, mode=$MODE)"
echo "============================================================"
cd "$TOS_ROOT"
python "$TOS_ROOT/run_territorial_explore.py" \
    --run-id "$SEED" \
    --mode "$MODE" \
    --max-steps "$MAX_STEPS" \
    --data-dir "$DATA_DIR" \
    --output-dir "$TERR_OUT" \
    --render-mode text

# ---------- Step 2: locate territorial files ----------
echo
echo "[pipeline] Locating territorial output files..."
TERR_MSGS=$(find "$TERR_OUT" -name messages.json | head -1)
TERR_EXPLOG=$(find "$TERR_OUT" -name exploration_turn_logs.json | head -1)
TERR_HSTATE=$(find "$TERR_OUT" -name history_state.json | head -1)

if [[ -z "$TERR_MSGS" || -z "$TERR_EXPLOG" ]]; then
    echo "ERROR: territorial run did not produce expected files."
    echo "  messages.json: $TERR_MSGS"
    echo "  exploration_turn_logs.json: $TERR_EXPLOG"
    find "$TERR_OUT" -type f | head -20
    exit 2
fi
echo "  messages: $TERR_MSGS"
echo "  exp_log:  $TERR_EXPLOG"
echo "  hstate:   $TERR_HSTATE"

# ---------- Step 3: overwrite Qwen sample dir with our trajectory ----------
echo
echo "============================================================"
echo "Step 2/4: Inject territorial trajectory into Qwen sample dir"
echo "============================================================"
cp "$TERR_MSGS"    "$QWEN_SAMPLE_DIR/messages.json"
cp "$TERR_EXPLOG"  "$QWEN_SAMPLE_DIR/exploration_turn_logs.json"
[[ -n "$TERR_HSTATE" ]] && cp "$TERR_HSTATE" "$QWEN_SAMPLE_DIR/history_state.json"
echo "[pipeline] copied territorial files into $QWEN_SAMPLE_DIR"

# Wipe stale eval / aggregate so spatial_run.py truly re-runs them
rm -f "$QWEN_SAMPLE_DIR/evaluation_turn_logs.json"
rm -f "$QWEN_SAMPLE_DIR/metrics.json"

# ---------- Step 4: re-run eval + cogmap + aggregate with Qwen ----------
echo
echo "============================================================"
echo "Step 3/4: Re-run eval + cogmap + aggregate (Qwen answers)"
echo "============================================================"
export OPENAI_API_KEY=dummy   # vllm doesn't validate
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

python "$TOS_ROOT/scripts/SpatialGym/spatial_run.py" \
    --phase eval \
    --eval-override \
    --model-name qwen2-5-vl-7b \
    --num 1 \
    --seed-range "$SEED-$SEED" \
    --data-dir "$DATA_DIR" \
    --output-root "$QWEN_ROOT" \
    --render-mode text \
    --exp-type active \
    --inference-mode direct \
    --max-exp-steps "$MAX_STEPS" \
    2>&1 | tail -40

python "$TOS_ROOT/scripts/SpatialGym/spatial_run.py" \
    --phase cogmap \
    --cogmap-override \
    --model-name qwen2-5-vl-7b \
    --num 1 \
    --seed-range "$SEED-$SEED" \
    --data-dir "$DATA_DIR" \
    --output-root "$QWEN_ROOT" \
    --render-mode text \
    --exp-type active \
    --inference-mode direct \
    --max-exp-steps "$MAX_STEPS" \
    2>&1 | tail -20

python "$TOS_ROOT/scripts/SpatialGym/spatial_run.py" \
    --phase aggregate \
    --model-name qwen2-5-vl-7b \
    --num 1 \
    --seed-range "$SEED-$SEED" \
    --data-dir "$DATA_DIR" \
    --output-root "$QWEN_ROOT" \
    --render-mode text \
    --exp-type active \
    --inference-mode direct \
    --max-exp-steps "$MAX_STEPS" \
    2>&1 | tail -10

# ---------- Step 5: print scores side-by-side ----------
echo
echo "============================================================"
echo "Step 4/4: Score comparison"
echo "============================================================"
python - <<EOF
import json
qwen_metrics = json.load(open("$BACKUP_DIR/metrics.json"))
terr_metrics = json.load(open("$QWEN_SAMPLE_DIR/metrics.json"))

def show(label, m):
    ev = m.get("evaluation", {})
    ov = ev.get("overall", {})
    print(f"\n[{label}]")
    print(f"  overall avg_accuracy: {ov.get('avg_accuracy', '?'):.4f}  (n={ov.get('n_total','?')}, score={ov.get('total_score','?'):.4f})")
    print(f"  per_task:")
    for tname, td in ev.get("per_task", {}).items():
        print(f"    {tname:40s}  acc={td.get('avg_accuracy', 0):.3f}  ({td.get('task_score',0):.2f}/{td.get('n_total',0)})")
    expl = m.get("exploration", {})
    print(f"  exploration:")
    print(f"    n_steps={expl.get('n_exploration_steps','?')}  valid_ratio={expl.get('valid_action_ratio','?')}  node_cov={expl.get('last_node_coverage','?'):.3f}  edge_cov={expl.get('last_edge_coverage','?'):.3f}")

show("QWEN baseline (explore + eval both Qwen)", qwen_metrics)
show("TERRITORIAL ($MODE) explore + Qwen eval", terr_metrics)

# Delta
qov = qwen_metrics.get("evaluation", {}).get("overall", {}).get("avg_accuracy", 0)
tov = terr_metrics.get("evaluation", {}).get("overall", {}).get("avg_accuracy", 0)
delta = tov - qov
arrow = "✅" if delta > 0 else "⚠️"
print(f"\n{arrow}  Δ overall avg_accuracy = {delta:+.4f}  (territorial - qwen)")
EOF

echo
echo "[pipeline] done. Territorial trajectory + scores at:"
echo "  $QWEN_SAMPLE_DIR"
echo "Qwen originals preserved at:"
echo "  $BACKUP_DIR"
echo
echo "To restore Qwen baseline:"
echo "  cp $BACKUP_DIR/* $QWEN_SAMPLE_DIR/"
