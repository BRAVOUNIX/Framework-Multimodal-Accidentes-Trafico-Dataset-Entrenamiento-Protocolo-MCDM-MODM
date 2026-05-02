#!/usr/bin/env bash
# ============================================================
# csvreporte.sh — Visualizar columnas específicas de un CSV
#
# USO:
#   csvreporte.sh <archivo.csv> <prefijo> [prefijo2 ...]
#   csvreporte.sh <archivo.csv> all
#
# EJEMPLOS:
#   csvreporte.sh results.csv det
#   csvreporte.sh results.csv cls
#   csvreporte.sh results.csv vlm
#   csvreporte.sh results.csv llm
#   csvreporte.sh results.csv det cls
#   csvreporte.sh results.csv det cls vlm llm
#   csvreporte.sh results.csv all
#
# ── EDITAR AQUÍ: columnas específicas por prefijo ────────────
# Dejar vacío ("") para mostrar TODAS las columnas del prefijo.
# ─────────────────────────────────────────────────────────────

COLS_DET="image_name det_objects det_precision det_recall det_f1 det_map_50 det_map_50_95 det_iou det_latency_ms"

COLS_CLS="cls_predicted cls_confidence cls_accuracy cls_precision cls_recall cls_f1"

COLS_VLM="vlm_tokens vlm_bleu1 vlm_bleu2 vlm_bleu4 vlm_meteor vlm_rouge1_f vlm_rougeL_f vlm_perplexity vlm_bertscore"

COLS_LLM="llm_tokens llm_type llm_bleu1 llm_bleu2 llm_bleu4 llm_meteor llm_rouge1_f llm_rougeL_f llm_perplexity llm_bertscore llm_mmlu llm_mauve"

# ─────────────────────────────────────────────────────────────
# No modificar debajo de esta línea
# ─────────────────────────────────────────────────────────────
set -e

# ── Validaciones ─────────────────────────────────────────────
if [ $# -lt 2 ]; then
    echo ""
    echo "Uso: csvreporte.sh <archivo.csv> <prefijo> [prefijo2 ...]"
    echo ""
    echo "Prefijos disponibles:"
    echo "  det   → $(echo $COLS_DET | tr ' ' ', ')"
    echo "  cls   → $(echo $COLS_CLS | tr ' ' ', ')"
    echo "  vlm   → $(echo $COLS_VLM | tr ' ' ', ')"
    echo "  llm   → $(echo $COLS_LLM | tr ' ' ', ')"
    echo "  all   → todas las columnas del CSV"
    echo ""
    echo "Ejemplos:"
    echo "  csvreporte.sh results.csv det"
    echo "  csvreporte.sh results.csv det cls"
    echo "  csvreporte.sh results.csv det cls vlm llm"
    echo "  csvreporte.sh results.csv all"
    echo ""
    exit 1
fi

CSV_FILE="$1"

if [ ! -f "$CSV_FILE" ]; then
    echo "❌ Archivo no encontrado: $CSV_FILE"
    exit 1
fi

shift
PREFIXES=("$@")

# ── Construir lista de columnas a mostrar ─────────────────────
build_col_filter() {
    local prefixes=("$@")

    if [ "${prefixes[0]}" = "all" ]; then
        echo "cols = list(df.columns)"
        return
    fi

    local explicit_cols=()
    local fallback_conditions=()

    for prefix in "${prefixes[@]}"; do
        case "$prefix" in
            det)
                if [ -n "$COLS_DET" ]; then
                    for c in $COLS_DET; do explicit_cols+=("'$c'"); done
                else
                    fallback_conditions+=("c.startswith('det')")
                fi
                ;;
            cls)
                if [ -n "$COLS_CLS" ]; then
                    for c in $COLS_CLS; do explicit_cols+=("'$c'"); done
                else
                    fallback_conditions+=("c.startswith('cls')")
                fi
                ;;
            vlm)
                if [ -n "$COLS_VLM" ]; then
                    for c in $COLS_VLM; do explicit_cols+=("'$c'"); done
                else
                    fallback_conditions+=("c.startswith('vlm')")
                fi
                ;;
            llm)
                if [ -n "$COLS_LLM" ]; then
                    for c in $COLS_LLM; do explicit_cols+=("'$c'"); done
                else
                    fallback_conditions+=("c.startswith('llm')")
                fi
                ;;
            *)
                fallback_conditions+=("c.startswith('${prefix}')")
                ;;
        esac
    done

    local filter=""

    if [ ${#explicit_cols[@]} -gt 0 ] && [ ${#fallback_conditions[@]} -gt 0 ]; then
        local el; el=$(IFS=","; echo "${explicit_cols[*]}")
        local fc; fc=$(IFS=" or "; echo "${fallback_conditions[*]}")
        filter="explicit=[${el}]; fallback=[c for c in df.columns if ${fc}]; cols=[c for c in explicit+fallback if c in df.columns]"
    elif [ ${#explicit_cols[@]} -gt 0 ]; then
        local el; el=$(IFS=","; echo "${explicit_cols[*]}")
        filter="explicit=[${el}]; cols=[c for c in explicit if c in df.columns]"
    elif [ ${#fallback_conditions[@]} -gt 0 ]; then
        local fc; fc=$(IFS=" or "; echo "${fallback_conditions[*]}")
        filter="cols=[c for c in df.columns if ${fc}]"
    else
        filter="cols=list(df.columns)"
    fi

    echo "$filter"
}

FILTER=$(build_col_filter "${PREFIXES[@]}")
LABEL=$(IFS="+"; echo "${PREFIXES[*]}" | tr '[:lower:]' '[:upper:]')

# ── Ejecutar Python + paginar ─────────────────────────────────
python3 - "$CSV_FILE" "$LABEL" "$FILTER" << 'PYEOF' | (command -v less &>/dev/null && less -S || command -v more &>/dev/null && more || cat)
import sys
import pandas as pd

csv_file    = sys.argv[1]
label       = sys.argv[2]
filter_expr = sys.argv[3]

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',    None)
pd.set_option('display.width',       None)
pd.set_option('display.max_colwidth',None)

df = pd.read_csv(csv_file)

exec(filter_expr)

if not cols:
    print(f"⚠️  No se encontraron columnas para: {label}")
    sys.exit(1)

total_rows = len(df)
total_cols = len(cols)

print(f"{'='*72}")
print(f"  Archivo  : {csv_file}")
print(f"  Sección  : {label}")
print(f"  Filas    : {total_rows}  |  Columnas: {total_cols}")
print(f"  Columnas : {', '.join(cols)}")
print(f"{'='*72}")
print()
print(df[cols].to_string(index=False))
print()
print(f"[{total_rows} filas × {total_cols} columnas]  — q: salir  ←→: scroll horizontal")
PYEOF
