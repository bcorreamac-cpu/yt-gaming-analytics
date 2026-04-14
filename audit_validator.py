#!/usr/bin/env python3
"""
Audit Validator — Joy Of Gaming
Verifica clasificaciones, corrige errores, recalcula keywords y regenera dashboard.
"""

import re
import json
import numpy as np
import pandas as pd
from datetime import datetime

INPUT_CSV = "videos_categorized.csv"
RAW_CSV = "videos_raw_data.csv"
OUTPUT_CSV = "videos_categorized.csv"
AUDIT_HTML = "audit_report.html"

# ============================================================
# PASO 1: Verificar que game_name aparezca en título o descripción
# ============================================================

def build_search_terms(game_name):
    """Genera términos de búsqueda a partir del nombre del juego."""
    terms = [game_name.lower()]
    # Versiones simplificadas
    simple = game_name.lower()
    simple = re.sub(r'[:\-–—]', ' ', simple)
    simple = re.sub(r'\s+', ' ', simple).strip()
    terms.append(simple)
    # Palabras clave principales (primera y segunda palabra significativa)
    words = [w for w in simple.split() if len(w) > 2 and w not in ('the','and','for','del','las','los','von','der')]
    if words:
        terms.append(words[0])
        if len(words) > 1:
            terms.append(words[0] + ' ' + words[1])
    return list(set(terms))


def audit_classifications(df):
    """Audita cada video verificando que el juego asignado tenga relación con el título."""
    skip_games = {'Unknown (Template)', 'Test / Interno', 'Stream / GoPro',
                  'Otro / No Identificado', 'Otro Contenido'}

    results = []
    for idx, row in df.iterrows():
        game = str(row.get('game_name', ''))
        if game in skip_games:
            results.append({'idx': idx, 'status': 'SKIP', 'reason': 'No clasificado'})
            continue

        title = str(row.get('title', '')).lower()
        desc = str(row.get('description', '')).lower()
        tags = str(row.get('tags', '')).lower()

        search_terms = build_search_terms(game)
        found_in_title = any(t in title for t in search_terms)
        found_in_desc = any(t in desc for t in search_terms)
        found_in_tags = any(t in tags for t in search_terms)

        if found_in_title:
            results.append({'idx': idx, 'status': 'OK', 'reason': 'En título'})
        elif found_in_desc or found_in_tags:
            results.append({'idx': idx, 'status': 'OK_DESC', 'reason': 'En descripción/tags'})
        else:
            results.append({'idx': idx, 'status': 'REVISAR', 'reason': 'No encontrado',
                           'title': row.get('title', ''), 'assigned_game': game,
                           'assigned_genre': row.get('game_genre', '')})

    return results


def try_reclassify(title, tags):
    """Intenta reclasificar un video mal clasificado basándose en su título."""
    # Importar el clasificador
    from categorizer import classify_game
    game, genre = classify_game(title, tags)
    return game, genre


# ============================================================
# PASO 2: Recalcular keywords SOLO desde el título
# ============================================================

TITLE_KEYWORDS = [
    'Ultra Realistic', '4K', 'HDR', '60FPS', '60fps', '60 FPS',
    'PS5', 'PS4', 'Ray Tracing', 'Cinematic', 'BEST', 'NEVER',
    'AMAZING', 'INCREDIBLE', 'BEAUTIFUL', 'STUNNING', 'INSANE',
    'EPIC', 'FREE ROAM', 'First Person', 'FIRST PERSON',
    'Next Gen', 'NEXT GEN', 'Stealth', 'STEALTH',
    'Open World', 'OPEN WORLD', 'Gameplay', 'GAMEPLAY',
    'Realistic Graphics', 'REALISTIC', 'Full Match', 'FULL MATCH',
    'Real Life', 'REAL LIFE', 'BRUTAL', 'Ultra',
]


def recalculate_keyword_lifts(df):
    """Recalcula el impacto de keywords basándose SOLO en el campo title."""
    seen = set()
    results = []

    for kw in TITLE_KEYWORDS:
        kw_lower = kw.lower()
        if kw_lower in seen:
            continue
        seen.add(kw_lower)

        # Verificar SOLO en título, no en descripción ni tags
        mask = df['title'].str.contains(kw, case=False, na=False)
        with_kw = df[mask]
        without_kw = df[~mask]

        if len(with_kw) < 3:
            continue

        avg_with = with_kw['views'].mean()
        avg_without = without_kw['views'].mean()
        lift = ((avg_with - avg_without) / avg_without * 100) if avg_without > 0 else 0

        # Verificación: confirmar que cada video del grupo realmente tiene la keyword en el título
        false_positives = 0
        for _, row in with_kw.head(50).iterrows():
            if kw.lower() not in str(row['title']).lower():
                false_positives += 1

        results.append({
            'keyword': kw,
            'count': len(with_kw),
            'avg_views_with': round(avg_with, 0),
            'avg_views_without': round(avg_without, 0),
            'lift_pct': round(lift, 1),
            'false_positives': false_positives,
            'verified': false_positives == 0,
            'avg_avd_with': round(
                with_kw.loc[with_kw['avg_view_duration_sec'] > 0, 'avg_view_duration_sec'].mean(), 1
            ) if (with_kw['avg_view_duration_sec'] > 0).any() else 0,
            'avg_engagement_with': round(with_kw['engagement_rate'].mean(), 3),
        })

    return sorted(results, key=lambda x: x['lift_pct'], reverse=True)


# ============================================================
# PASO 3: Generar reporte HTML
# ============================================================

def generate_audit_report(df, audit_results, keyword_results, corrections):
    """Genera audit_report.html con los resultados."""
    ok_count = sum(1 for r in audit_results if r['status'] in ('OK', 'OK_DESC'))
    skip_count = sum(1 for r in audit_results if r['status'] == 'SKIP')
    review_count = sum(1 for r in audit_results if r['status'] == 'REVISAR')
    corrected_count = len(corrections)
    total = len(audit_results)

    review_items = [r for r in audit_results if r['status'] == 'REVISAR']

    html = f'''<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Audit Report — Joy Of Gaming</title>
<style>
body{{background:#0E1117;color:#E6EDF3;font-family:'Segoe UI',system-ui,sans-serif;padding:32px;line-height:1.6}}
h1{{color:#FF4444;border-bottom:2px solid #FF4444;padding-bottom:8px}}
h2{{color:#FDCB6E;margin-top:24px}}
.stat{{display:inline-block;background:#161B22;border:1px solid #30363D;border-radius:8px;padding:16px 24px;margin:8px;text-align:center}}
.stat .n{{font-size:28px;font-weight:700}}.stat .l{{font-size:11px;color:#8B949E;text-transform:uppercase}}
.ok .n{{color:#00B894}}.err .n{{color:#FF4444}}.fix .n{{color:#6C5CE7}}.skip .n{{color:#8B949E}}
table{{width:100%;border-collapse:collapse;margin-top:12px;font-size:12px}}
th{{background:#1C2333;color:#8B949E;padding:8px;text-align:left;font-size:10px;text-transform:uppercase}}
td{{padding:6px 8px;border-bottom:1px solid #30363D}}
tr:hover td{{background:#1C2333}}
.pos{{color:#00B894}}.neg{{color:#FF4444}}
</style>
</head>
<body>
<h1>Audit Report — Joy Of Gaming</h1>
<p>Generado: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>

<div>
<div class="stat ok"><div class="n">{ok_count}</div><div class="l">Bien clasificados</div></div>
<div class="stat err"><div class="n">{review_count}</div><div class="l">Necesitan revisión</div></div>
<div class="stat fix"><div class="n">{corrected_count}</div><div class="l">Corregidos automát.</div></div>
<div class="stat skip"><div class="n">{skip_count}</div><div class="l">Sin clasificar</div></div>
</div>
<p>Total: {total} videos. Precisión: {ok_count/max(total-skip_count,1)*100:.1f}%</p>

<h2>Errores Encontrados ({review_count})</h2>
<table>
<thead><tr><th>#</th><th>Título Real</th><th>Juego Asignado (INCORRECTO)</th><th>Género</th><th>Corrección</th></tr></thead>
<tbody>'''

    for i, r in enumerate(review_items):
        correction = corrections.get(r['idx'], {})
        corr_text = f"→ {correction.get('new_game', '?')}" if correction else "Manual"
        html += f"<tr><td>{i+1}</td><td>{r.get('title','')[:70]}</td><td class='neg'>{r.get('assigned_game','')}</td><td>{r.get('assigned_genre','')}</td><td class='pos'>{corr_text}</td></tr>"

    html += '''</tbody></table>

<h2>Análisis de Keywords (Solo Título, Verificado)</h2>
<table>
<thead><tr><th>Keyword</th><th>Videos</th><th>Avg Views (con)</th><th>Avg Views (sin)</th><th>Lift %</th><th>Verificado</th></tr></thead>
<tbody>'''

    for kw in keyword_results:
        cls = 'pos' if kw['lift_pct'] >= 0 else 'neg'
        verified = '✓' if kw['verified'] else f'⚠ {kw["false_positives"]} FP'
        html += f"<tr><td><b>{kw['keyword']}</b></td><td>{kw['count']}</td><td>{kw['avg_views_with']:,.0f}</td><td>{kw['avg_views_without']:,.0f}</td><td class='{cls}'>{'+' if kw['lift_pct']>0 else ''}{kw['lift_pct']}%</td><td>{verified}</td></tr>"

    html += '''</tbody></table>
</body></html>'''

    with open(AUDIT_HTML, 'w') as f:
        f.write(html)
    print(f"  Reporte generado: {AUDIT_HTML}")


# ============================================================
# PASO 4: Corregir y regenerar
# ============================================================

def main():
    print("=" * 60)
    print("  Audit Validator — Joy Of Gaming")
    print("=" * 60)

    df = pd.read_csv(INPUT_CSV)
    df['engagement_rate'] = np.where(
        df['views'] > 0,
        (df['likes'] + df['comments']) / df['views'] * 100,
        0,
    )

    # PASO 1: Auditar clasificaciones
    print("\n[1/4] Auditando clasificaciones...")
    audit_results = audit_classifications(df)

    ok = sum(1 for r in audit_results if r['status'] in ('OK', 'OK_DESC'))
    review = [r for r in audit_results if r['status'] == 'REVISAR']
    skip = sum(1 for r in audit_results if r['status'] == 'SKIP')
    print(f"  OK: {ok}, Revisar: {len(review)}, Skip: {skip}")

    # PASO 2: Intentar corregir automáticamente
    print("\n[2/4] Corrigiendo errores...")
    corrections = {}
    for r in review:
        idx = r['idx']
        row = df.iloc[idx]
        new_game, new_genre = try_reclassify(row['title'], row.get('tags', ''))

        # Solo corregir si el nuevo resultado es diferente y tiene sentido
        if new_game != row['game_name'] and new_game not in ('Unknown', 'Otro / No Identificado'):
            corrections[idx] = {
                'old_game': row['game_name'],
                'new_game': new_game,
                'old_genre': row['game_genre'],
                'new_genre': new_genre,
            }
            df.at[idx, 'game_name'] = new_game
            df.at[idx, 'game_genre'] = new_genre
            print(f"  CORREGIDO: '{row['game_name']}' → '{new_game}' | {row['title'][:60]}")

    print(f"  Correcciones automáticas: {len(corrections)}")

    # PASO 3: Recalcular keywords solo desde título
    print("\n[3/4] Recalculando keywords (solo título)...")
    keyword_results = recalculate_keyword_lifts(df)
    print(f"  Keywords analizadas: {len(keyword_results)}")
    for kw in keyword_results[:5]:
        print(f"    {kw['keyword']:20s} | {kw['count']:4d} vids | lift: {kw['lift_pct']:+.1f}% | verified: {kw['verified']}")

    # Guardar CSV corregido
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n  CSV corregido guardado: {OUTPUT_CSV}")

    # Generar reporte HTML
    print("\n[4/4] Generando reporte de auditoría...")
    generate_audit_report(df, audit_results, keyword_results, corrections)

    # Actualizar strategic_data.json con keywords corregidas
    print("\n  Actualizando strategic_data.json con keywords verificadas...")
    with open('strategic_data.json') as f:
        sdata = json.load(f)

    # Reemplazar keywords con las verificadas
    sdata['keywords'] = [kw for kw in keyword_results if kw['verified']]

    # Recalcular genre_all con datos corregidos
    genre_rows = []
    for g, grp in df.groupby('game_genre'):
        genre_rows.append({
            'genre': g, 'num_videos': len(grp),
            'total_views': int(grp['views'].sum()),
            'avg_views': round(grp['views'].mean(), 0),
            'avg_avd': round(
                grp.loc[grp['avg_view_duration_sec'] > 0, 'avg_view_duration_sec'].mean(), 1
            ) if (grp['avg_view_duration_sec'] > 0).any() else 0,
            'avg_engagement': round(grp['engagement_rate'].mean(), 3),
            'avg_retention_30s': round(
                grp.loc[grp['retention_30s'] > 0, 'retention_30s'].mean(), 1
            ) if (grp['retention_30s'] > 0).any() else None,
        })
    sdata['genre_all'] = sorted(genre_rows, key=lambda x: x['avg_views'], reverse=True)

    # Recalcular games_by_genre
    games_by_genre = {}
    for genre, ggrp in df.groupby('game_genre'):
        games = []
        for game, gg in ggrp.groupby('game_name'):
            games.append({
                'game': game, 'num_videos': len(gg),
                'total_views': int(gg['views'].sum()),
                'avg_views': round(gg['views'].mean(), 0),
            })
        games_by_genre[genre] = sorted(games, key=lambda x: x['total_views'], reverse=True)
    sdata['games_by_genre'] = games_by_genre

    # Recalcular top_games
    tg = df.groupby('game_name').agg(
        total_views=('views', 'sum'), num_videos=('video_id', 'count'),
        avg_views=('views', 'mean'), avg_avd=('avg_view_duration_sec', 'mean'),
        avg_engagement=('engagement_rate', 'mean'),
    ).reset_index().nlargest(15, 'total_views')
    sdata['top_games'] = tg.round(1).to_dict('records')

    with open('strategic_data.json', 'w') as f:
        json.dump(sdata, f, ensure_ascii=False)

    print(f"  strategic_data.json actualizado")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  RESUMEN DE AUDITORÍA")
    print(f"{'=' * 60}")
    print(f"  Videos auditados: {len(audit_results)}")
    print(f"  Bien clasificados: {ok} ({ok / max(len(audit_results) - skip, 1) * 100:.1f}%)")
    print(f"  Corregidos automáticamente: {len(corrections)}")
    print(f"  Pendientes de revisión manual: {len(review) - len(corrections)}")
    print(f"  Keywords verificadas: {sum(1 for k in keyword_results if k['verified'])}/{len(keyword_results)}")
    print(f"\n  Archivos generados:")
    print(f"    - {OUTPUT_CSV} (datos corregidos)")
    print(f"    - {AUDIT_HTML} (reporte visual)")
    print(f"    - strategic_data.json (actualizado)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
