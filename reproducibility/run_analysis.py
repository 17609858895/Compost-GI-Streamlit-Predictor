"""Re-run the archived compost analyses with explicit stages and fresh core fits."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import platform
import runpy
import sys
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import pandas as pd
import joblib
import generate_compost_figures as gcf
import major_revision_sensitivity as revision

PACKAGES = ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'lightgbm', 'catboost',
            'shap', 'matplotlib', 'seaborn', 'scipy', 'Pillow', 'openpyxl', 'joblib']

def verify_input():
    provenance = json.loads((ROOT / 'data/provenance.json').read_text(encoding='utf-8'))
    digest = hashlib.sha256(gcf.DATA_PATH.read_bytes()).hexdigest()
    if digest != provenance['workbook_sha256']:
        raise ValueError('Workbook SHA-256 differs from the archived analysis input.')
    return digest

def comparison(generated, reference):
    a, b = pd.read_csv(generated), pd.read_csv(reference)
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return {'shape_and_columns_match': False, 'pass': False}
    nonnumeric = a.select_dtypes(exclude='number').columns
    strings_match = a[nonnumeric].fillna('').equals(b[nonnumeric].fillna(''))
    numeric = a.select_dtypes(include='number').columns
    left, right = a[numeric].to_numpy(), b[numeric].to_numpy()
    delta = np.abs(left - right)
    return {'shape_and_columns_match': True, 'labels_match': strings_match,
            'max_absolute_numeric_difference': float(np.nanmax(delta)) if delta.size else 0,
            'pass': bool(strings_match and np.allclose(left, right, rtol=1e-6, atol=1e-6, equal_nan=True))}

def core(df, out):
    print('Re-fitting P0, P0+EC and P1 across five random and five proxy-group splits...', flush=True)
    gcf.evaluate_primary_feature_sets(df, force=True)
    # Force SHAP recomputation instead of silently accepting archived arrays.
    for suffix in ['_values.npy', '_meta.json']:
        (gcf.DATA_OUT / ('primary_shap_XGBoost_P1' + suffix)).unlink(missing_ok=True)
    print('Re-fitting day ablations, Segment A controls, P1 conformal calibration and SHAP...', flush=True)
    with (out / 'revision_run.txt').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
        revision.main()
    models = out / 'models'
    models.mkdir(exist_ok=True)
    data = df[df[gcf.TARGET].notna()].copy()
    for protocol in ['P0', 'P1']:
        pipe = gcf.build_pipe('XGBoost', gcf.FEATURE_SETS[protocol])
        pipe.fit(data[gcf.FEATURE_SETS[protocol]], data[gcf.TARGET])
        joblib.dump({'pipeline': pipe, 'features': gcf.FEATURE_SETS[protocol],
                     'target': gcf.TARGET, 'protocol': protocol,
                     'fit_rows': len(data), 'purpose': 'Full-data research refit; not an external validation model.'},
                    models / f'XGBoost_{protocol}.joblib')
    checks = {}
    p = 'primary_feature_sets_XGBoost_GI.csv'
    checks[p] = comparison(gcf.DATA_OUT / p, ROOT / 'reference_results/main' / p)
    for p in ['day_sensitivity_raw.csv', 'segment_a_p0_p1_control_raw.csv',
              'split_conformal_raw.csv', 'shap_group_normalisation.csv']:
        checks[p] = comparison(revision.OUT_DIR / p, ROOT / 'reference_results/revision' / p)
    recomputed = np.load(gcf.DATA_OUT / 'primary_shap_XGBoost_P1_values.npy')
    original = np.load(ROOT / 'reference_results/main/primary_shap_XGBoost_P1_values.npy')
    checks['P1_SHAP_array'] = {'shape': list(recomputed.shape),
       'max_absolute_difference': float(np.max(np.abs(recomputed - original))),
       'pass': bool(np.allclose(recomputed, original, rtol=1e-6, atol=1e-6))}
    return checks

def tables(df, out):
    dest = out / 'tables'
    dest.mkdir(exist_ok=True)
    perf = pd.read_csv(gcf.DATA_OUT / 'primary_feature_sets_XGBoost_GI.csv')
    perf.groupby(['feature_set', 'validation'])[['train_r2', 'test_r2', 'gap', 'test_rmse', 'test_mae']].mean().reset_index().to_csv(dest / 'Table1_protocol_comparison.csv', index=False)
    roles = {'GI': 'Primary maturity target', 'C/N': 'Secondary maturity target',
             'NH4-N': 'Supplementary maturity target', 'NO3-N': 'Supplementary maturity target',
             'CO2': 'Exploratory gas target', 'CH4': 'Exploratory gas target',
             'N2O': 'Exploratory gas target', 'NH3': 'Exploratory gas target'}
    rows = [{'target': col, 'valid_records': int(df[col].notna().sum()),
             'batch_proxy_groups': int(df.loc[df[col].notna(), 'batch_id'].nunique()),
             'role': roles[name], 'missing_percent': float(df[col].isna().mean() * 100)} for name, col in gcf.TARGETS.items()]
    pd.DataFrame(rows).to_csv(dest / 'TableS1_target_availability.csv', index=False)
    models = gcf.DATA_OUT / 'model_comparison_GI (%)_P1.csv'
    if models.exists():
        raw = pd.read_csv(models)
        raw.groupby(['validation', 'model'])[['train_r2', 'test_r2', 'gap', 'test_rmse', 'test_mae']].mean().reset_index().to_csv(dest / 'TableS3_model_comparison_all_nine.csv', index=False)
        raw = raw[raw['model'].isin(['XGBoost', 'HGB', 'LightGBM', 'CatBoost'])]
        raw.groupby(['validation', 'model'])[['train_r2', 'test_r2', 'gap', 'test_rmse', 'test_mae']].mean().reset_index().to_csv(dest / 'TableS3_four_leading_models.csv', index=False)
    params = {name: model.get_params() for name, model in gcf.model_library().items()}
    (dest / 'TableS2_model_parameters.json').write_text(json.dumps(params, indent=2, default=str) + '\n', encoding='utf-8')

def figures(df, out, only_audit=False):
    import create_fig4_revision_audit as audit
    audit.ANALYSIS = revision.OUT_DIR
    audit.OUT = gcf.OUT / 'Fig04_revision_audit'
    audit.SUB = audit.OUT / 'subfigures'
    audit.OUT.mkdir(parents=True, exist_ok=True)
    audit.SUB.mkdir(exist_ok=True)
    audit.main()
    if only_audit:
        return
    if not (gcf.DATA_OUT / 'overfit_reduction_candidates.csv').exists():
        raise FileNotFoundError('Run --stage regularisation before plotting Fig. 5g-l.')
    for label, func in [
        ('source data-audit figure', gcf.fig1_data_audit),
        ('Fig. 2', gcf.fig2_target_distribution), ('Fig. 3', gcf.fig3_correlation),
        ('Fig. 5', gcf.fig4_model_comparison), ('Fig. 6', gcf.fig5_pred_vs_obs),
        ('Fig. 7', gcf.fig6_optimism_gap), ('Fig. 8', gcf.fig7_residual_reliability),
        ('Fig. 10', gcf.fig8_shap_importance), ('Fig. 11', gcf.fig9_pdp_ale),
        ('Fig. S1', gcf.fig_s1_learning_curves), ('Fig. S2', gcf.fig_s3_gas_targets),
        ('Fig. S3', gcf.fig10_feature_ablation)]:
        print('Plotting ' + label, flush=True)
        func(df)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['core', 'models', 'targets', 'regularisation', 'tables', 'audit-figure', 'figures', 'all'], default='core')
    parser.add_argument('--output', type=Path, default=ROOT / 'generated')
    args = parser.parse_args()
    out = args.output.resolve()
    protected = [ROOT / p for p in ['data', 'scripts', 'reference_results', 'validation']]
    if out == ROOT or any(out == p or p in out.parents for p in protected):
        parser.error('Output must be separate from archived inputs, scripts, references and validation.')
    out.mkdir(parents=True, exist_ok=True)
    gcf.OUT = out / 'figures_compost'
    gcf.DATA_OUT = gcf.OUT / '_data'
    gcf.DATA_OUT.mkdir(parents=True, exist_ok=True)
    revision.OUT_DIR = out / 'revision'
    revision.OUT_DIR.mkdir(exist_ok=True)
    digest = verify_input()
    df = gcf.load_compost_data()
    assert len(df) == 4248 and df[gcf.TARGET].notna().sum() == 2571
    assert df.loc[df[gcf.TARGET].notna(), 'batch_id'].nunique() == 326
    df.to_csv(out / 'compost_calibrated.csv', index=False, encoding='utf-8-sig')
    print(f'Input verified: {len(df)} records, 2571 GI labels, 326 GI proxy groups.', flush=True)
    checks = {}
    if args.stage in ['core', 'all']:
        checks.update(core(df, out))
    if args.stage in ['models', 'all']:
        print('Re-fitting the nine-model P1 comparison...', flush=True)
        gcf.evaluate_models(df, force=True)
        name = 'model_comparison_GI (%)_P1.csv'
        checks[name] = comparison(gcf.DATA_OUT / name, ROOT / 'reference_results/main' / name)
    if args.stage in ['targets', 'all']:
        print('Re-fitting all supplementary targets...', flush=True)
        gcf.target_perf(df, force=True)
        name = 'target_random_group_XGBoost.csv'
        checks[name] = comparison(gcf.DATA_OUT / name, ROOT / 'reference_results/main' / name)
    if args.stage in ['regularisation', 'all']:
        print('Re-fitting the recovered 11-candidate regularisation experiment...', flush=True)
        with (out / 'regularisation_run.txt').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
            runpy.run_path(str(ROOT / 'scripts/run_regularisation.py'), run_name='__main__')
        name = 'overfit_reduction_candidates.csv'
        checks[name] = comparison(gcf.DATA_OUT / name, ROOT / 'reference_results/main' / name)
    if args.stage in ['tables', 'all', 'core']:
        tables(df, out)
    if args.stage in ['figures', 'audit-figure', 'all']:
        gcf.configure_style()
        figures(df, out, only_audit=args.stage == 'audit-figure')
    report = {'stage': args.stage, 'python': platform.python_version(),
              'platform': platform.system(), 'versions': {p: version(p) for p in PACKAGES},
              'workbook_sha256': digest, 'records': len(df), 'gi_records': 2571,
              'gi_proxy_groups': 326, 'seeds': gcf.SEEDS, 'comparisons': checks,
              'all_comparisons_passed': all(v['pass'] for v in checks.values()) if checks else None,
              'limitations': ['P0-specific SHAP/conformal results were not reported in the manuscript.',
                  'Workflow artwork and the Streamlit screenshot are not statistical plots.',
                  'Original row-level study identities and provisional field mapping remain unresolved.']}
    (out / f'verification_{args.stage}.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'stage': args.stage, 'comparisons': checks}, indent=2), flush=True)
    if checks and not report['all_comparisons_passed']:
        raise SystemExit('Recomputed results differ from archived reference values; inspect verification report.')

if __name__ == '__main__':
    main()
