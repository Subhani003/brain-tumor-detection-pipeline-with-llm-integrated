import React, { useState, useEffect, useCallback, useRef, useMemo, Suspense } from 'react';
import * as THREE from 'three';
import {
    Container, Box, Typography, Button, Paper, Grid, CircularProgress,
    Alert, Card, CardContent, CardMedia, Chip, AppBar, Toolbar,
    Tabs, Tab, IconButton, Tooltip, Dialog, DialogContent, DialogTitle, DialogActions,
    Drawer, TextField, Fab
} from '@mui/material';
import ChatIcon from '@mui/icons-material/Chat';
import CloseIcon from '@mui/icons-material/Close';
import SendIcon from '@mui/icons-material/Send';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import LinkedInIcon from '@mui/icons-material/LinkedIn';
import DownloadIcon from '@mui/icons-material/Download';
import ScienceIcon from '@mui/icons-material/Science';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import GroupsIcon from '@mui/icons-material/Groups';
import VisibilityIcon from '@mui/icons-material/Visibility';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';
import ShieldIcon from '@mui/icons-material/Shield';
import Dither from './Dither';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Float, useGLTF } from '@react-three/drei';
import { EffectComposer, Bloom } from '@react-three/postprocessing';
import { OrbitControls as ThreeOrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { useT } from './i18n';
import './App.css';


/* ── Model Comparison Chart (side-by-side per-model confidence) ── */
function ModelComparisonChart({ r }) {
    const { t } = useT();
    if (!r || !r.models) return null;

    // Order models consistently and pick distinct accent colors per model.
    // v2 keys take priority; v1 keys kept for backward-compat fallback.
    const MODEL_ORDER = [
        'convnext_tiny', 'efficientnet_b3', 'resnet50',
        'densenet169',   'efficientnetb3',   // v1 legacy
    ];
    const MODEL_ACCENT = {
        // v2
        convnext_tiny:    '#a78bfa',
        efficientnet_b3:  '#f5a623',
        resnet50:         '#38ef7d',
        // v1 legacy
        densenet169:      '#667eea',
        efficientnetb3:   '#f5a623',
    };
    const CLASS_ACCENT = {
        'Glioma':     '#f5576c',
        'Meningioma': '#667eea',
        'No Tumor':   '#38ef7d',
        'Pituitary':  '#f5a623',
    };

    const entries = MODEL_ORDER
        .filter(k => r.models[k])
        .map(k => [k, r.models[k]]);
    if (entries.length === 0) return null;

    const bestKey = r.best_model;
    const agreement = r.agreement || {};
    const unanimous = !!agreement.unanimous;
    const agreeing = agreement.agreeing_count || 0;
    const total = agreement.total_models || entries.length;

    return (
        <Paper className="glass-card" elevation={0}
            sx={{ p: 3, mt: 2.5, borderRadius: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 1, mb: 0.5 }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 700, color: '#667eea' }}>
                    {t('mc.title')}
                </Typography>
                <Chip
                    label={unanimous ? `${agreeing}/${total} ${t('mc.unanimous')}` : `${agreeing}/${total} ${t('mc.agree')}`}
                    size="small"
                    sx={{
                        fontWeight: 700, letterSpacing: 0.5,
                        bgcolor: unanimous ? 'rgba(56,239,125,0.18)' : 'rgba(245,166,35,0.18)',
                        color: unanimous ? '#38ef7d' : '#f5a623',
                        border: `1px solid ${unanimous ? 'rgba(56,239,125,0.45)' : 'rgba(245,166,35,0.45)'}`,
                    }}
                />
            </Box>
            <Typography variant="caption" sx={{ display: 'block', opacity: 0.7, mb: 2 }}>
                {t('mc.subtitle')}
            </Typography>

            {entries.map(([key, m]) => {
                const isBest = key === bestKey;
                const accent = MODEL_ACCENT[key] || '#94a3b8';
                const u = r.uncertainty || {};
                const winnerClass = (isBest && u.class_name) ? u.class_name : m.class;
                const conf = (isBest && typeof u.mean_confidence === 'number')
                    ? u.mean_confidence
                    : (Number(m.confidence) || 0);
                const classColor = CLASS_ACCENT[winnerClass] || '#94a3b8';
                let band = null;
                if (isBest && typeof u.epistemic === 'number') {
                    if (u.epistemic < 0.005) band = { label: t('mc.band.verylow'), color: '#38ef7d' };
                    else if (u.epistemic < 0.05) band = { label: t('mc.band.low'), color: '#a3e635' };
                    else if (u.epistemic < 0.10) band = { label: t('mc.band.moderate'), color: '#f5a623' };
                    else band = { label: t('mc.band.high'), color: '#f5576c' };
                }
                return (
                    <Box key={key} sx={{
                        mt: 1.2, p: 1.3, borderRadius: 2,
                        background: isBest ? 'rgba(255,255,255,0.06)' : 'rgba(255,255,255,0.025)',
                        border: `1px solid ${isBest ? accent + '88' : 'rgba(255,255,255,0.06)'}`,
                        boxShadow: isBest ? `0 0 0 2px ${accent}22` : 'none',
                        transition: 'all .25s ease',
                    }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 0.5 }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.8 }}>
                                <Box sx={{ width: 8, height: 8, borderRadius: '50%', background: accent }} />
                                <Typography variant="body2" sx={{ fontWeight: 700, color: '#fff' }}>
                                    {m.name}
                                </Typography>
                                {isBest && (
                                    <Chip label={t('mc.best_badge')} size="small" sx={{
                                        height: 16, fontSize: 9, ml: 0.4, fontWeight: 700, letterSpacing: 0.5,
                                        bgcolor: `${accent}33`, color: accent,
                                    }} />
                                )}
                                {!isBest && (
                                    <Chip label={t('mc.tta_badge')} size="small" sx={{
                                        height: 16, fontSize: 9, ml: 0.4, fontWeight: 600, letterSpacing: 0.5,
                                        bgcolor: 'rgba(255,255,255,0.06)', color: 'rgba(255,255,255,0.6)',
                                    }} />
                                )}
                            </Box>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.8 }}>
                                <Chip label={`${t('mc.predicts')} ${winnerClass}`} size="small" sx={{
                                    height: 22, fontSize: 12, fontWeight: 700,
                                    bgcolor: `${classColor}22`, color: classColor,
                                    border: `1px solid ${classColor}55`,
                                }} />
                                <Typography variant="body2" sx={{ fontWeight: 700, color: accent, minWidth: 56, textAlign: 'right' }}>
                                    {(conf * 100).toFixed(1)}%
                                </Typography>
                            </Box>
                        </Box>

                        {isBest && band && (
                            <Box sx={{
                                display: 'flex', flexWrap: 'wrap', gap: 0.8, mt: 0.8, px: 0.5,
                                alignItems: 'center',
                            }}>
                                <Chip label={band.label} size="small" sx={{
                                    height: 18, fontSize: 10, fontWeight: 700, letterSpacing: 0.3,
                                    bgcolor: `${band.color}25`, color: band.color,
                                    border: `1px solid ${band.color}55`,
                                }} />
                                <Typography variant="caption" sx={{ fontSize: 10, opacity: 0.75 }}>
                                    σ = {u.epistemic.toFixed(4)}
                                </Typography>
                                {typeof u.ci_lower === 'number' && typeof u.ci_upper === 'number' && (
                                    <Typography variant="caption" sx={{ fontSize: 10, opacity: 0.65 }}>
                                        · 95% CI [{u.ci_lower.toFixed(3)}, {u.ci_upper.toFixed(3)}]
                                    </Typography>
                                )}
                            </Box>
                        )}
                    </Box>
                );
            })}

            {!unanimous && (
                <Typography variant="caption" sx={{
                    display: 'block', mt: 1.5, opacity: 0.75, fontStyle: 'italic',
                    color: '#f5a623',
                }}>
                    Models disagree — review recommended. See OOD / Focus-Crop signals in Analysis Results.
                </Typography>
            )}
        </Paper>
    );
}


/* ── Probability Bars Card (per-class softmax, shown below Model Comparison in the left panel) ── */
function ProbabilityBarsCard({ r }) {
    const u = r?.uncertainty || {};
    // MC Dropout-averaged probabilities live in `uncertainty.probabilities`
    // (20 stochastic forward passes). Fall back to TTA probs if missing.
    const probs = u.probabilities || r?.prediction?.probabilities;
    if (!probs) return null;
    const CLASS_COLOR = {
        'Glioma':     '#f5576c',
        'Meningioma': '#667eea',
        'No Tumor':   '#38ef7d',
        'Pituitary':  '#f5a623',
    };
    const bestModel = r.best_model || '';
    const bestAcc = r.models?.[bestModel]?.accuracy;
    const epi = u.epistemic;
    const ale = u.aleatoric;
    const total = u.total_uncertainty;
    const ciL = u.ci_lower;
    const ciU = u.ci_upper;
    // Verdict band based on epistemic std
    let band = null;
    if (typeof epi === 'number') {
        if (epi < 0.005) band = { label: 'Very low uncertainty', color: '#38ef7d' };
        else if (epi < 0.05) band = { label: 'Low uncertainty', color: '#a3e635' };
        else if (epi < 0.10) band = { label: 'Moderate uncertainty', color: '#f5a623' };
        else band = { label: 'High uncertainty — review', color: '#f5576c' };
    }
    return (
        <Paper className="glass-card" elevation={0} sx={{ p: 2.5, mt: 2.5, borderRadius: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 1, mb: 0.5 }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 700, color: '#667eea' }}>
                    Final Probabilities (MC Dropout)
                </Typography>
                {bestModel && (
                    <Chip label={`Best: ${r.models?.[bestModel]?.name || bestModel}${bestAcc ? ' · ' + bestAcc : ''}`}
                        size="small"
                        sx={{
                            fontWeight: 700, letterSpacing: 0.3,
                            bgcolor: 'rgba(102,126,234,0.18)',
                            color: '#a3b3ff',
                            border: '1px solid rgba(102,126,234,0.45)',
                        }} />
                )}
            </Box>
            <Typography variant="caption" sx={{ display: 'block', opacity: 0.7, mb: 1.5 }}>
                Best-model softmax averaged across 20 stochastic forward passes with dropout enabled (Monte Carlo Dropout). Spread on non-winning classes reflects epistemic uncertainty. When the model is genuinely confident, bars saturate to 100/0/0/0 — use the metrics row below to read uncertainty even when the bars look flat.
            </Typography>

            {/* Uncertainty metric strip — visible even when bars saturate */}
            {(typeof epi === 'number' || band) && (
                <Box sx={{
                    display: 'flex', flexWrap: 'wrap', gap: 1, mb: 1.5, p: 1.2, borderRadius: 2,
                    background: 'rgba(255,255,255,0.04)',
                    border: '1px solid rgba(255,255,255,0.07)',
                }}>
                    {band && (
                        <Chip label={band.label} size="small" sx={{
                            fontWeight: 700, letterSpacing: 0.3, fontSize: 11,
                            bgcolor: `${band.color}25`, color: band.color,
                            border: `1px solid ${band.color}55`,
                        }} />
                    )}
                    {typeof epi === 'number' && (
                        <Typography variant="caption" sx={{ fontSize: 11, opacity: 0.85 }}>
                            <strong>epistemic</strong> σ = {epi.toFixed(4)}
                        </Typography>
                    )}
                    {typeof ale === 'number' && (
                        <Typography variant="caption" sx={{ fontSize: 11, opacity: 0.85 }}>
                            <strong>aleatoric</strong> = {ale.toFixed(4)}
                        </Typography>
                    )}
                    {typeof total === 'number' && (
                        <Typography variant="caption" sx={{ fontSize: 11, opacity: 0.85 }}>
                            <strong>total</strong> = {total.toFixed(4)}
                        </Typography>
                    )}
                    {typeof ciL === 'number' && typeof ciU === 'number' && (
                        <Typography variant="caption" sx={{ fontSize: 11, opacity: 0.85 }}>
                            <strong>95% CI</strong> [{ciL.toFixed(3)}, {ciU.toFixed(3)}]
                        </Typography>
                    )}
                </Box>
            )}
            {Object.entries(probs).map(([name, prob]) => {
                const color = CLASS_COLOR[name] || '#667eea';
                return (
                    <Box key={name} sx={{ mb: 1.5 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.4 }}>
                            <Typography variant="body2" sx={{ fontWeight: 600, color: '#fff' }}>{name}</Typography>
                            <Typography variant="body2" sx={{ fontWeight: 700, color: color }}>
                                {(prob * 100).toFixed(1)}%
                            </Typography>
                        </Box>
                        <Box sx={{ width: '100%', height: 9, bgcolor: 'rgba(255,255,255,0.06)', borderRadius: 2, overflow: 'hidden' }}>
                            <Box sx={{
                                width: `${prob * 100}%`, height: '100%',
                                background: `linear-gradient(90deg, ${color}88, ${color})`,
                                borderRadius: 2,
                                transition: 'width 0.5s ease',
                                boxShadow: `0 0 6px ${color}55`,
                            }} />
                        </Box>
                    </Box>
                );
            })}
        </Paper>
    );
}


/* ── Navigation toggle (Analysis ↔ Metrics) ────────────── */
function NavigationToggle({ view, setView }) {
    const { t } = useT();
    const make = (key, label) => (
        <Box
            onClick={() => setView(key)}
            sx={{
                px: 1.6, py: 0.5, borderRadius: 1.5, cursor: 'pointer',
                fontSize: 12, fontWeight: 700, letterSpacing: 0.5,
                color: view === key ? '#fff' : 'rgba(255,255,255,0.55)',
                background: view === key
                    ? 'linear-gradient(135deg,#667eea 0%,#764ba2 100%)'
                    : 'rgba(255,255,255,0.06)',
                border: `1px solid ${view === key ? 'transparent' : 'rgba(255,255,255,0.12)'}`,
                transition: 'all .15s ease',
                userSelect: 'none',
            }}
        >
            {label}
        </Box>
    );
    return (
        <Box sx={{ display: 'flex', gap: 0.5, alignItems: 'center' }}>
            {make('analysis', t('nav.analysis'))}
            {make('metrics',  t('nav.metrics'))}
            {make('atlas',    t('nav.atlas'))}
            {make('pipeline', t('nav.pipeline'))}
        </Box>
    );
}


/* ── Metrics Page: per-model comparison (accuracy + efficiency) ── */
function MetricsPage() {
    const { t } = useT();
    const [data, setData]       = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError]     = useState(null);

    const fetchMetrics = useCallback(async () => {
        setLoading(true); setError(null);
        try {
            const resp = await axios.get(`${API_URL}/api/metrics`, { timeout: 60000 });
            setData(resp.data);
        } catch (e) {
            setError(e.response?.data?.error || e.message || 'Network error');
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => { fetchMetrics(); }, [fetchMetrics]);

    if (loading && !data) {
        return (
            <Box sx={{ textAlign: 'center', py: 8 }}>
                <CircularProgress sx={{ color: '#667eea' }} />
                <Typography variant="body2" sx={{ mt: 2, opacity: 0.7 }}>
                    {t('metrics.loading')}
                </Typography>
            </Box>
        );
    }
    if (error) {
        return (
            <Alert severity="error" sx={{ mt: 4 }}>
                Could not load metrics: {error}
            </Alert>
        );
    }
    if (!data) return null;

    const ACCENT = {
        // v2 models
        convnext_tiny:    '#a78bfa',
        efficientnet_b3:  '#f5a623',
        resnet50:         '#38ef7d',
        // legacy keys (in case /api/metrics is still on the old report files)
        densenet169:      '#667eea',
        efficientnetb3:   '#f5a623',
    };
    const CLASS_KEYS = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'];

    // Find a global max for accuracy axis (0..1)
    const accs = data.models
        .map(m => m.test_metrics?.test_accuracy)
        .filter(v => typeof v === 'number');
    const hasMetrics = accs.length > 0;

    return (
        <Box>
            {/* Top: device + caveat */}
            <Paper className="glass-card" elevation={0} sx={{ p: 3, mb: 3 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 1 }}>
                    <Typography variant="h5" sx={{ fontWeight: 700, color: '#667eea' }}>
                        {t('metrics.title')}
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Chip label={data.device}    size="small" sx={{ bgcolor: 'rgba(102,126,234,0.18)', color: '#a3b3ff', fontWeight: 700 }} />
                        {data.cuda_name && (
                            <Chip label={data.cuda_name} size="small" sx={{ bgcolor: 'rgba(56,239,125,0.18)', color: '#38ef7d', fontWeight: 700 }} />
                        )}
                        <Button onClick={() => fetchMetrics()} size="small" sx={{ color: '#a3b3ff' }}>{t('metrics.refresh')}</Button>
                    </Box>
                </Box>
                <Typography variant="caption" sx={{ display: 'block', opacity: 0.75, mt: 1 }}>
                    {data.note}
                </Typography>
            </Paper>

            {/* Comparison table */}
            <Paper className="glass-card" elevation={0} sx={{ p: 3, mb: 3 }}>
                <Typography variant="h6" sx={{ fontWeight: 700, color: '#667eea', mb: 2 }}>{t('metrics.overview')}</Typography>
                <Box sx={{ overflowX: 'auto' }}>
                    <Box component="table" sx={{
                        width: '100%', borderCollapse: 'collapse',
                        '& th, & td': { textAlign: 'left', py: 1.1, px: 1.5, fontSize: 13, borderBottom: '1px solid rgba(255,255,255,0.06)' },
                        '& th': { fontWeight: 700, color: '#a3b3ff', letterSpacing: 0.5, fontSize: 11, textTransform: 'uppercase' },
                    }}>
                        <thead><tr>
                            <th>{t('metrics.col.model')}</th><th>{t('metrics.col.params')}</th><th>{t('metrics.col.size')}</th>
                            <th>{t('metrics.col.single')}</th><th>{t('metrics.col.tta')}</th>
                            <th>{t('metrics.col.acc')}</th><th>{t('metrics.col.balanced')}</th>
                        </tr></thead>
                        <tbody>
                        {data.models.map(m => {
                            const lat = m.latency || {};
                            const tm  = m.test_metrics;
                            return (
                                <tr key={m.key}>
                                    <td>
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                            <Box sx={{ width: 10, height: 10, borderRadius: '50%', background: ACCENT[m.key] || '#94a3b8' }} />
                                            <strong>{m.name}</strong>
                                        </Box>
                                    </td>
                                    <td>{m.parameters_M}</td>
                                    <td>{m.size_mb ?? '—'}</td>
                                    <td>{lat.single_pass_ms_mean?.toFixed(1) ?? '—'} <span style={{ opacity: 0.5 }}>±{lat.single_pass_ms_std?.toFixed(1)}</span></td>
                                    <td>{lat.tta5_ms_mean?.toFixed(1) ?? '—'} <span style={{ opacity: 0.5 }}>±{lat.tta5_ms_std?.toFixed(1)}</span></td>
                                    <td>{tm ? `${(tm.test_accuracy * 100).toFixed(2)}%` : <em style={{ opacity: 0.5 }}>not measured</em>}</td>
                                    <td>{tm ? `${(tm.test_balanced_acc * 100).toFixed(2)}%` : '—'}</td>
                                </tr>
                            );
                        })}
                        </tbody>
                    </Box>
                </Box>
            </Paper>

            {/* Ensemble agreement + soft-vote (only when v2 metrics are present) */}
            {data.source === 'v2' && data.ensemble && (
                <Paper className="glass-card" elevation={0} sx={{ p: 3, mb: 3 }}>
                    <Typography variant="h6" sx={{ fontWeight: 700, color: '#667eea', mb: 0.5 }}>
                        {t('metrics.ensemble.title') || 'Ensemble & agreement'}
                    </Typography>
                    <Typography variant="caption" sx={{ display: 'block', opacity: 0.7, mb: 2 }}>
                        {t('metrics.ensemble.subtitle') || 'Soft-vote = average of the three softmax outputs. Agreement = how often two (or all three) models pick the same top-1 class on the locked test set.'}
                    </Typography>
                    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' }, gap: 1.5 }}>
                        {[
                            { label: 'Soft-vote accuracy', value: `${((data.ensemble.soft_vote_test_acc || 0) * 100).toFixed(2)}%`, color: '#a78bfa' },
                            { label: 'Soft-vote balanced', value: `${((data.ensemble.soft_vote_balanced_acc || 0) * 100).toFixed(2)}%`, color: '#22d3ee' },
                            { label: '3-way agreement',    value: `${(data.ensemble.three_way_agreement_pct || 0).toFixed(2)}%`, color: '#38ef7d' },
                            { label: 'Test set size',      value: '2,114', color: '#f5a623' },
                        ].map((s, i) => (
                            <Box key={i} sx={{
                                p: 1.8, borderRadius: 2,
                                background: `linear-gradient(135deg, ${s.color}11, ${s.color}06)`,
                                border: `1px solid ${s.color}44`,
                            }}>
                                <Typography sx={{ fontSize: 10.5, letterSpacing: 1, opacity: 0.75, textTransform: 'uppercase', fontWeight: 700, color: s.color }}>
                                    {s.label}
                                </Typography>
                                <Typography sx={{ fontSize: 22, fontWeight: 800, color: '#fff', lineHeight: 1.2, mt: 0.3 }}>
                                    {s.value}
                                </Typography>
                            </Box>
                        ))}
                    </Box>
                    {data.ensemble.pairwise_agreement_pct && (
                        <Box sx={{ mt: 2.5 }}>
                            <Typography variant="caption" sx={{ display: 'block', opacity: 0.7, mb: 1, letterSpacing: 0.5, textTransform: 'uppercase', fontSize: 10 }}>
                                Pairwise agreement
                            </Typography>
                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                                {Object.entries(data.ensemble.pairwise_agreement_pct).map(([pair, pct]) => {
                                    const [a, b] = pair.split('__');
                                    return (
                                        <Box key={pair} sx={{
                                            px: 1.5, py: 0.6, borderRadius: 1.5,
                                            background: 'rgba(255,255,255,0.04)',
                                            border: '1px solid rgba(255,255,255,0.08)',
                                            fontSize: 12, color: '#cbd5e1',
                                        }}>
                                            <strong style={{ color: ACCENT[a] || '#a3b3ff' }}>{a}</strong>
                                            <span style={{ opacity: 0.5, margin: '0 6px' }}>↔</span>
                                            <strong style={{ color: ACCENT[b] || '#a3b3ff' }}>{b}</strong>
                                            <span style={{ marginLeft: 8, fontWeight: 700, color: '#fff' }}>{pct.toFixed(2)}%</span>
                                        </Box>
                                    );
                                })}
                            </Box>
                        </Box>
                    )}
                </Paper>
            )}

            {/* Inference latency bar chart — how long each model takes to classify one image */}
            {(() => {
                const lats = data.models
                    .map(m => ({
                        key: m.key,
                        name: m.name,
                        accent: ACCENT[m.key] || '#94a3b8',
                        single: m.latency?.single_pass_ms_mean,
                        tta:    m.latency?.tta5_ms_mean,
                    }))
                    .filter(p => typeof p.tta === 'number');
                if (lats.length === 0) return null;
                const maxTta = Math.max(...lats.map(p => p.tta));
                return (
                    <Paper className="glass-card" elevation={0} sx={{ p: 3, mb: 3 }}>
                        <Typography variant="h6" sx={{ fontWeight: 700, color: '#667eea', mb: 0.5 }}>
                            {t('metrics.latency.title')}
                        </Typography>
                        <Typography variant="caption" sx={{ display: 'block', opacity: 0.7, mb: 2 }}>
                            {t('metrics.latency.subtitle')}
                        </Typography>
                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                            {lats.map(p => {
                                const w = (p.tta / maxTta) * 100;
                                const singleW = typeof p.single === 'number' ? (p.single / maxTta) * 100 : null;
                                return (
                                    <Box key={p.key}>
                                        <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                                            <Typography variant="body2" sx={{ fontWeight: 700, color: p.accent }}>
                                                {p.name}
                                            </Typography>
                                            <Typography variant="caption" sx={{ opacity: 0.8 }}>
                                                <strong>{p.tta.toFixed(1)} ms</strong> · {(1000 / p.tta).toFixed(1)} img/s
                                                {typeof p.single === 'number' && (
                                                    <span style={{ opacity: 0.6 }}> · single {p.single.toFixed(1)} ms</span>
                                                )}
                                            </Typography>
                                        </Box>
                                        <Box sx={{
                                            position: 'relative', height: 18, borderRadius: 1.5,
                                            background: 'rgba(255,255,255,0.05)', overflow: 'hidden',
                                            border: `1px solid ${p.accent}33`,
                                        }}>
                                            <Box sx={{
                                                width: `${w}%`, height: '100%',
                                                background: `linear-gradient(90deg, ${p.accent}aa, ${p.accent})`,
                                                boxShadow: `0 0 6px ${p.accent}55`,
                                                transition: 'width 0.6s ease',
                                            }} />
                                            {singleW != null && (
                                                <Box sx={{
                                                    position: 'absolute', left: `${singleW}%`, top: -2, bottom: -2,
                                                    width: 2, background: '#fff', opacity: 0.85,
                                                }} title={`single pass ${p.single.toFixed(1)} ms`} />
                                            )}
                                        </Box>
                                    </Box>
                                );
                            })}
                        </Box>
                    </Paper>
                );
            })()}

            {/* Accuracy bar chart */}
            {hasMetrics && (
                <Paper className="glass-card" elevation={0} sx={{ p: 3, mb: 3 }}>
                    <Typography variant="h6" sx={{ fontWeight: 700, color: '#667eea', mb: 2 }}>
                        {t('metrics.accuracy.title')}
                    </Typography>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                        {data.models.map(m => {
                            const tm = m.test_metrics;
                            if (!tm) return null;
                            const accent = ACCENT[m.key] || '#94a3b8';
                            const a  = tm.test_accuracy * 100;
                            const b  = tm.test_balanced_acc * 100;
                            return (
                                <Box key={m.key}>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
                                        <Typography variant="body2" sx={{ fontWeight: 700 }}>{m.name}</Typography>
                                        <Typography variant="caption" sx={{ opacity: 0.7 }}>
                                            acc {a.toFixed(2)}% &middot; balanced {b.toFixed(2)}%
                                        </Typography>
                                    </Box>
                                    <Box sx={{ position: 'relative', height: 12, borderRadius: 1.5, background: 'rgba(255,255,255,0.05)', overflow: 'hidden' }}>
                                        <Box sx={{
                                            position: 'absolute', left: 0, top: 0, bottom: 0,
                                            width: `${a}%`,
                                            background: `linear-gradient(90deg, ${accent}88, ${accent})`,
                                            boxShadow: `0 0 8px ${accent}66`,
                                        }} />
                                        <Box sx={{
                                            position: 'absolute', left: `${b}%`, top: -2, bottom: -2,
                                            width: 2, background: '#fff', opacity: 0.85,
                                        }} title={`balanced ${b.toFixed(2)}%`} />
                                    </Box>
                                </Box>
                            );
                        })}
                    </Box>
                    <Typography variant="caption" sx={{ display: 'block', mt: 1.5, opacity: 0.6 }}>
                        {t('metrics.accuracy.hint')}
                    </Typography>
                </Paper>
            )}

            {/* Per-class recall grouped bar chart */}
            {hasMetrics && (
                <Paper className="glass-card" elevation={0} sx={{ p: 3, mb: 3 }}>
                    <Typography variant="h6" sx={{ fontWeight: 700, color: '#667eea', mb: 0.5 }}>
                        {t('metrics.perclass.title')}
                    </Typography>
                    <Typography variant="caption" sx={{ display: 'block', opacity: 0.7, mb: 2 }}>
                        {t('metrics.perclass.subtitle')}
                    </Typography>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                        {CLASS_KEYS.map(cls => (
                            <Box key={cls}>
                                <Typography variant="caption" sx={{ opacity: 0.7, letterSpacing: 0.5, textTransform: 'uppercase', fontSize: 10 }}>
                                    {cls}
                                </Typography>
                                <Box sx={{ display: 'flex', gap: 1, mt: 0.5 }}>
                                    {data.models.map(m => {
                                        const accent = ACCENT[m.key] || '#94a3b8';
                                        const v = m.test_metrics?.per_class?.[cls];
                                        // per_class[cls] is now either a number (legacy) or an object {recall, precision, f1, support}
                                        const r = typeof v === 'number' ? v
                                            : (typeof v?.recall === 'number' ? v.recall : null);
                                        const pct = r != null ? r * 100 : 0;
                                        return (
                                            <Box key={m.key} sx={{ flex: 1 }}>
                                                <Box sx={{
                                                    position: 'relative', height: 26,
                                                    borderRadius: 1.2, overflow: 'hidden',
                                                    background: 'rgba(255,255,255,0.05)',
                                                    border: `1px solid ${accent}33`,
                                                }}>
                                                    <Box sx={{
                                                        width: `${pct}%`, height: '100%',
                                                        background: `linear-gradient(90deg, ${accent}aa, ${accent})`,
                                                        transition: 'width 0.6s ease',
                                                    }} />
                                                    <Typography variant="caption" sx={{
                                                        position: 'absolute', right: 6, top: '50%',
                                                        transform: 'translateY(-50%)', fontWeight: 700,
                                                        fontSize: 11, color: '#fff',
                                                        textShadow: '0 0 4px rgba(0,0,0,0.6)',
                                                    }}>
                                                        {pct ? `${pct.toFixed(2)}%` : '—'}
                                                    </Typography>
                                                </Box>
                                                <Typography variant="caption" sx={{ display: 'block', opacity: 0.6, fontSize: 10, mt: 0.3 }}>
                                                    {m.name}
                                                </Typography>
                                            </Box>
                                        );
                                    })}
                                </Box>
                            </Box>
                        ))}
                    </Box>
                </Paper>
            )}

            {/* Confusion matrices (v2 only — figures saved by src/eval_v2.py) */}
            {data.source === 'v2' && data.models.some(m => m.test_metrics?.confusion_png) && (
                <Paper className="glass-card" elevation={0} sx={{ p: 3, mb: 3 }}>
                    <Typography variant="h6" sx={{ fontWeight: 700, color: '#667eea', mb: 0.5 }}>
                        {t('metrics.confusion.title') || 'Confusion matrices'}
                    </Typography>
                    <Typography variant="caption" sx={{ display: 'block', opacity: 0.7, mb: 2 }}>
                        {t('metrics.confusion.subtitle') || 'Per-model confusion on the locked 2 114-image test set. Diagonal = correct classifications. The 4th panel shows the soft-vote ensemble.'}
                    </Typography>
                    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)' }, gap: 1.5 }}>
                        {data.models.map(m => {
                            const fn = m.test_metrics?.confusion_png;
                            if (!fn) return null;
                            const accent = ACCENT[m.key] || '#94a3b8';
                            const acc = m.test_metrics?.test_accuracy;
                            return (
                                <Box key={m.key} sx={{
                                    borderRadius: 2, overflow: 'hidden',
                                    border: `1px solid ${accent}44`,
                                    background: 'rgba(0,0,0,0.25)',
                                }}>
                                    <Box sx={{
                                        px: 1.5, py: 0.8, display: 'flex',
                                        justifyContent: 'space-between', alignItems: 'center',
                                        borderBottom: `1px solid ${accent}33`,
                                        background: `linear-gradient(90deg, ${accent}22, transparent)`,
                                    }}>
                                        <Typography sx={{ fontSize: 12, fontWeight: 700, color: accent, letterSpacing: 0.5 }}>
                                            {m.name}
                                        </Typography>
                                        <Typography sx={{ fontSize: 11, opacity: 0.85, fontWeight: 700 }}>
                                            {acc != null ? `${(acc * 100).toFixed(2)}%` : '—'}
                                        </Typography>
                                    </Box>
                                    <Box component="img"
                                         src={`${API_URL}/api/reports/${fn}`}
                                         alt={`${m.name} confusion matrix`}
                                         sx={{ width: '100%', display: 'block' }} />
                                </Box>
                            );
                        })}
                        {/* Ensemble soft-vote panel — same grid cell so it sits next to the 3 model matrices */}
                        {data.ensemble?.confusion_png && (() => {
                            const accent = '#67e8f9';
                            const ens = data.ensemble;
                            const acc = ens.soft_vote_test_acc;
                            return (
                                <Box sx={{
                                    borderRadius: 2, overflow: 'hidden',
                                    border: `2px solid ${accent}88`,
                                    background: 'rgba(0,0,0,0.25)',
                                    boxShadow: `0 0 16px ${accent}22`,
                                }}>
                                    <Box sx={{
                                        px: 1.5, py: 0.8, display: 'flex',
                                        justifyContent: 'space-between', alignItems: 'center',
                                        borderBottom: `1px solid ${accent}55`,
                                        background: `linear-gradient(90deg, ${accent}33, transparent)`,
                                    }}>
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.8 }}>
                                            <Typography sx={{ fontSize: 12, fontWeight: 800, color: accent, letterSpacing: 0.5 }}>
                                                Ensemble · soft-vote
                                            </Typography>
                                            <Box sx={{ fontSize: 9.5, fontWeight: 700, letterSpacing: 0.3,
                                                       px: 0.7, py: 0.1, borderRadius: 0.6,
                                                       background: `${accent}33`, color: '#fff' }}>
                                                3-model
                                            </Box>
                                        </Box>
                                        <Typography sx={{ fontSize: 11, opacity: 0.9, fontWeight: 700 }}>
                                            {acc != null ? `${(acc * 100).toFixed(2)}%` : '—'}
                                        </Typography>
                                    </Box>
                                    <Box component="img"
                                         src={`${API_URL}/api/reports/${ens.confusion_png}`}
                                         alt="Ensemble soft-vote confusion matrix"
                                         sx={{ width: '100%', display: 'block' }} />
                                    {(typeof ens.ensemble_corrected_cases === 'number'
                                      || typeof ens.ensemble_outvoted_cases === 'number') && (
                                        <Box sx={{
                                            px: 1.5, py: 0.8,
                                            borderTop: `1px solid ${accent}33`,
                                            display: 'flex', justifyContent: 'space-around', gap: 1,
                                            background: 'rgba(0,0,0,0.25)',
                                        }}>
                                            <Box sx={{ textAlign: 'center' }}>
                                                <Typography sx={{ fontSize: 9.5, opacity: 0.65,
                                                                  letterSpacing: 0.5, textTransform: 'uppercase' }}>
                                                    voting corrected
                                                </Typography>
                                                <Typography sx={{ fontSize: 16, fontWeight: 800, color: '#38ef7d' }}>
                                                    +{ens.ensemble_corrected_cases ?? 0}
                                                </Typography>
                                            </Box>
                                            <Box sx={{ textAlign: 'center' }}>
                                                <Typography sx={{ fontSize: 9.5, opacity: 0.65,
                                                                  letterSpacing: 0.5, textTransform: 'uppercase' }}>
                                                    voting outvoted
                                                </Typography>
                                                <Typography sx={{ fontSize: 16, fontWeight: 800, color: '#f5a623' }}>
                                                    −{ens.ensemble_outvoted_cases ?? 0}
                                                </Typography>
                                            </Box>
                                            <Box sx={{ textAlign: 'center' }}>
                                                <Typography sx={{ fontSize: 9.5, opacity: 0.65,
                                                                  letterSpacing: 0.5, textTransform: 'uppercase' }}>
                                                    net gain
                                                </Typography>
                                                <Typography sx={{ fontSize: 16, fontWeight: 800, color: '#fff' }}>
                                                    {(ens.ensemble_corrected_cases ?? 0)
                                                     - (ens.ensemble_outvoted_cases ?? 0) >= 0 ? '+' : ''}
                                                    {(ens.ensemble_corrected_cases ?? 0)
                                                     - (ens.ensemble_outvoted_cases ?? 0)}
                                                </Typography>
                                            </Box>
                                        </Box>
                                    )}
                                </Box>
                            );
                        })()}
                    </Box>
                </Paper>
            )}

            {/* Efficiency vs Accuracy — SVG scatter plot */}
            {hasMetrics && (() => {
                const points = data.models
                    .map(m => ({
                        key: m.key,
                        name: m.name,
                        accent: ACCENT[m.key] || '#94a3b8',
                        latency: m.latency?.tta5_ms_mean,
                        accuracy: m.test_metrics?.test_accuracy,
                    }))
                    .filter(p => typeof p.latency === 'number' && typeof p.accuracy === 'number');
                if (points.length === 0) return null;

                const W = 680, H = 380;
                const M = { top: 24, right: 24, bottom: 56, left: 64 };
                const innerW = W - M.left - M.right;
                const innerH = H - M.top - M.bottom;

                const lats = points.map(p => p.latency);
                const accs = points.map(p => p.accuracy * 100);
                const xMin = Math.max(0, Math.floor(Math.min(...lats) * 0.85));
                const xMax = Math.ceil(Math.max(...lats) * 1.15);
                const yMin = Math.max(0, Math.floor(Math.min(...accs) - 1));
                const yMax = Math.min(100, Math.ceil(Math.max(...accs) + 1));

                const sx = v => M.left + (v - xMin) / (xMax - xMin) * innerW;
                const sy = v => M.top + (1 - (v - yMin) / (yMax - yMin)) * innerH;

                const xTicks = Array.from({ length: 5 }, (_, i) => xMin + (xMax - xMin) * i / 4);
                const yTicks = Array.from({ length: 5 }, (_, i) => yMin + (yMax - yMin) * i / 4);

                return (
                    <Paper className="glass-card" elevation={0} sx={{ p: 3, mb: 3 }}>
                        <Typography variant="h6" sx={{ fontWeight: 700, color: '#667eea', mb: 2 }}>
                            {t('metrics.eff.title')}
                        </Typography>
                        <Box sx={{ display: 'flex', justifyContent: 'center', overflowX: 'auto' }}>
                            <svg width={W} height={H} style={{ maxWidth: '100%' }} viewBox={`0 0 ${W} ${H}`}>
                                {/* Best-tradeoff zone shading (top-left) */}
                                <rect x={M.left} y={M.top}
                                      width={innerW * 0.45} height={innerH * 0.45}
                                      fill="rgba(56,239,125,0.07)" stroke="rgba(56,239,125,0.15)" />
                                <text x={M.left + 8} y={M.top + 18}
                                      fontSize="11" fontWeight="700" fill="rgba(56,239,125,0.85)">
                                    {t('metrics.eff.best')}
                                </text>
                                {/* Gridlines + tick labels */}
                                {xTicks.map((t, i) => (
                                    <g key={`xt-${i}`}>
                                        <line x1={sx(t)} y1={M.top} x2={sx(t)} y2={M.top + innerH}
                                              stroke="rgba(255,255,255,0.07)" />
                                        <text x={sx(t)} y={M.top + innerH + 18} textAnchor="middle"
                                              fontSize="11" fill="rgba(255,255,255,0.55)">
                                            {t.toFixed(1)}
                                        </text>
                                    </g>
                                ))}
                                {yTicks.map((t, i) => (
                                    <g key={`yt-${i}`}>
                                        <line x1={M.left} y1={sy(t)} x2={M.left + innerW} y2={sy(t)}
                                              stroke="rgba(255,255,255,0.07)" />
                                        <text x={M.left - 8} y={sy(t) + 4} textAnchor="end"
                                              fontSize="11" fill="rgba(255,255,255,0.55)">
                                            {t.toFixed(1)}%
                                        </text>
                                    </g>
                                ))}
                                {/* Axes */}
                                <line x1={M.left} y1={M.top + innerH}
                                      x2={M.left + innerW} y2={M.top + innerH}
                                      stroke="rgba(255,255,255,0.25)" />
                                <line x1={M.left} y1={M.top}
                                      x2={M.left} y2={M.top + innerH}
                                      stroke="rgba(255,255,255,0.25)" />
                                {/* Axis labels */}
                                <text x={M.left + innerW / 2} y={H - 12} textAnchor="middle"
                                      fontSize="12" fill="rgba(255,255,255,0.75)" fontWeight="600">
                                    {t('metrics.eff.xaxis')}
                                </text>
                                <text x={-(M.top + innerH / 2)} y={18} textAnchor="middle"
                                      transform="rotate(-90)"
                                      fontSize="12" fill="rgba(255,255,255,0.75)" fontWeight="600">
                                    {t('metrics.eff.yaxis')}
                                </text>
                                {/* Data points */}
                                {points.map(p => {
                                    const cx = sx(p.latency);
                                    const cy = sy(p.accuracy * 100);
                                    return (
                                        <g key={p.key}>
                                            <circle cx={cx} cy={cy} r={10} fill={p.accent}
                                                    stroke="#fff" strokeWidth={2} opacity={0.95} />
                                            <text x={cx + 16} y={cy - 4} fontSize="13"
                                                  fontWeight="700" fill={p.accent}>
                                                {p.name}
                                            </text>
                                            <text x={cx + 16} y={cy + 12} fontSize="10.5"
                                                  fill="rgba(255,255,255,0.6)">
                                                {(p.accuracy * 100).toFixed(2)}% · {p.latency.toFixed(1)}ms
                                            </text>
                                        </g>
                                    );
                                })}
                            </svg>
                        </Box>
                        <Typography variant="caption" sx={{ display: 'block', mt: 1, opacity: 0.65 }}>
                            {t('metrics.eff.subtitle')}
                        </Typography>
                    </Paper>
                );
            })()}

            {/* Per-Class Precision / Recall / F1 — only if full metrics are available */}
            {hasMetrics && data.models.some(m => m.test_metrics?.macro_f1 != null) && (
                <Paper className="glass-card" elevation={0} sx={{ p: 3, mb: 3 }}>
                    <Typography variant="h6" sx={{ fontWeight: 700, color: '#667eea', mb: 0.5 }}>
                        {t('metrics.prf.title')}
                    </Typography>
                    <Typography variant="caption" sx={{ display: 'block', opacity: 0.7, mb: 2 }}>
                        {t('metrics.prf.subtitle')}
                    </Typography>
                    <Box sx={{ overflowX: 'auto' }}>
                        <Box component="table" sx={{
                            width: '100%', borderCollapse: 'collapse',
                            '& th, & td': { textAlign: 'center', py: 0.9, px: 1.2, fontSize: 12,
                                borderBottom: '1px solid rgba(255,255,255,0.06)' },
                            '& th': { fontWeight: 700, color: '#a3b3ff', fontSize: 10.5,
                                letterSpacing: 0.5, textTransform: 'uppercase' },
                            '& td:first-of-type, & th:first-of-type': { textAlign: 'left' },
                        }}>
                            <thead><tr>
                                <th>{t('metrics.prf.col.model')}</th>
                                <th>{t('metrics.prf.col.precision')}</th><th>{t('metrics.prf.col.recall')}</th><th>{t('metrics.prf.col.f1')}</th><th>{t('metrics.prf.col.support')}</th>
                            </tr></thead>
                            <tbody>
                            {data.models.map(m => {
                                const tm = m.test_metrics;
                                if (!tm?.per_class) return null;
                                const accent = ACCENT[m.key] || '#94a3b8';
                                return (
                                    <React.Fragment key={m.key}>
                                        {CLASS_KEYS.map((cls, i) => {
                                            const pc = tm.per_class?.[cls] || {};
                                            return (
                                                <tr key={`${m.key}-${cls}`}>
                                                    <td>
                                                        {i === 0 ? (
                                                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                                <Box sx={{ width: 8, height: 8, borderRadius: '50%', background: accent }} />
                                                                <strong style={{ color: accent }}>{m.name}</strong>
                                                                <span style={{ opacity: 0.7 }}>· {cls}</span>
                                                            </Box>
                                                        ) : (
                                                            <span style={{ opacity: 0.7, paddingLeft: 18 }}>· {cls}</span>
                                                        )}
                                                    </td>
                                                    <td>{typeof pc.precision === 'number' ? `${(pc.precision * 100).toFixed(2)}%` : '—'}</td>
                                                    <td>{typeof pc.recall === 'number' ? `${(pc.recall * 100).toFixed(2)}%` : '—'}</td>
                                                    <td><strong>{typeof pc.f1 === 'number' ? `${(pc.f1 * 100).toFixed(2)}%` : '—'}</strong></td>
                                                    <td style={{ opacity: 0.7 }}>{pc.support ?? '—'}</td>
                                                </tr>
                                            );
                                        })}
                                        {tm.macro_f1 != null && (
                                            <tr style={{ background: `${accent}10` }}>
                                                <td><em style={{ opacity: 0.85, paddingLeft: 18 }}>{t('metrics.prf.macro')}</em></td>
                                                <td>{(tm.macro_precision * 100).toFixed(2)}%</td>
                                                <td>{(tm.macro_recall * 100).toFixed(2)}%</td>
                                                <td><strong>{(tm.macro_f1 * 100).toFixed(2)}%</strong></td>
                                                <td style={{ opacity: 0.7 }}>—</td>
                                            </tr>
                                        )}
                                    </React.Fragment>
                                );
                            })}
                            </tbody>
                        </Box>
                    </Box>
                </Paper>
            )}

            {/* Confusion Matrices — one heat-mapped 4x4 grid per model */}
            {hasMetrics && data.models.some(m => Array.isArray(m.test_metrics?.confusion_matrix)) && (
                <Paper className="glass-card" elevation={0} sx={{ p: 3, mb: 3 }}>
                    <Typography variant="h6" sx={{ fontWeight: 700, color: '#667eea', mb: 0.5 }}>
                        {t('metrics.cm.title')}
                    </Typography>
                    <Typography variant="caption" sx={{ display: 'block', opacity: 0.7, mb: 2.5 }}>
                        {t('metrics.cm.subtitle')}
                    </Typography>
                    <Box sx={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
                        gap: 3,
                    }}>
                        {data.models.map(m => {
                            const cm = m.test_metrics?.confusion_matrix;
                            const order = m.test_metrics?.class_order || CLASS_KEYS;
                            if (!Array.isArray(cm)) return null;
                            const accent = ACCENT[m.key] || '#94a3b8';
                            // For color scaling on mistakes: find the max off-diagonal value
                            let maxErr = 1;
                            for (let i = 0; i < cm.length; i++) {
                                for (let j = 0; j < cm[i].length; j++) {
                                    if (i !== j && cm[i][j] > maxErr) maxErr = cm[i][j];
                                }
                            }
                            return (
                                <Box key={m.key} sx={{
                                    p: 2, borderRadius: 2,
                                    background: 'rgba(255,255,255,0.03)',
                                    border: `1px solid ${accent}55`,
                                }}>
                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
                                        <Box sx={{ width: 10, height: 10, borderRadius: '50%', background: accent }} />
                                        <Typography variant="subtitle2" sx={{ fontWeight: 700, color: accent }}>
                                            {m.name}
                                        </Typography>
                                    </Box>
                                    <Box sx={{ overflowX: 'auto' }}>
                                        <Box sx={{
                                            display: 'grid',
                                            gridTemplateColumns: `auto repeat(${order.length}, 1fr)`,
                                            gap: 0.4,
                                            minWidth: 280,
                                        }}>
                                            {/* Top-left corner (empty / label) */}
                                            <Box sx={{ fontSize: 9, color: 'rgba(255,255,255,0.55)',
                                                       display: 'flex', alignItems: 'flex-end', justifyContent: 'center',
                                                       writingMode: 'horizontal-tb', pb: 0.5, pr: 0.5 }}>
                                                actual ↓<br/>pred →
                                            </Box>
                                            {/* Column headers (predicted) */}
                                            {order.map(c => (
                                                <Box key={`col-${c}`} sx={{
                                                    textAlign: 'center', fontSize: 10, fontWeight: 700,
                                                    color: 'rgba(255,255,255,0.7)', pb: 0.5,
                                                    letterSpacing: 0.3,
                                                }}>
                                                    {c.replace('Meningioma', 'Mening.').replace('No Tumor', 'NoTumor').replace('Pituitary', 'Pituit.').replace('Glioma', 'Glioma')}
                                                </Box>
                                            ))}
                                            {/* Body rows */}
                                            {order.map((rowName, i) => (
                                                <React.Fragment key={`row-${rowName}`}>
                                                    <Box sx={{
                                                        textAlign: 'right', fontSize: 10, fontWeight: 700,
                                                        color: 'rgba(255,255,255,0.7)', pr: 0.5,
                                                        display: 'flex', alignItems: 'center', justifyContent: 'flex-end',
                                                    }}>
                                                        {rowName}
                                                    </Box>
                                                    {order.map((colName, j) => {
                                                        const v = (cm[i] && cm[i][j]) || 0;
                                                        const isDiag = i === j;
                                                        // Color: diagonal = green intensity by row support; off-diag = red intensity by maxErr
                                                        const rowTotal = cm[i].reduce((a, b) => a + b, 0) || 1;
                                                        let bg, color;
                                                        if (isDiag) {
                                                            const intensity = Math.min(1, v / rowTotal);
                                                            bg = `rgba(56, 239, 125, ${0.15 + intensity * 0.55})`;
                                                            color = intensity > 0.5 ? '#fff' : '#caffc4';
                                                        } else {
                                                            const intensity = v === 0 ? 0 : Math.min(1, v / maxErr);
                                                            bg = v === 0
                                                                ? 'rgba(255,255,255,0.04)'
                                                                : `rgba(245, 87, 108, ${0.15 + intensity * 0.55})`;
                                                            color = intensity > 0.5 ? '#fff' : 'rgba(255,255,255,0.85)';
                                                        }
                                                        return (
                                                            <Box key={`${i}-${j}`} sx={{
                                                                aspectRatio: '1', minHeight: 42, minWidth: 42,
                                                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                                                fontWeight: isDiag ? 700 : 500,
                                                                fontSize: 13, color, background: bg,
                                                                borderRadius: 1,
                                                                border: isDiag ? '1px solid rgba(56,239,125,0.4)' : '1px solid rgba(255,255,255,0.05)',
                                                            }}
                                                            title={`actual=${rowName}, predicted=${colName}: ${v}`}>
                                                                {v}
                                                            </Box>
                                                        );
                                                    })}
                                                </React.Fragment>
                                            ))}
                                        </Box>
                                    </Box>
                                    {/* Per-model accuracy summary */}
                                    {(() => {
                                        let diag = 0, total = 0;
                                        for (let i = 0; i < cm.length; i++) {
                                            for (let j = 0; j < cm[i].length; j++) {
                                                total += cm[i][j];
                                                if (i === j) diag += cm[i][j];
                                            }
                                        }
                                        const errs = total - diag;
                                        return (
                                            <Typography variant="caption" sx={{ display: 'block', mt: 1.5, opacity: 0.75 }}>
                                                <span style={{ color: '#38ef7d', fontWeight: 700 }}>{diag}</span> {t('metrics.cm.correct')}
                                                · <span style={{ color: '#f5576c', fontWeight: 700 }}>{errs}</span> {t('metrics.cm.wrong')}
                                                · {total} {t('metrics.cm.total')} <strong>{((diag / total) * 100).toFixed(2)}%</strong>
                                            </Typography>
                                        );
                                    })()}
                                </Box>
                            );
                        })}
                    </Box>
                </Paper>
            )}

            {!hasMetrics && (
                <Alert severity="info" sx={{ mt: 2 }}>
                    Test metrics not available yet. Train models with <code>src/train_combined.py</code> and reports will appear in <code>reports/combined/</code>.
                </Alert>
            )}
        </Box>
    );
}


/* ── Pipeline Page: visual explanation of the analysis flow ─── */
function PipelineStage({ num, color, title, subtitle, children, isLast }) {
    return (
        <Box sx={{ display: 'flex', gap: 2.5, position: 'relative' }}>
            {/* Number bubble + connector line */}
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flexShrink: 0 }}>
                <Box sx={{
                    width: 38, height: 38, borderRadius: '50%',
                    background: `linear-gradient(135deg, ${color}, ${color}cc)`,
                    color: '#fff', fontWeight: 800, fontSize: 16,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    boxShadow: `0 4px 12px ${color}55`,
                    border: '2px solid rgba(255,255,255,0.15)',
                    zIndex: 2,
                }}>{num}</Box>
                {!isLast && (
                    <Box sx={{
                        width: 2, flex: 1, minHeight: 24, mt: 0.5, mb: 0.5,
                        background: `linear-gradient(180deg, ${color}88, rgba(255,255,255,0.08))`,
                    }} />
                )}
            </Box>
            {/* Stage card */}
            <Box sx={{
                flex: 1, mb: isLast ? 0 : 2.5,
                p: 2.5, borderRadius: 2,
                background: 'rgba(255,255,255,0.04)',
                border: `1px solid ${color}33`,
            }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 700, color, mb: 0.4 }}>
                    {title}
                </Typography>
                <Typography variant="caption" sx={{ display: 'block', opacity: 0.8, mb: 1.5, fontSize: 12 }}>
                    {subtitle}
                </Typography>
                {children}
            </Box>
        </Box>
    );
}


/* Small visual building blocks used by stages */
function MiniMRI({ size = 70, withBox = false, boxColor = '#38ef7d', boxAt = [0.3, 0.3, 0.35, 0.35] }) {
    // SVG sketch of a brain with optional bbox
    return (
        <svg width={size} height={size} viewBox="0 0 100 100" style={{ display: 'block' }}>
            <defs>
                <radialGradient id="brainGrad" cx="50%" cy="50%">
                    <stop offset="0%" stopColor="#3a3a4a" />
                    <stop offset="100%" stopColor="#1a1a24" />
                </radialGradient>
            </defs>
            <rect x="0" y="0" width="100" height="100" fill="#0d0d14" rx="6" />
            <ellipse cx="50" cy="52" rx="32" ry="36" fill="url(#brainGrad)" stroke="#5b5b75" strokeWidth="1.5" />
            <path d="M 50 18 Q 30 30 28 52 Q 30 74 50 86" fill="none" stroke="#7a7a99" strokeWidth="0.8" opacity="0.6" />
            <path d="M 50 18 Q 70 30 72 52 Q 70 74 50 86" fill="none" stroke="#7a7a99" strokeWidth="0.8" opacity="0.6" />
            <ellipse cx="48" cy="60" rx="3" ry="4" fill="#e0e0f0" opacity="0.85" />
            {withBox && (
                <rect
                    x={boxAt[0] * 100} y={boxAt[1] * 100}
                    width={boxAt[2] * 100} height={boxAt[3] * 100}
                    fill="none" stroke={boxColor} strokeWidth="2"
                    rx="2"
                />
            )}
        </svg>
    );
}


function ModelChip({ name, color }) {
    return (
        <Box sx={{
            px: 1.2, py: 0.4, borderRadius: 1.2, fontSize: 11, fontWeight: 700,
            background: `${color}22`, color, border: `1px solid ${color}55`,
            display: 'inline-flex', alignItems: 'center', gap: 0.6,
        }}>
            <Box sx={{ width: 6, height: 6, borderRadius: '50%', background: color }} />
            {name}
        </Box>
    );
}


function Arrow({ vertical = false }) {
    return vertical ? (
        <Box sx={{ color: 'rgba(255,255,255,0.4)', fontSize: 18, textAlign: 'center', my: 0.5 }}>↓</Box>
    ) : (
        <Box sx={{ color: 'rgba(255,255,255,0.4)', fontSize: 18, mx: 1 }}>→</Box>
    );
}


function PipelinePage() {
    const { t } = useT();
    const PURPLE  = '#667eea';
    const ORANGE  = '#f5a623';
    const GREEN   = '#38ef7d';
    const PINK    = '#f5576c';
    const TEAL    = '#22d3ee';

    return (
        <Box>
            {/* Header */}
            <Paper className="glass-card" elevation={0} sx={{ p: 3, mb: 3 }}>
                <Typography variant="h5" sx={{ fontWeight: 700, color: PURPLE, mb: 0.5 }}>
                    {t('pipe.title')}
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.8 }}>
                    {t('pipe.subtitle')}
                </Typography>
            </Paper>

            {/* Pipeline stages */}
            <Paper className="glass-card" elevation={0} sx={{ p: 3, mb: 3 }}>

                <PipelineStage num="1" color={PURPLE} title={t('pipe.1.title')} subtitle={t('pipe.1.sub')}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                        <Box sx={{
                            border: '2px dashed rgba(102,126,234,0.5)', borderRadius: 1.5,
                            p: 1.5, display: 'flex', flexDirection: 'column', alignItems: 'center',
                            background: 'rgba(102,126,234,0.06)', minWidth: 120,
                        }}>
                            <Typography sx={{ fontSize: 11, opacity: 0.7 }}>{t('pipe.1.dropimage')}</Typography>
                            <MiniMRI size={56} />
                        </Box>
                        <Typography variant="caption" sx={{ opacity: 0.65, maxWidth: 280 }}>
                            {t('pipe.1.note')}
                        </Typography>
                    </Box>
                </PipelineStage>

                <PipelineStage num="2" color={PURPLE} title={t('pipe.2.title')} subtitle={t('pipe.2.sub')}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                        <MiniMRI size={64} />
                        <Arrow />
                        <Box sx={{ filter: 'contrast(1.4) brightness(1.05)' }}>
                            <MiniMRI size={64} />
                        </Box>
                        <Box sx={{ ml: 1, fontSize: 11, opacity: 0.7, lineHeight: 1.4 }}>
                            <div>{t('pipe.2.raw')}</div>
                            <div>↓ CLAHE</div>
                            <div>↓ 224×224</div>
                            <div>↓ normalize</div>
                        </Box>
                    </Box>
                </PipelineStage>

                <PipelineStage num="3" color={ORANGE} title={t('pipe.3.title')} subtitle={t('pipe.3.sub')}>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap', alignItems: 'center' }}>
                            <ModelChip name="ConvNeXt-Tiny" color="#667eea" />
                            <ModelChip name="EfficientNet-B3" color="#f5a623" />
                            <ModelChip name="ResNet-50" color="#38ef7d" />
                            <Arrow />
                            <ModelChip name={t('pipe.3.vote')} color="#94a3b8" />
                        </Box>
                        <Typography variant="caption" sx={{ opacity: 0.7, fontSize: 11 }}>
                            {t('pipe.3.note')}
                        </Typography>
                    </Box>
                </PipelineStage>

                <PipelineStage num="4" color={TEAL} title={t('pipe.4.title')} subtitle={t('pipe.4.sub')}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                        <svg width="120" height="46" viewBox="0 0 120 46">
                            <path d="M 5 40 Q 30 38 50 25 Q 70 4 90 25 Q 110 38 115 40"
                                  fill="none" stroke={TEAL} strokeWidth="2" />
                            <line x1="60" y1="6" x2="60" y2="40" stroke={TEAL} strokeWidth="1" strokeDasharray="2 2" opacity="0.5" />
                            <text x="60" y="46" fontSize="9" fill="rgba(255,255,255,0.6)" textAnchor="middle">σ</text>
                        </svg>
                        <Box sx={{ fontSize: 11, opacity: 0.75, lineHeight: 1.5 }}>
                            <div><strong style={{ color: GREEN }}>σ &lt; 0.005</strong> · {t('pipe.4.verylow')}</div>
                            <div><strong style={{ color: ORANGE }}>σ &lt; 0.10</strong> · {t('pipe.4.moderate')}</div>
                            <div><strong style={{ color: PINK }}>σ ≥ 0.10</strong> · {t('pipe.4.high')}</div>
                        </Box>
                    </Box>
                </PipelineStage>

                <PipelineStage num="5" color={TEAL} title={t('pipe.5.title')} subtitle={t('pipe.5.sub')}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, flexWrap: 'wrap' }}>
                        <Box sx={{
                            px: 1.4, py: 0.8, borderRadius: 1.5,
                            background: `${GREEN}22`, color: GREEN, border: `1px solid ${GREEN}55`,
                            fontSize: 12, fontWeight: 700,
                        }}>
                            {t('pipe.5.indist')}
                        </Box>
                        <Arrow />
                        <Box sx={{
                            px: 1.4, py: 0.8, borderRadius: 1.5,
                            background: `${PINK}22`, color: PINK, border: `1px solid ${PINK}55`,
                            fontSize: 12, fontWeight: 700,
                        }}>
                            {t('pipe.5.outdist')}
                        </Box>
                        <Box sx={{ ml: 1, fontSize: 11, opacity: 0.7, lineHeight: 1.5, maxWidth: 260 }}>
                            {t('pipe.5.note')}
                        </Box>
                    </Box>
                </PipelineStage>

                <PipelineStage num="6" color={TEAL} title={t('pipe.6.title')} subtitle={t('pipe.6.sub')}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                        <MiniMRI size={64} withBox boxColor={TEAL} />
                        <Arrow />
                        <Box sx={{
                            width: 64, height: 64, borderRadius: 1.5, position: 'relative',
                            border: `2px solid ${TEAL}`, overflow: 'hidden',
                        }}>
                            <MiniMRI size={64} />
                        </Box>
                        <Box sx={{ ml: 1, fontSize: 11, opacity: 0.7, lineHeight: 1.5, maxWidth: 240 }}>
                            {t('pipe.6.note')}
                        </Box>
                    </Box>
                </PipelineStage>

                <PipelineStage num="7" color={GREEN} title={t('pipe.7.title')} subtitle={t('pipe.7.sub')}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                        <MiniMRI size={72} />
                        <Arrow />
                        <MiniMRI size={72} withBox boxColor={GREEN} />
                        <Box sx={{ ml: 1, fontSize: 11, opacity: 0.7, lineHeight: 1.5 }}>
                            {t('pipe.7.note')}
                        </Box>
                    </Box>
                </PipelineStage>

                <PipelineStage num="8" color={GREEN} title={t('pipe.8.title')} subtitle={t('pipe.8.sub')}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                        <MiniMRI size={72} withBox boxColor={GREEN} />
                        <Arrow />
                        <svg width={72} height={72} viewBox="0 0 100 100">
                            <rect x="0" y="0" width="100" height="100" fill="#0d0d14" rx="6" />
                            <ellipse cx="50" cy="52" rx="32" ry="36" fill="#2a2a38" stroke="#5b5b75" strokeWidth="1" />
                            <path d="M 38 50 Q 42 42 50 44 Q 58 46 60 54 Q 56 62 48 62 Q 40 60 38 50 Z"
                                  fill={`${GREEN}88`} stroke={GREEN} strokeWidth="1.5" />
                        </svg>
                        <Box sx={{ ml: 1, fontSize: 11, opacity: 0.7 }}>{t('pipe.8.note')}</Box>
                    </Box>
                </PipelineStage>

                <PipelineStage num="9" color={PURPLE} title={t('pipe.9.title')} subtitle={t('pipe.9.sub')}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                        {['L1','L2','L3','L4'].map((lvl, i) => (
                            <React.Fragment key={lvl}>
                                <Box sx={{
                                    width: 46, height: 46, borderRadius: 1.5,
                                    border: `1.5px solid ${PURPLE}66`, background: `${PURPLE}${11 + i * 5}`,
                                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                                    fontSize: 11, fontWeight: 800, color: PURPLE,
                                }}>{lvl}</Box>
                                {i < 3 && <Arrow />}
                            </React.Fragment>
                        ))}
                        <Box sx={{ ml: 1, fontSize: 11, opacity: 0.7, lineHeight: 1.5, maxWidth: 260 }}>
                            {t('pipe.9.note')}
                        </Box>
                    </Box>
                </PipelineStage>

                <PipelineStage num="10" color={PURPLE} title={t('pipe.10.title')} subtitle={t('pipe.10.sub')}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                        <Box sx={{
                            display: 'grid', gridTemplateColumns: 'repeat(3, 28px)', gap: 0.4,
                        }}>
                            {[
                                'R fr','A mid','L fr',
                                'R t/p','sellar','L t/p',
                                'R occ','P mid','L occ',
                            ].map((label, i) => (
                                <Box key={i} sx={{
                                    width: 28, height: 28, borderRadius: 0.6,
                                    border: '1px solid rgba(255,255,255,0.18)',
                                    background: i === 4 ? `${PURPLE}33` : 'rgba(255,255,255,0.04)',
                                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                                    fontSize: 8, color: i === 4 ? PURPLE : 'rgba(255,255,255,0.6)',
                                    fontWeight: i === 4 ? 700 : 500,
                                }}>{label}</Box>
                            ))}
                        </Box>
                        <Box sx={{ fontSize: 11, opacity: 0.75, lineHeight: 1.5, maxWidth: 240 }}>
                            {t('pipe.10.note')}
                        </Box>
                    </Box>
                </PipelineStage>

                <PipelineStage num="11" color={ORANGE} title={t('pipe.11.title')} subtitle={t('pipe.11.sub')}>
                    <Box sx={{
                        p: 1.5, borderRadius: 1.5, fontSize: 13,
                        background: 'rgba(0,0,0,0.25)', fontFamily: 'monospace',
                        color: 'rgba(255,255,255,0.85)',
                    }}>
                        score = (base + size_bonus) × (0.6 + 0.4 × confidence)
                    </Box>
                </PipelineStage>

                <PipelineStage num="12" color={PINK} title={t('pipe.12.title')} subtitle={t('pipe.12.sub')}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                        <ModelChip name="YOLO + SAM" color={GREEN} />
                        <Typography sx={{ fontSize: 14, opacity: 0.5 }}>vs</Typography>
                        <ModelChip name="MedGemma vision" color={PINK} />
                        <Box sx={{ ml: 1, fontSize: 11, opacity: 0.7, lineHeight: 1.5, maxWidth: 260 }}>
                            {t('pipe.12.note')}
                        </Box>
                    </Box>
                </PipelineStage>

                <PipelineStage num="13" color={PINK} title={t('pipe.13.title')} subtitle={t('pipe.13.sub')}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                        {[t('pipe.13.blur'), t('pipe.13.brightness'), t('pipe.13.noise'), t('pipe.13.scanner')].map((lbl, i) => (
                            <Box key={i} sx={{
                                px: 1.1, py: 0.6, borderRadius: 1.5, fontSize: 10.5, fontWeight: 700,
                                background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.12)',
                                display: 'flex', alignItems: 'center', gap: 0.5,
                            }}>
                                <Box component="span" sx={{ color: GREEN }}>✓</Box>{lbl}
                            </Box>
                        ))}
                        <Box sx={{ ml: 1, fontSize: 11, opacity: 0.7, lineHeight: 1.5, maxWidth: 260 }}>
                            {t('pipe.13.note')}
                        </Box>
                    </Box>
                </PipelineStage>

                <PipelineStage num="14" color={PINK} title={t('pipe.14.title')} subtitle={t('pipe.14.sub')}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, flexWrap: 'wrap' }}>
                        <Box sx={{
                            p: 1.3, borderRadius: 1.5, background: 'rgba(0,0,0,0.25)',
                            fontFamily: 'monospace', fontSize: 11, color: 'rgba(255,255,255,0.7)',
                            maxWidth: 200,
                        }}>
                            {'{ class, conf,'}<br/>{'  bbox, size,'}<br/>{'  region, σ }'}
                        </Box>
                        <Arrow />
                        <Box sx={{
                            px: 1.2, py: 0.8, borderRadius: 1.5,
                            background: `${PINK}22`, color: PINK,
                            border: `1px solid ${PINK}55`, fontSize: 12, fontWeight: 700,
                        }}>
                            MedGemma 1.5 4B
                        </Box>
                        <Arrow />
                        <Box sx={{
                            p: 1.2, borderRadius: 1.5, background: 'rgba(255,255,255,0.05)',
                            fontSize: 11, lineHeight: 1.5, maxWidth: 220,
                            border: '1px solid rgba(255,255,255,0.1)',
                        }}>
                            <div style={{ color: PINK, fontWeight: 700 }}>{t('pipe.14.sample.h1')}</div>
                            <div style={{ opacity: 0.7 }}>{t('pipe.14.sample.found')}</div>
                            <div style={{ color: PINK, fontWeight: 700, marginTop: 4 }}>{t('pipe.14.sample.h2')}</div>
                            <div style={{ opacity: 0.7 }}>{t('pipe.14.sample.mean')}</div>
                        </Box>
                    </Box>
                </PipelineStage>

                <PipelineStage num="15" color={TEAL} title={t('pipe.15.title')} subtitle={t('pipe.15.sub')}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                        <Box sx={{
                            width: 46, height: 46, borderRadius: '50%',
                            background: `${TEAL}22`, border: `1px solid ${TEAL}55`,
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                        }}>
                            <ChatIcon sx={{ color: TEAL, fontSize: 22 }} />
                        </Box>
                        <Box sx={{ fontSize: 11, opacity: 0.75, lineHeight: 1.5, maxWidth: 300 }}>
                            {t('pipe.15.note')}
                        </Box>
                    </Box>
                </PipelineStage>

                <PipelineStage num="16" color={GREEN} title={t('pipe.16.title')} subtitle={t('pipe.16.sub')} isLast>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, flexWrap: 'wrap' }}>
                        <Box sx={{
                            position: 'relative', display: 'inline-block',
                        }}>
                            <MiniMRI size={72} withBox boxColor={PINK} boxAt={[0.55, 0.55, 0.3, 0.3]} />
                            <Box sx={{
                                position: 'absolute', top: -8, right: -8,
                                fontSize: 9, fontWeight: 700, color: PINK,
                                background: 'rgba(0,0,0,0.7)', borderRadius: 0.6,
                                px: 0.5, py: 0.2, border: `1px solid ${PINK}66`,
                            }}>{t('pipe.16.auto')}</Box>
                        </Box>
                        <Arrow />
                        <Box sx={{
                            position: 'relative', display: 'inline-block',
                        }}>
                            <MiniMRI size={72} withBox boxColor={GREEN} boxAt={[0.2, 0.25, 0.35, 0.35]} />
                            <Box sx={{
                                position: 'absolute', top: -8, right: -8,
                                fontSize: 9, fontWeight: 700, color: GREEN,
                                background: 'rgba(0,0,0,0.7)', borderRadius: 0.6,
                                px: 0.5, py: 0.2, border: `1px solid ${GREEN}66`,
                            }}>{t('pipe.16.manual')}</Box>
                        </Box>
                        <Arrow />
                        <Box sx={{ fontSize: 11, opacity: 0.75, lineHeight: 1.5, maxWidth: 220 }}>
                            {t('pipe.16.note')}
                        </Box>
                    </Box>
                </PipelineStage>

            </Paper>

            {/* Cross-dataset validation note — a pre-computed benchmark, not a per-request step */}
            <Paper className="glass-card" elevation={0} sx={{ p: 3, mb: 3, borderLeft: `3px solid ${GREEN}` }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 700, color: GREEN, mb: 0.5 }}>
                    {t('pipe.crossds.title')}
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.8, fontSize: 12.5, lineHeight: 1.6 }}>
                    {t('pipe.crossds.note')}
                </Typography>
            </Paper>

            {/* Bottom: tech stack summary */}
            <Paper className="glass-card" elevation={0} sx={{ p: 3, mb: 3 }}>
                <Typography variant="subtitle1" sx={{ fontWeight: 700, color: PURPLE, mb: 1.5 }}>
                    {t('pipe.stack.title')}
                </Typography>
                <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 1.5 }}>
                    {[
                        { label: t('pipe.stack.classifiers'),  value: 'ConvNeXt-Tiny + EfficientNet-B3 + ResNet-50' },
                        { label: t('pipe.stack.uncertainty'),  value: 'MC Dropout (T=20) + Energy-based OOD' },
                        { label: t('pipe.stack.consistency'),  value: 'Focus-crop re-classification' },
                        { label: t('pipe.stack.detector'),     value: 'YOLO11n / Cheng dataset' },
                        { label: t('pipe.stack.segmenter'),    value: 'MobileSAM (bbox-prompted)' },
                        { label: t('pipe.stack.explainability'), value: 'Grad-CAM + Grad-CAM++ + LayerCAM (4 levels)' },
                        { label: t('pipe.stack.robustness'),   value: 'Blur / brightness / noise / scanner-artifact stress-test' },
                        { label: t('pipe.stack.vlm'),          value: 'MedGemma 1.5 4B via Ollama' },
                        { label: t('pipe.stack.serverui'),     value: 'Flask + React (MUI)' },
                    ].map(item => (
                        <Box key={item.label} sx={{
                            p: 1.4, borderRadius: 1.5,
                            background: 'rgba(255,255,255,0.04)',
                            border: '1px solid rgba(255,255,255,0.08)',
                        }}>
                            <Typography sx={{ fontSize: 10, letterSpacing: 1, textTransform: 'uppercase',
                                              opacity: 0.6, fontWeight: 700 }}>
                                {item.label}
                            </Typography>
                            <Typography sx={{ fontSize: 12.5, fontWeight: 600, mt: 0.4 }}>
                                {item.value}
                            </Typography>
                        </Box>
                    ))}
                </Box>
            </Paper>
        </Box>
    );
}


/* ── Trust Verdict strip — one-line summary at the top of Analysis Results ── */
function TrustVerdictStrip({ r }) {
    const { t } = useT();
    if (!r) return null;

    const ood          = r?.ood?.is_ood === true;
    const needsReview  = r?.uncertainty?.needs_review === true;
    const epistemic    = Number(r?.uncertainty?.epistemic) || 0;
    const unanimous    = r?.agreement?.unanimous !== false; // treat undefined as unanimous
    const focusOK      = r?.focus_crop?.agreement?.is_consistent !== false;

    const reasons = [];
    if (ood)         reasons.push(t('verdict.reason.ood'));
    if (needsReview) reasons.push(t('verdict.reason.review'));
    if (epistemic > 0.10) reasons.push(t('verdict.reason.epistemic'));
    if (!unanimous)  reasons.push(t('verdict.reason.disagree'));
    if (!focusOK)    reasons.push(t('verdict.reason.focus'));

    // Verdict: RED if OOD or 2+ flags, YELLOW if 1 flag, GREEN otherwise
    let level = 'green';
    if (ood || reasons.length >= 2) level = 'red';
    else if (reasons.length === 1)   level = 'yellow';

    const cfg = {
        green:  { color: '#38ef7d', label: t('verdict.green'),  icon: '✓' },
        yellow: { color: '#f5a623', label: t('verdict.yellow'), icon: '!' },
        red:    { color: '#f5576c', label: t('verdict.red'),    icon: '×' },
    }[level];

    return (
        <Box sx={{
            mb: 2, p: 1.5, borderRadius: 2,
            background: `linear-gradient(90deg, ${cfg.color}22 0%, ${cfg.color}08 100%)`,
            border: `1px solid ${cfg.color}55`,
            display: 'flex', alignItems: 'center', gap: 1.5, flexWrap: 'wrap',
            position: 'relative',
            '&::before': {
                content: '""', position: 'absolute', left: 0, top: 8, bottom: 8, width: 3,
                background: cfg.color, borderRadius: 4, boxShadow: `0 0 8px ${cfg.color}`,
            },
        }}>
            <Box sx={{
                width: 28, height: 28, borderRadius: '50%',
                background: `${cfg.color}33`, color: cfg.color,
                border: `2px solid ${cfg.color}`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 16, fontWeight: 800, ml: 1,
            }}>{cfg.icon}</Box>
            <Box sx={{ flex: 1, minWidth: 120 }}>
                <Typography sx={{
                    fontSize: 12, fontWeight: 800, color: cfg.color, letterSpacing: 0.5,
                    textTransform: 'uppercase',
                }}>
                    {cfg.label}
                </Typography>
                {reasons.length > 0 && (
                    <Typography sx={{ fontSize: 11, opacity: 0.75, mt: 0.2 }}>
                        {reasons.join(' · ')}
                    </Typography>
                )}
                {reasons.length === 0 && (
                    <Typography sx={{ fontSize: 11, opacity: 0.75, mt: 0.2 }}>
                        {t('verdict.all_clean')}
                    </Typography>
                )}
            </Box>
        </Box>
    );
}


/* ── "Questions to ask your doctor" — patient-facing static list per class ── */
function QuestionsForDoctorCard({ predictedClass }) {
    const { t } = useT();
    if (!predictedClass || predictedClass === 'No Tumor') return null;

    // i18n keys: 'questions.{ClassKey}.{0..N}'
    const classKey = predictedClass.replace(/\s+/g, '');  // "No Tumor" → "NoTumor"
    const items = [];
    for (let i = 0; i < 8; i++) {
        const key = `questions.${classKey}.${i}`;
        const val = t(key);
        if (val && val !== key) items.push(val); else break;
    }
    if (items.length === 0) return null;

    const accent = '#22d3ee';
    return (
        <Paper className="glass-card" elevation={0}
            sx={{
                p: 3, mt: 2.5, borderRadius: 3,
                background: 'linear-gradient(135deg, rgba(34,211,238,0.04) 0%, rgba(167,139,250,0.03) 100%)',
                border: '1px solid rgba(34,211,238,0.18)',
            }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.2, mb: 0.4 }}>
                <Box sx={{
                    width: 4, height: 18, borderRadius: 1,
                    background: `linear-gradient(180deg, ${accent}, #a78bfa)`,
                    boxShadow: `0 0 8px ${accent}66`,
                }} />
                <Typography sx={{
                    fontWeight: 800, fontSize: 14, letterSpacing: 0.8,
                    background: `linear-gradient(90deg, ${accent}, #c4b5fd)`,
                    WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
                    textTransform: 'uppercase',
                }}>
                    {t('questions.title')}
                </Typography>
            </Box>
            <Typography variant="caption" sx={{ display: 'block', opacity: 0.7, mb: 1.5, fontSize: 11 }}>
                {t('questions.subtitle')}
            </Typography>
            <Box component="ol" sx={{ pl: 2.5, m: 0 }}>
                {items.map((q, i) => (
                    <Box component="li" key={i} sx={{
                        fontSize: 12.5, lineHeight: 1.55, opacity: 0.9, mb: 0.6,
                        '&::marker': { color: accent, fontWeight: 700 },
                    }}>
                        {q}
                    </Box>
                ))}
            </Box>
        </Paper>
    );
}


/* ── Symptom Selector — patient-driven malignancy adjustment ── */
function SymptomSelector({ predictedClass, baseScore, onAdjustedScore }) {
    const { t } = useT();
    const [selected, setSelected] = useState(new Set());

    if (!predictedClass || predictedClass === 'No Tumor') return null;

    // Symptoms keyed by class. Each has a key (stable id), label (i18n), weight (0..1).
    const SYMPTOMS = {
        'Glioma': [
            { k: 'headache',  w: 0.4 },
            { k: 'seizures',  w: 0.9 },
            { k: 'cognitive', w: 0.7 },
            { k: 'motor',     w: 0.7 },
            { k: 'speech',    w: 0.6 },
            { k: 'vomiting',  w: 0.5 },
            { k: 'vision',    w: 0.5 },
        ],
        'Meningioma': [
            { k: 'headache',     w: 0.3 },
            { k: 'seizures',     w: 0.6 },
            { k: 'weakness',     w: 0.5 },
            { k: 'vision_eye',   w: 0.5 },
            { k: 'hearing',      w: 0.4 },
            { k: 'memory',       w: 0.4 },
        ],
        'Pituitary': [
            { k: 'vision_field', w: 0.8 },
            { k: 'headache',     w: 0.3 },
            { k: 'hormonal_f',   w: 0.6 },
            { k: 'acromegaly',   w: 0.6 },
            { k: 'cushing',      w: 0.6 },
            { k: 'fatigue',      w: 0.4 },
            { k: 'libido',       w: 0.5 },
        ],
    };

    const list = SYMPTOMS[predictedClass] || [];
    if (list.length === 0) return null;

    const toggle = (key) => {
        setSelected(prev => {
            const next = new Set(prev);
            if (next.has(key)) next.delete(key); else next.add(key);
            // recompute adjusted score
            const totalW = list
                .filter(s => next.has(s.k))
                .reduce((a, s) => a + s.w, 0);
            const bonus = Math.min(2.0, totalW * 0.7);
            const adjusted = Math.min(10, +(Number(baseScore) + bonus).toFixed(1));
            onAdjustedScore?.({
                adjusted, bonus: +bonus.toFixed(1), count: next.size,
            });
            return next;
        });
    };

    const ACCENT = '#f472b6';

    return (
        <Box sx={{ mt: 2.5 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.2, mb: 0.4 }}>
                <Box sx={{
                    width: 4, height: 18, borderRadius: 1,
                    background: `linear-gradient(180deg, ${ACCENT}, #a78bfa)`,
                    boxShadow: `0 0 8px ${ACCENT}66`,
                }} />
                <Typography sx={{
                    fontWeight: 800, fontSize: 14, letterSpacing: 0.8,
                    background: `linear-gradient(90deg, ${ACCENT}, #c4b5fd)`,
                    WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
                    textTransform: 'uppercase',
                }}>
                    {t('symptoms.title')}
                </Typography>
            </Box>
            <Typography variant="caption" sx={{ display: 'block', opacity: 0.7, mb: 1.5, fontSize: 11 }}>
                {t('symptoms.subtitle')}
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.8 }}>
                {list.map(s => {
                    const isSel = selected.has(s.k);
                    return (
                        <Box key={s.k}
                            onClick={() => toggle(s.k)}
                            sx={{
                                cursor: 'pointer', userSelect: 'none',
                                px: 1.3, py: 0.6, borderRadius: 1.5,
                                fontSize: 11.5, fontWeight: 600,
                                background: isSel ? `${ACCENT}22` : 'rgba(255,255,255,0.04)',
                                color: isSel ? ACCENT : 'rgba(255,255,255,0.75)',
                                border: `1px solid ${isSel ? ACCENT + '88' : 'rgba(255,255,255,0.1)'}`,
                                transition: 'all .15s ease',
                                '&:hover': { background: isSel ? `${ACCENT}33` : 'rgba(255,255,255,0.08)' },
                            }}>
                            {isSel ? '✓ ' : ''}{t(`symptoms.${predictedClass.replace(/\s+/g,'')}.${s.k}`)}
                        </Box>
                    );
                })}
            </Box>
        </Box>
    );
}


/* Localized static clinical context — keeps the panel in sync with the
   current UI language without needing to re-fetch. Backend still provides
   `data.clinical_context` (used as the fallback), but when the user toggles
   the language we re-render against this local table. */
const CLINICAL_CONTEXT_LOCAL = {
    en: {
        Glioma: {
            primer: 'Gliomas arise from glial cells (astrocytes, oligodendrocytes). Often infiltrative — borders may extend beyond what is visible. Grades range I–IV; III–IV (anaplastic / glioblastoma) are highly malignant.',
            size_interpretation: {
                small:  'Small enhancing focus — could be early-stage or low-grade. Still warrants close follow-up.',
                medium: 'Moderate volume — common in grade II–III gliomas.',
                large:  'Large mass — concerning for high-grade (glioblastoma) with mass effect; urgent neurosurgical review.',
            },
            subtypes: ['Astrocytoma', 'Oligodendroglioma', 'Glioblastoma (GBM)'],
        },
        Meningioma: {
            primer: 'Meningiomas arise from arachnoid cap cells of the meninges. Mostly benign (WHO grade I, ~90%). Slow-growing. Usually extra-axial with broad dural attachment.',
            size_interpretation: {
                small:  'Small lesion — incidental meningiomas <2 cm are often watched, not operated.',
                medium: 'Moderate size — symptomatic depending on location; surgery often considered.',
                large:  'Large mass — likely symptomatic (headache, focal deficits) but grade can still be benign.',
            },
            subtypes: ['WHO grade I (benign)', 'WHO grade II (atypical)', 'WHO grade III (anaplastic, rare)'],
        },
        Pituitary: {
            primer: 'Pituitary adenomas arise in the sella turcica. Almost always benign. Classified by size: microadenoma <10 mm, macroadenoma ≥10 mm. May be functioning (hormone-secreting) or non-functioning.',
            size_interpretation: {
                small:  'Likely microadenoma (<10 mm). Often hormone-secreting (prolactin, ACTH, GH).',
                medium: 'Macroadenoma — may cause visual field deficits if compressing the optic chiasm.',
                large:  'Large macroadenoma — risk of chiasmal compression, hypopituitarism; neurosurgery + endocrinology referral.',
            },
            subtypes: ['Non-functioning adenoma', 'Prolactinoma', 'GH-secreting (acromegaly)', 'ACTH-secreting (Cushing)'],
        },
    },
    es: {
        Glioma: {
            primer: 'Los gliomas surgen de células gliales (astrocitos, oligodendrocitos). A menudo son infiltrativos — los bordes pueden extenderse más allá de lo visible. Los grados van de I a IV; III–IV (anaplásico / glioblastoma) son altamente malignos.',
            size_interpretation: {
                small:  'Foco pequeño con realce — podría ser etapa temprana o bajo grado. Igualmente requiere seguimiento cercano.',
                medium: 'Volumen moderado — común en gliomas de grado II–III.',
                large:  'Masa grande — preocupante por alto grado (glioblastoma) con efecto masa; revisión neuroquirúrgica urgente.',
            },
            subtypes: ['Astrocitoma', 'Oligodendroglioma', 'Glioblastoma (GBM)'],
        },
        Meningioma: {
            primer: 'Los meningiomas surgen de las células aracnoideas de las meninges. Mayormente benignos (OMS grado I, ~90%). De crecimiento lento. Usualmente extra-axiales con amplia inserción dural.',
            size_interpretation: {
                small:  'Lesión pequeña — los meningiomas incidentales <2 cm a menudo se vigilan, no se operan.',
                medium: 'Tamaño moderado — sintomático según la ubicación; se considera cirugía con frecuencia.',
                large:  'Masa grande — probablemente sintomática (cefalea, déficits focales) pero el grado aún puede ser benigno.',
            },
            subtypes: ['OMS grado I (benigno)', 'OMS grado II (atípico)', 'OMS grado III (anaplásico, raro)'],
        },
        Pituitary: {
            primer: 'Los adenomas hipofisarios surgen en la silla turca. Casi siempre benignos. Clasificados por tamaño: microadenoma <10 mm, macroadenoma ≥10 mm. Pueden ser funcionantes (secretores de hormonas) o no funcionantes.',
            size_interpretation: {
                small:  'Probablemente microadenoma (<10 mm). A menudo secretor hormonal (prolactina, ACTH, GH).',
                medium: 'Macroadenoma — puede causar déficits del campo visual si comprime el quiasma óptico.',
                large:  'Macroadenoma grande — riesgo de compresión quiasmática, hipopituitarismo; derivación a neurocirugía + endocrinología.',
            },
            subtypes: ['Adenoma no funcionante', 'Prolactinoma', 'Secretor de GH (acromegalia)', 'Secretor de ACTH (Cushing)'],
        },
    },
};

/* Localized "what this location suggests" notes — mirrors backend
   _CLINICAL_PRIORS_ES_NOTES + _CLINICAL_PRIORS so it reacts to language toggle. */
const REGION_EXPLANATION_LOCAL = {
    en: {
        Glioma: {
            typical:  'Gliomas most commonly arise in the cerebral hemispheres (frontal/temporal/parietal). Attention region is consistent with this prior.',
            atypical: 'Midline gliomas (brainstem, thalamic, diffuse midline) do occur but are less common. Recommend review.',
        },
        Meningioma: {
            typical:  'Meningiomas typically arise from dural surfaces (parasagittal, convexity, falx, posterior midline). Consistent with this prior.',
            atypical: 'Meningiomas are rarely deep / intraparenchymal. A meningioma label with attention deep in the cerebrum is atypical — recommend review.',
        },
        Pituitary: {
            typical:  'Pituitary adenomas arise in the sella turcica (deep central). Consistent with this prior.',
            atypical: 'Pituitary tumors occupy the sella turcica. Attention outside the deep central region is atypical — recommend review.',
        },
    },
    es: {
        Glioma: {
            typical:  'Los gliomas suelen surgir en los hemisferios cerebrales (frontal/temporal/parietal). La región de atención es consistente con este patrón.',
            atypical: 'Los gliomas de línea media (tronco encefálico, talámico, difuso de línea media) existen pero son menos comunes. Se recomienda revisión.',
        },
        Meningioma: {
            typical:  'Los meningiomas suelen surgir de superficies durales (parasagital, convexidad, hoz, línea media posterior). Consistente con este patrón.',
            atypical: 'Los meningiomas rara vez son profundos / intraparenquimatosos. Una etiqueta de meningioma con atención en zona cerebral profunda es atípica — se recomienda revisión.',
        },
        Pituitary: {
            typical:  'Los adenomas hipofisarios surgen en la silla turca (central profundo). Consistente con este patrón.',
            atypical: 'Los tumores hipofisarios ocupan la silla turca. La atención fuera de la región central profunda es atípica — se recomienda revisión.',
        },
    },
};

/* ── Clinical Context panel (hybrid: static medical primer + MedGemma fields) ── */
function ClinicalContextPanel({ data, predictedClass }) {
    const { t, lang } = useT();
    const serverCtx = data?.clinical_context;
    const localCtx  = CLINICAL_CONTEXT_LOCAL[lang]?.[predictedClass];
    // Prefer the locally-stored, language-reactive copy. Fall back to whatever
    // the backend shipped at upload time (covers unknown classes).
    const ctx = localCtx
        ? {
            primer: localCtx.primer,
            size_interpretation: localCtx.size_interpretation?.[data?.size_category] || serverCtx?.size_interpretation,
            subtypes: localCtx.subtypes,
            typical_workup: serverCtx?.typical_workup,
        }
        : serverCtx;
    const mg  = data?.medgemma_assessment || {};
    const serverRegion = data?.region || {};
    // Re-localize region.explanation against the current UI language.
    const consistency = serverRegion.consistency;
    const localExpl = REGION_EXPLANATION_LOCAL[lang]?.[predictedClass]?.[consistency];
    const region = { ...serverRegion, explanation: localExpl || serverRegion.explanation };

    // Guard: nothing to show if no clinical_context (e.g. No Tumor).
    if (!ctx) return null;

    const ACCENT = '#22d3ee';     // cyan — distinct from purple/orange used elsewhere
    const PURPLE = '#a78bfa';
    const PINK   = '#f472b6';

    // Grade → color / label
    const grade = (mg.grade_estimate || '').toLowerCase();
    const gradeMap = {
        low:          { label: t('cctx.grade.low'),         color: '#38ef7d' },
        intermediate: { label: t('cctx.grade.intermediate'),color: '#f5a623' },
        high:         { label: t('cctx.grade.high'),        color: '#f5576c' },
        not_applicable:{label: t('cctx.grade.na'),          color: '#94a3b8' },
    };
    const gradeShow = gradeMap[grade];

    // Growth pattern
    const growth = (mg.growth_pattern || '').toLowerCase();
    const growthMap = {
        focal:         t('cctx.growth.focal'),
        infiltrative:  t('cctx.growth.infiltrative'),
        mass_effect:   t('cctx.growth.mass_effect'),
        uncertain:     t('cctx.growth.uncertain'),
    };

    // Mass effect
    const mass = (mg.mass_effect || '').toLowerCase();
    const massMap = {
        none:     { label: t('cctx.mass.none'),     color: '#38ef7d' },
        mild:     { label: t('cctx.mass.mild'),     color: '#a3e635' },
        moderate: { label: t('cctx.mass.moderate'), color: '#f5a623' },
        severe:   { label: t('cctx.mass.severe'),   color: '#f5576c' },
    };
    const massShow = massMap[mass];

    const differential = Array.isArray(mg.differential) ? mg.differential.filter(Boolean) : [];

    // Mini-pill component
    const Pill = ({ label, value, color = ACCENT, tag }) => (
        <Box sx={{
            p: 1.5, borderRadius: 2,
            background: `linear-gradient(135deg, ${color}11 0%, rgba(255,255,255,0.02) 100%)`,
            border: `1px solid ${color}44`,
            position: 'relative',
            backdropFilter: 'blur(6px)',
        }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 0.4 }}>
                <Typography sx={{
                    fontSize: 9.5, letterSpacing: 1, textTransform: 'uppercase',
                    opacity: 0.7, fontWeight: 700, color,
                }}>{label}</Typography>
                {tag && (
                    <Box sx={{
                        fontSize: 8.5, fontWeight: 700, letterSpacing: 0.5,
                        px: 0.6, py: 0.1, borderRadius: 0.6,
                        background: `${color}22`, color, border: `1px solid ${color}55`,
                    }}>{tag}</Box>
                )}
            </Box>
            {value}
        </Box>
    );

    return (
        <Box sx={{ mt: 3, position: 'relative' }}>
            {/* Section header with futuristic divider */}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.2, mb: 0.4 }}>
                <Box sx={{
                    width: 4, height: 18, borderRadius: 1,
                    background: `linear-gradient(180deg, ${ACCENT}, ${PURPLE})`,
                    boxShadow: `0 0 8px ${ACCENT}66`,
                }} />
                <Typography sx={{
                    fontWeight: 800, fontSize: 14, letterSpacing: 0.8,
                    background: `linear-gradient(90deg, ${ACCENT}, ${PURPLE})`,
                    WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
                    textTransform: 'uppercase',
                }}>
                    {t('cctx.title')}
                </Typography>
            </Box>
            <Typography variant="caption" sx={{ display: 'block', opacity: 0.65, mb: 1.5, fontSize: 11 }}>
                {t('cctx.subtitle')}
            </Typography>

            <Box sx={{
                display: 'grid',
                gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)' },
                gap: 1.2,
            }}>
                {/* Type primer (static reference) */}
                {ctx.primer && (
                    <Box sx={{ gridColumn: { xs: '1', sm: '1 / -1' } }}>
                        <Pill
                            label={`${t('cctx.primer')} · ${predictedClass}`}
                            color={PURPLE}
                            tag={t('cctx.static_tag')}
                            value={
                                <Typography sx={{ fontSize: 12.5, lineHeight: 1.55, opacity: 0.92 }}>
                                    {ctx.primer}
                                </Typography>
                            }
                        />
                    </Box>
                )}

                {/* Size interpretation (static rule applied) */}
                {ctx.size_interpretation && (
                    <Pill
                        label={t('cctx.size_meaning')}
                        color={ACCENT}
                        tag={t('cctx.static_tag')}
                        value={
                            <Typography sx={{ fontSize: 12.5, lineHeight: 1.5, opacity: 0.92 }}>
                                {ctx.size_interpretation}
                            </Typography>
                        }
                    />
                )}

                {/* Location interpretation (already had this, surface it here) */}
                {region.explanation && (
                    <Pill
                        label={t('cctx.location_meaning')}
                        color={ACCENT}
                        tag={t('cctx.static_tag')}
                        value={
                            <Typography sx={{ fontSize: 12.5, lineHeight: 1.5, opacity: 0.92 }}>
                                {region.explanation}
                            </Typography>
                        }
                    />
                )}

                {/* MedGemma badges row: grade · growth · mass effect */}
                {(gradeShow || growthMap[growth] || massShow) && (
                    <Box sx={{ gridColumn: { xs: '1', sm: '1 / -1' } }}>
                        <Pill
                            label={t('cctx.scan_specific')}
                            color={PINK}
                            tag={t('cctx.medgemma_tag')}
                            value={
                                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 0.5 }}>
                                    {gradeShow && (
                                        <Box sx={{
                                            px: 1.2, py: 0.4, borderRadius: 1.2, fontSize: 11, fontWeight: 700,
                                            background: `${gradeShow.color}22`, color: gradeShow.color,
                                            border: `1px solid ${gradeShow.color}55`,
                                        }}>
                                            {t('cctx.grade')}: {gradeShow.label}
                                        </Box>
                                    )}
                                    {growthMap[growth] && (
                                        <Box sx={{
                                            px: 1.2, py: 0.4, borderRadius: 1.2, fontSize: 11, fontWeight: 700,
                                            background: 'rgba(167,139,250,0.18)', color: PURPLE,
                                            border: `1px solid ${PURPLE}55`,
                                        }}>
                                            {t('cctx.growth')}: {growthMap[growth]}
                                        </Box>
                                    )}
                                    {massShow && (
                                        <Box sx={{
                                            px: 1.2, py: 0.4, borderRadius: 1.2, fontSize: 11, fontWeight: 700,
                                            background: `${massShow.color}22`, color: massShow.color,
                                            border: `1px solid ${massShow.color}55`,
                                        }}>
                                            {t('cctx.mass_effect')}: {massShow.label}
                                        </Box>
                                    )}
                                </Box>
                            }
                        />
                    </Box>
                )}

                {/* Differential considerations (MedGemma) */}
                {differential.length > 0 && (
                    <Pill
                        label={t('cctx.differential')}
                        color={PINK}
                        tag={t('cctx.medgemma_tag')}
                        value={
                            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.6, mt: 0.5 }}>
                                {differential.map((d, i) => (
                                    <Box key={i} sx={{
                                        px: 1, py: 0.3, borderRadius: 1, fontSize: 11.5, fontWeight: 600,
                                        background: 'rgba(244,114,182,0.12)', color: '#fff',
                                        border: '1px solid rgba(244,114,182,0.35)',
                                    }}>{d}</Box>
                                ))}
                            </Box>
                        }
                    />
                )}

                {/* Subtypes — kept as compact reference (not duplicated in MedGemma impression).
                    "Next step" and "Typical workup" pills intentionally removed: they overlap
                    with the Diagnostic Impression card below. */}
                {Array.isArray(ctx.subtypes) && ctx.subtypes.length > 0 && (
                    <Box sx={{ gridColumn: { xs: '1', sm: '1 / -1' } }}>
                        <Pill
                            label={t('cctx.subtypes')}
                            color={PURPLE}
                            tag={t('cctx.static_tag')}
                            value={
                                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.6, mt: 0.5 }}>
                                    {ctx.subtypes.map((s, i) => (
                                        <Box key={i} sx={{
                                            px: 1, py: 0.3, borderRadius: 1, fontSize: 11, fontWeight: 600,
                                            background: 'rgba(167,139,250,0.12)', color: '#e9d5ff',
                                            border: '1px solid rgba(167,139,250,0.35)',
                                        }}>{s}</Box>
                                    ))}
                                </Box>
                            }
                        />
                    </Box>
                )}
            </Box>
        </Box>
    );
}


/* ── Brain Atlas Page (3D interactive brain) ─────────────────── */
/* Region positions calibrated for the NIH3D mesh (after auto-fit center+scale).
   The model uses NIH3D RAS convention: X = left/right, Y = anterior/posterior
   (A+), Z = inferior/superior (S+). After auto-scaling so the mesh fits a
   ~2.8-unit diameter and is centered at origin, the regions sit in the
   following scene-space coordinates:                                            */
/* Positions / scales tuned for the NIH3D mesh after it auto-fits to a ~4.2-unit
   diameter. Axes (NIH3D RAS+): X = left/right (L−, R+), Y = anterior/posterior
   (A+), Z = inferior/superior (S+).                                              */
const BRAIN_REGIONS = [
    // Frontal lobes
    { id: 'frontal_l',   pos: [-0.75,  1.30,  0.60], scale: [0.80, 0.80, 0.80], color: '#a78bfa' },
    { id: 'frontal_r',   pos: [ 0.75,  1.30,  0.60], scale: [0.80, 0.80, 0.80], color: '#a78bfa' },
    // Parietal lobes
    { id: 'parietal_l',  pos: [-0.65,  0.05,  1.00], scale: [0.80, 0.80, 0.65], color: '#22d3ee' },
    { id: 'parietal_r',  pos: [ 0.65,  0.05,  1.00], scale: [0.80, 0.80, 0.65], color: '#22d3ee' },
    // Temporal lobes
    { id: 'temporal_l',  pos: [-1.25,  0.30, -0.05], scale: [0.65, 0.80, 0.75], color: '#f5a623' },
    { id: 'temporal_r',  pos: [ 1.25,  0.30, -0.05], scale: [0.65, 0.80, 0.75], color: '#f5a623' },
    // Occipital lobes
    { id: 'occipital_l', pos: [-0.50, -1.30,  0.55], scale: [0.65, 0.80, 0.65], color: '#f472b6' },
    { id: 'occipital_r', pos: [ 0.50, -1.30,  0.55], scale: [0.65, 0.80, 0.65], color: '#f472b6' },
    // Cerebellum
    { id: 'cerebellum',  pos: [ 0.00, -1.50, -0.60], scale: [0.95, 0.65, 0.60], color: '#38ef7d' },
    // Brainstem
    { id: 'brainstem',   pos: [ 0.00, -0.60, -1.10], scale: [0.32, 0.50, 0.38], color: '#94a3b8' },
    // Pituitary (sellar)
    { id: 'pituitary',   pos: [ 0.00,  0.30, -0.80], scale: [0.28, 0.28, 0.28], color: '#fde047' },
];

/* Map backend `malignancy.region.label` → an atlas region id (one of the 11
   in BRAIN_REGIONS). Language-agnostic: laterality comes from `side` (which
   the backend keeps in English: left/right/midline) and lobe identification
   relies on roots that are the same in EN and ES — "frontal", "parietal",
   "temporal", "occipital", and the ES-only stems "cerebel" and "tronco". */
function regionLabelToAtlasId(label, side, predClass) {
    if (!label) return null;
    const L = label.toLowerCase();
    const cls = (predClass || '').toLowerCase();
    const isLeft  = side === 'left';
    const isRight = side === 'right';

    if (cls.includes('pituitary'))                          return 'pituitary';
    // EN: "sellar" / ES: "sellar" or "hipofisaria"
    if (L.includes('sellar') || L.includes('hipofisaria'))  return 'pituitary';
    // Posterior midline / cerebellum / brainstem
    // EN: "cerebellum" "brainstem" "posterior midline"
    // ES: "cerebelo" "tronco" "línea media posterior"
    if (L.includes('tronco'))                               return 'brainstem';
    if (L.includes('cerebel')                                  // cerebellum / cerebelo
        || L.includes('posterior midline')
        || L.includes('línea media posterior')
        || L.includes('linea media posterior'))             return 'cerebellum';
    // Lobes — same root in EN and ES, laterality from `side`
    if (L.includes('frontal')) {
        if (isLeft)  return 'frontal_l';
        if (isRight) return 'frontal_r';
    }
    if (L.includes('temporal') || L.includes('parietal')) {
        if (isLeft)  return 'temporal_l';
        if (isRight) return 'temporal_r';
    }
    if (L.includes('occipital')) {
        if (isLeft)  return 'occipital_l';
        if (isRight) return 'occipital_r';
    }
    return null;
}

/* X-ray brain — unlit MeshBasicMaterial with additive blending.
   Every face contributes a tiny amount of white to the framebuffer; where
   many faces overlap (gyri folds, the subcortical inside the hemispheres)
   they accumulate into bright white, while thin single-layer regions stay
   faint. Same material across all 3 GLBs so subcortical isn't purple — it
   just glows brighter because the cortex sits in front and behind it. */
function makeGlassBrainMaterial() {
    return new THREE.MeshBasicMaterial({
        color:       0xffffff,
        transparent: true,
        opacity:     0.08,
        depthWrite:  false,
        side:        THREE.DoubleSide,
        blending:    THREE.AdditiveBlending,
    });
}

/* NIH 3D Print Exchange brain (CC0) — left hemisphere + right hemisphere +
   subcortical structures (thalamus, basal ganglia, cerebellum, brainstem).
   Each GLB is loaded by useGLTF, traversed once on mount to replace its
   materials with the glass-fresnel shader, then mounted as a <primitive>.
   We also center+scale the whole group to fit our camera framing.          */
/* DTI white-matter tracts — lazily mounted child so the ~58 MB GLB is only
   fetched when the user actually toggles them on. */
function DTITracts() {
    const { scene } = useGLTF('/models/dti.glb');
    const tractMat = useMemo(() => new THREE.MeshBasicMaterial({
        color:       0x66e0ff,
        transparent: true,
        opacity:     0.22,
        depthWrite:  false,
        side:        THREE.DoubleSide,
        blending:    THREE.AdditiveBlending,
    }), []);
    useEffect(() => {
        scene.traverse(obj => {
            if (obj.isMesh) {
                obj.material = tractMat;
                obj.frustumCulled = false;
                obj.renderOrder = 2;
            }
        });
    }, [scene, tractMat]);
    return <primitive object={scene} />;
}

function NIH3DBrain({ showTracts }) {
    const { scene: lh } = useGLTF('/models/lh.glb');
    const { scene: rh } = useGLTF('/models/rh.glb');
    const { scene: sc } = useGLTF('/models/subcortical.glb');

    // One glass material for the entire brain (cortex + subcortical) so the
    // subcortical structures are the same off-white as the hemispheres — light
    // passes through both via transmission, giving the layered x-ray effect
    // without requiring a separate color for the inner mesh.
    const glassMat = useMemo(() => makeGlassBrainMaterial(), []);

    useEffect(() => {
        [lh, rh, sc].forEach(root => {
            root.traverse(obj => {
                if (obj.isMesh) {
                    obj.material = glassMat;
                    obj.frustumCulled = false;
                    obj.renderOrder = 1;
                }
            });
        });
    }, [lh, rh, sc, glassMat]);

    // Compute the combined bounding box once to center + scale the group.
    // Defensive: ensure world matrices are up to date and fall back to a
    // reasonable fixed scale if the box ends up empty/degenerate.
    const fit = useMemo(() => {
        const box = new THREE.Box3();
        [lh, rh, sc].forEach(s => {
            s.updateMatrixWorld(true);
            box.expandByObject(s);
        });
        if (box.isEmpty() || !isFinite(box.min.x)) {
            return { scale: 0.015, center: new THREE.Vector3() };
        }
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        const targetDiameter = 4.2;     // bigger brain on screen
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = maxDim > 0.0001 ? targetDiameter / maxDim : 0.022;
        return { scale, center };
    }, [lh, rh, sc]);

    return (
        <group
            scale={fit.scale}
            position={[-fit.center.x * fit.scale, -fit.center.y * fit.scale, -fit.center.z * fit.scale]}
        >
            <primitive object={lh} />
            <primitive object={rh} />
            <primitive object={sc} />
            {showTracts && (
                <Suspense fallback={null}>
                    <DTITracts />
                </Suspense>
            )}
        </group>
    );
}

// Tell drei to preload the GLBs so the brain shows up immediately when the
// Atlas tab opens (the files are cached after the first visit anyway).
// DTI is NOT preloaded — only fetched when the user toggles it on.
useGLTF.preload('/models/lh.glb');
useGLTF.preload('/models/rh.glb');
useGLTF.preload('/models/subcortical.glb');


/* (kept for reference — procedural fallback, no longer used)                */
function makeBrainGeometry({ rx = 1.40, ry = 1.18, rz = 1.55, noiseAmp = 0.08, segments = 110 }) {
    const geom = new THREE.SphereGeometry(1, segments, Math.floor(segments * 0.66));
    const pos = geom.attributes.position;
    const tmp = new THREE.Vector3();
    for (let i = 0; i < pos.count; i++) {
        tmp.fromBufferAttribute(pos, i);
        const ox = tmp.x, oy = tmp.y, oz = tmp.z;
        // Ellipsoid base
        tmp.x = ox * rx;
        tmp.y = oy * ry;
        tmp.z = oz * rz;
        // Multi-octave noise displacement → gyri / sulci
        const n =
            Math.sin(ox * 3.5)  * Math.cos(oy * 3.5)  * 0.40 +
            Math.sin(oy * 5.5)  * Math.cos(oz * 5.5)  * 0.25 +
            Math.sin(ox * 9.0)  * Math.cos(oz * 9.0)  * 0.18 +
            Math.sin((ox+oy+oz) * 16.0) * 0.12;
        tmp.x += ox * n * noiseAmp;
        tmp.y += oy * n * noiseAmp;
        tmp.z += oz * n * noiseAmp;
        // Longitudinal fissure (gentle cleft at midline, top)
        if (Math.abs(ox) < 0.18 && oy > 0.05) {
            const cleft = (0.18 - Math.abs(ox)) * 0.18;
            tmp.y -= cleft * 1.2;
            tmp.x *= 1 - cleft * 0.6;
        }
        // Frontal bulge slightly forward
        if (oz > 0.6) tmp.z += (oz - 0.6) * 0.12;
        pos.setXYZ(i, tmp.x, tmp.y, tmp.z);
    }
    pos.needsUpdate = true;
    geom.computeVertexNormals();
    return geom;
}

function ProceduralBrain() {
    const cerebrum   = useMemo(() => makeBrainGeometry({ rx: 1.40, ry: 1.18, rz: 1.55, noiseAmp: 0.085, segments: 120 }), []);
    const cerebellum = useMemo(() => makeBrainGeometry({ rx: 0.95, ry: 0.50, rz: 0.62, noiseAmp: 0.16,  segments: 64  }), []);

    const matRef1 = useRef();
    const matRef2 = useRef();
    // Subtle breathing of the clearcoat for an organic feel
    useFrame(({ clock }) => {
        const t = clock.getElapsedTime();
        if (matRef1.current) matRef1.current.clearcoat = 0.35 + Math.sin(t * 0.6) * 0.05;
        if (matRef2.current) matRef2.current.clearcoat = 0.35 + Math.sin(t * 0.6 + 1) * 0.05;
    });

    return (
        <group>
            {/* Cerebrum */}
            <mesh geometry={cerebrum} position={[0, 0.18, 0]} castShadow receiveShadow>
                <meshPhysicalMaterial
                    ref={matRef1}
                    color="#3d3a48"
                    metalness={0.05}
                    roughness={0.55}
                    clearcoat={0.4}
                    clearcoatRoughness={0.55}
                    sheen={0.45}
                    sheenColor="#a78bfa"
                    sheenRoughness={0.7}
                />
            </mesh>
            {/* Cerebellum (smaller, lower-back) */}
            <mesh geometry={cerebellum} position={[0, -0.75, -0.95]} rotation={[0.15, 0, 0]}>
                <meshPhysicalMaterial
                    ref={matRef2}
                    color="#36333f"
                    metalness={0.05}
                    roughness={0.5}
                    clearcoat={0.4}
                    clearcoatRoughness={0.55}
                    sheen={0.4}
                    sheenColor="#22d3ee"
                    sheenRoughness={0.7}
                />
            </mesh>
            {/* Brainstem (thin, downward) */}
            <mesh position={[0, -1.05, -0.30]} rotation={[0.25, 0, 0]}>
                <cylinderGeometry args={[0.16, 0.20, 0.55, 24]} />
                <meshPhysicalMaterial
                    color="#34313c"
                    metalness={0.05}
                    roughness={0.6}
                    clearcoat={0.3}
                    sheen={0.3}
                    sheenColor="#a78bfa"
                />
            </mesh>
        </group>
    );
}


/* OrbitControls wrapper. Adds:
   - Snappier rotation feel (lower damping, custom rotateSpeed)
   - Slower, less dizzying auto-rotate when idle
   - Cinematic camera animation to focus a region when one is selected
     (lerps camera.position + ctrl.target toward the region's outward normal). */
function BrainControls({ autoRotate, activeRegion }) {
    const { camera, gl } = useThree();
    const ctrlRef = useRef();
    const animating = useRef(false);
    const targetPos = useRef(new THREE.Vector3());
    const targetLook = useRef(new THREE.Vector3());

    useEffect(() => {
        // NIH3D mesh: anterior = +Y, superior = +Z, right = +X.
        // Tell the camera that "up" is the top of the head (+Z) so OrbitControls
        // orbits around the vertical brain axis (otherwise rotation feels like
        // it's tumbling because three.js defaults up to +Y).
        camera.up.set(0, 0, 1);
        camera.lookAt(0, 0, 0);
        const ctrl = new ThreeOrbitControls(camera, gl.domElement);
        ctrl.enableDamping = true;
        ctrl.dampingFactor = 0.12;
        ctrl.minDistance = 3.5;
        ctrl.maxDistance = 14;
        ctrl.enablePan = false;
        ctrl.rotateSpeed = 0.8;
        ctrl.zoomSpeed = 0.8;
        ctrl.autoRotate = autoRotate;
        ctrl.autoRotateSpeed = 0.35;
        ctrl.target.set(0, 0, 0);
        ctrlRef.current = ctrl;
        // If user starts dragging, kill any in-flight camera animation
        const onStart = () => { animating.current = false; };
        ctrl.addEventListener('start', onStart);
        return () => {
            ctrl.removeEventListener('start', onStart);
            ctrl.dispose();
        };
    }, [camera, gl]);

    useEffect(() => {
        if (ctrlRef.current) ctrlRef.current.autoRotate = autoRotate;
    }, [autoRotate]);

    // When a region becomes active, queue a camera move to look at it from
    // outside (along the vector from origin to the region).
    useEffect(() => {
        if (!activeRegion || !ctrlRef.current) {
            animating.current = false;
            return;
        }
        const regionVec = new THREE.Vector3(...activeRegion.pos);
        const dir = regionVec.clone();
        if (dir.lengthSq() < 0.04) {
            // Region nearly at center (e.g. pituitary) — pick a default vantage
            dir.set(0.4, 0.3, 0.5);
        }
        dir.normalize();
        const dist = 6.0;
        // Slight superior offset (+Z is up since camera.up = +Z) for a 3/4 view
        targetPos.current.copy(dir).multiplyScalar(dist).add(new THREE.Vector3(0, 0, 0.4));
        targetLook.current.copy(regionVec);
        animating.current = true;
    }, [activeRegion]);

    useFrame(() => {
        const ctrl = ctrlRef.current;
        if (!ctrl) return;
        if (animating.current) {
            camera.position.lerp(targetPos.current, 0.08);
            ctrl.target.lerp(targetLook.current, 0.08);
            if (camera.position.distanceToSquared(targetPos.current) < 0.0008) {
                animating.current = false;
            }
        }
        ctrl.update();
    });
    return null;
}

/* Per-region representation: an INVISIBLE sphere acts as the raycast volume
   for hover/click (preserves the existing onPointerOver / onClick UX). When
   the region is active we additionally render a translucent emissive halo
   over the underlying procedural brain, slightly larger than the hit-volume
   so the highlight extends past the mesh surface and glows under Bloom. */
function BrainRegionMesh({ region, active, onEnter, onLeave, onClick }) {
    const haloRef = useRef();
    useFrame(() => {
        if (!haloRef.current) return;
        const m = haloRef.current.material;
        const targetEm = active ? 1.2 : 0;
        const targetOp = active ? 0.40 : 0;
        m.emissiveIntensity += (targetEm - m.emissiveIntensity) * 0.18;
        m.opacity += (targetOp - m.opacity) * 0.18;
    });
    const haloScale = region.scale.map(s => s * 1.10);
    return (
        <>
            {/* Invisible hit-volume — same geometry as before, no material rendering */}
            <mesh
                position={region.pos}
                scale={region.scale}
                onPointerOver={(e) => { e.stopPropagation(); onEnter(); }}
                onPointerOut={() => onLeave()}
                onClick={(e) => { e.stopPropagation(); onClick(); }}
            >
                <sphereGeometry args={[1, 24, 24]} />
                <meshBasicMaterial transparent opacity={0} depthWrite={false} />
            </mesh>
            {/* Visible halo (always mounted; opacity/emissive lerped via useFrame) */}
            <mesh
                ref={haloRef}
                position={region.pos}
                scale={haloScale}
                raycast={() => null}      // halo never blocks raycasts
            >
                <sphereGeometry args={[1, 32, 32]} />
                <meshStandardMaterial
                    color={region.color}
                    emissive={region.color}
                    emissiveIntensity={0}
                    transparent
                    opacity={0}
                    depthWrite={false}
                    blending={THREE.AdditiveBlending}
                    side={THREE.FrontSide}
                />
            </mesh>
        </>
    );
}

function BrainAtlasPage({ pinRequest, clearPinRequest }) {
    const { t } = useT();
    const [hovered, setHovered] = useState(null);
    const [pinned, setPinned] = useState(null);
    const [showTracts, setShowTracts] = useState(false);
    // Honor cross-page pin requests (e.g. coming from a report "View in atlas" button)
    useEffect(() => {
        if (pinRequest && BRAIN_REGIONS.some(r => r.id === pinRequest)) {
            setPinned(pinRequest);
            clearPinRequest && clearPinRequest();
        }
    }, [pinRequest, clearPinRequest]);
    const active = pinned || hovered;
    const region = BRAIN_REGIONS.find(r => r.id === active);
    // Camera only animates on click (pinned), not on hover — otherwise hovering
    // continuously yanks the camera around.
    const pinnedRegion = BRAIN_REGIONS.find(r => r.id === pinned);
    const PURPLE = '#a78bfa';

    return (
        <Box>
            <Paper className="glass-card" elevation={0} sx={{ p: 3, mb: 3 }}>
                <Typography variant="h5" sx={{ fontWeight: 700, color: '#667eea', mb: 0.5 }}>
                    {t('atlas.title')}
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.8 }}>
                    {t('atlas.subtitle')}
                </Typography>
            </Paper>

            <Box sx={{
                display: 'grid',
                gridTemplateColumns: { xs: '1fr', md: '2fr 1fr' },
                gap: 3,
            }}>
                {/* 3D Canvas */}
                <Paper className="glass-card" elevation={0} sx={{
                    p: 1, position: 'relative', overflow: 'hidden',
                    borderRadius: 3,
                    background: 'radial-gradient(circle at center, rgba(102,126,234,0.06) 0%, rgba(0,0,0,0.2) 100%)',
                }}>
                    <Canvas
                        camera={{ position: [0, 7.5, 1.2], fov: 38 }}
                        style={{ height: '70vh', minHeight: 480 }}
                        gl={{ antialias: true, alpha: true }}
                        onPointerMissed={() => setPinned(null)}
                    >
                        <color attach="background" args={['#000000']} />
                        {/* MeshBasicMaterial is unlit — these lights only feed the
                            emissive halo on the active region highlight. */}
                        <ambientLight intensity={0.6} />
                        <BrainControls autoRotate={false} activeRegion={pinnedRegion} />
                        <Suspense fallback={
                            <mesh>
                                <sphereGeometry args={[0.6, 24, 24]} />
                                <meshBasicMaterial color="#a78bfa" wireframe transparent opacity={0.4} />
                            </mesh>
                        }>
                            <NIH3DBrain showTracts={showTracts} />
                        </Suspense>
                        {/* Invisible hit-volumes + emissive halos */}
                        {BRAIN_REGIONS.map(r => (
                            <BrainRegionMesh
                                key={r.id}
                                region={r}
                                active={active === r.id}
                                onEnter={() => setHovered(r.id)}
                                onLeave={() => setHovered(null)}
                                onClick={() => setPinned(pinned === r.id ? null : r.id)}
                            />
                        ))}
                        {/* Bloom sized for the fresnel rim glow + region halos.
                            Threshold is lowered because the fresnel edges aren't
                            saturated white, just bright off-white. */}
                        <EffectComposer>
                            <Bloom intensity={0.45} luminanceThreshold={0.55} luminanceSmoothing={0.35} mipmapBlur />
                        </EffectComposer>
                    </Canvas>
                    <Box sx={{
                        position: 'absolute', bottom: 10, left: 16,
                        display: 'flex', alignItems: 'center', gap: 0.8,
                        background: 'rgba(0,0,0,0.45)', borderRadius: 1.5,
                        px: 1.2, py: 0.5,
                    }}>
                        <Typography sx={{ fontSize: 10.5, opacity: 0.85 }}>
                            {t('atlas.hint')}
                        </Typography>
                    </Box>
                    {/* DTI tracts toggle (top-right overlay) */}
                    <Box
                        onClick={() => setShowTracts(s => !s)}
                        sx={{
                            position: 'absolute', top: 12, right: 14,
                            cursor: 'pointer', userSelect: 'none',
                            display: 'flex', alignItems: 'center', gap: 0.8,
                            background: showTracts ? 'rgba(102,224,255,0.18)' : 'rgba(0,0,0,0.45)',
                            border: `1px solid ${showTracts ? 'rgba(102,224,255,0.6)' : 'rgba(255,255,255,0.18)'}`,
                            borderRadius: 1.5, px: 1.3, py: 0.6,
                            transition: 'all 0.18s',
                            '&:hover': { background: 'rgba(102,224,255,0.12)' },
                        }}
                    >
                        <Box sx={{
                            width: 8, height: 8, borderRadius: '50%',
                            background: showTracts ? '#66e0ff' : 'rgba(255,255,255,0.3)',
                            boxShadow: showTracts ? '0 0 8px #66e0ff' : 'none',
                        }} />
                        <Typography sx={{ fontSize: 11, fontWeight: 600, opacity: 0.9 }}>
                            {t('atlas.tracts')}
                        </Typography>
                    </Box>
                </Paper>

                {/* Info panel */}
                <Paper className="glass-card" elevation={0} sx={{
                    p: 3, overflowY: 'auto', maxHeight: '70vh', minHeight: 480,
                    borderRadius: 3,
                }}>
                    {region ? (
                        <Box>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 2 }}>
                                <Box sx={{
                                    width: 14, height: 14, borderRadius: '50%',
                                    background: region.color,
                                    boxShadow: `0 0 14px ${region.color}aa`,
                                }} />
                                <Typography sx={{
                                    fontWeight: 800, fontSize: 18, color: region.color,
                                }}>
                                    {t(`atlas.region.${region.id}.name`)}
                                </Typography>
                            </Box>
                            <Typography sx={{
                                fontSize: 10, letterSpacing: 1.5, textTransform: 'uppercase',
                                opacity: 0.55, fontWeight: 700, mb: 0.8,
                            }}>
                                {t('atlas.functions')}
                            </Typography>
                            <Typography sx={{
                                fontSize: 12.5, lineHeight: 1.65, mb: 2.5,
                                whiteSpace: 'pre-line', opacity: 0.92,
                            }}>
                                {t(`atlas.region.${region.id}.functions`)}
                            </Typography>
                            <Box sx={{
                                p: 1.8, borderRadius: 1.5,
                                background: `${region.color}13`,
                                border: `1px solid ${region.color}44`,
                            }}>
                                <Typography sx={{
                                    fontSize: 10, letterSpacing: 1.2, textTransform: 'uppercase',
                                    opacity: 0.85, fontWeight: 700, color: region.color, mb: 0.6,
                                }}>
                                    {t('atlas.tumor_link')}
                                </Typography>
                                <Typography sx={{ fontSize: 12, lineHeight: 1.55 }}>
                                    {t(`atlas.region.${region.id}.tumor`)}
                                </Typography>
                            </Box>
                        </Box>
                    ) : (
                        <Box sx={{ textAlign: 'center', py: 6 }}>
                            <Box sx={{
                                fontSize: 32, mb: 2, opacity: 0.6,
                                color: PURPLE,
                            }}>◉</Box>
                            <Typography sx={{ fontSize: 14, opacity: 0.85, mb: 1, fontWeight: 600 }}>
                                {t('atlas.placeholder.title')}
                            </Typography>
                            <Typography sx={{ fontSize: 12, opacity: 0.6 }}>
                                {t('atlas.placeholder.body')}
                            </Typography>
                        </Box>
                    )}

                    {/* Clickable region picker — each entry lights up that zone on
                        the 3D brain. Doubles as a legend (colored dot per region). */}
                    <Box sx={{ mt: 3, pt: 2, borderTop: '1px solid rgba(255,255,255,0.08)' }}>
                        <Typography sx={{
                            fontSize: 9.5, letterSpacing: 1.5, textTransform: 'uppercase',
                            opacity: 0.55, fontWeight: 700, mb: 1.2,
                        }}>
                            {t('atlas.picker')}
                        </Typography>
                        {BRAIN_REGIONS.map(r => {
                            const isActive = active === r.id;
                            return (
                                <Box key={r.id}
                                    onClick={() => setPinned(pinned === r.id ? null : r.id)}
                                    onMouseEnter={() => setHovered(r.id)}
                                    onMouseLeave={() => setHovered(null)}
                                    sx={{
                                        display: 'flex', alignItems: 'center', gap: 1,
                                        px: 1, py: 0.55, mb: 0.3, borderRadius: 1,
                                        cursor: 'pointer', userSelect: 'none',
                                        background: isActive ? `${r.color}22` : 'transparent',
                                        border: `1px solid ${isActive ? r.color + '66' : 'transparent'}`,
                                        transition: 'background .12s ease, border-color .12s ease',
                                        '&:hover': {
                                            background: isActive ? `${r.color}33` : 'rgba(255,255,255,0.04)',
                                        },
                                    }}>
                                    <Box sx={{
                                        width: 10, height: 10, borderRadius: '50%',
                                        background: r.color,
                                        boxShadow: isActive ? `0 0 8px ${r.color}aa` : 'none',
                                    }} />
                                    <Typography sx={{
                                        fontSize: 11.5,
                                        opacity: isActive ? 1 : 0.82,
                                        fontWeight: isActive ? 700 : 500,
                                        color: isActive ? r.color : '#fff',
                                    }}>
                                        {t(`atlas.region.${r.id}.name`)}
                                    </Typography>
                                </Box>
                            );
                        })}
                        <Typography sx={{
                            display: 'block', fontSize: 10, opacity: 0.45, mt: 2, lineHeight: 1.4,
                            fontStyle: 'italic',
                        }}>
                            {t('atlas.model_credit')}
                        </Typography>
                    </Box>
                </Paper>
            </Box>
        </Box>
    );
}


/* ── Language toggle (EN / ES) ─────────────────────────── */
function LanguageToggle() {
    const { lang, setLang } = useT();
    const make = (code, label) => (
        <Box
            onClick={() => setLang(code)}
            sx={{
                px: 1.4, py: 0.4, borderRadius: 1.5, cursor: 'pointer',
                fontSize: 12, fontWeight: 700, letterSpacing: 0.8,
                color: lang === code ? '#fff' : 'rgba(255,255,255,0.55)',
                background: lang === code
                    ? 'linear-gradient(135deg,#667eea 0%,#764ba2 100%)'
                    : 'rgba(255,255,255,0.06)',
                border: `1px solid ${lang === code ? 'transparent' : 'rgba(255,255,255,0.12)'}`,
                transition: 'all .15s ease',
                userSelect: 'none',
            }}
        >
            {label}
        </Box>
    );
    return (
        <Box sx={{ display: 'flex', gap: 0.5, alignItems: 'center' }}>
            {make('en', 'EN')}
            {make('es', 'ES')}
        </Box>
    );
}


/* ── MedGemma Diagnostic Impression Card ────────────────── */

// Per-section visual styling — color + emoji accent + short purpose hint.
// Keys match the markdown headings emitted by the basic/advanced prompts.
const REPORT_SECTION_META = {
    // basic mode (en)
    'what we found':              { color: '#667eea', icon: '🔎' },
    'what it might mean':         { color: '#f5a623', icon: '💡' },
    'how sure the program is':    { color: '#38ef7d', icon: '📊' },
    'what to do next':            { color: '#f5576c', icon: '🩺' },
    'important to know':          { color: '#94a3b8', icon: 'ℹ️' },
    // basic mode (es)
    'qué encontramos':            { color: '#667eea', icon: '🔎' },
    'qué podría significar':      { color: '#f5a623', icon: '💡' },
    'qué tan seguro está el programa': { color: '#38ef7d', icon: '📊' },
    'qué hacer ahora':            { color: '#f5576c', icon: '🩺' },
    'es importante saber':        { color: '#94a3b8', icon: 'ℹ️' },
    // advanced mode (en)
    'impression':                 { color: '#667eea', icon: '🧠' },
    'imaging findings':           { color: '#f5a623', icon: '📷' },
    'differential diagnosis':     { color: '#f5576c', icon: '🔀' },
    'recommended workup':         { color: '#38ef7d', icon: '🧪' },
    'limitations':                { color: '#94a3b8', icon: '⚠️' },
    // advanced mode (es)
    'impresión':                  { color: '#667eea', icon: '🧠' },
    'hallazgos de imagen':        { color: '#f5a623', icon: '📷' },
    'diagnóstico diferencial':    { color: '#f5576c', icon: '🔀' },
    'estudios recomendados':      { color: '#38ef7d', icon: '🧪' },
    'limitaciones':               { color: '#94a3b8', icon: '⚠️' },
};

function parseReportMarkdown(text) {
    /* Split the report into { heading, body } sections plus a trailing
       disclaimer (any text after the last section that isn't a heading). */
    if (!text) return { sections: [], trailer: '' };
    const lines = text.replace(/\r\n/g, '\n').split('\n');
    const sections = [];
    let current = null;
    const trailerLines = [];
    let sawAnyHeading = false;
    for (const raw of lines) {
        const line = raw.trimEnd();
        const m = line.match(/^##\s+(.+?)\s*$/);
        if (m) {
            if (current) sections.push(current);
            current = { heading: m[1], body: [] };
            sawAnyHeading = true;
            continue;
        }
        if (!sawAnyHeading) continue; // ignore preamble before first heading
        if (current) current.body.push(line);
    }
    if (current) sections.push(current);

    // Pull any trailing "no heading" lines off the last section into the trailer
    // (the prompts terminate with a plain-text disclaimer paragraph).
    if (sections.length) {
        const last = sections[sections.length - 1];
        // Walk backwards over consecutive non-empty lines that look like the disclaimer:
        // a long sentence ending with "regulatory authority." or its Spanish equivalent.
        const text2 = last.body.join('\n').trim();
        const disclaimerMatch = text2.match(
            /(.*?)(\n\n[^#]*?(?:regulatory authority\.|autoridad regulatoria\.)\s*)$/s
        );
        if (disclaimerMatch) {
            last.body = disclaimerMatch[1].split('\n');
            trailerLines.push(disclaimerMatch[2].trim());
        }
    }

    // Normalize: trim leading/trailing blanks per section
    sections.forEach(s => {
        while (s.body.length && !s.body[0].trim()) s.body.shift();
        while (s.body.length && !s.body[s.body.length - 1].trim()) s.body.pop();
    });
    return { sections, trailer: trailerLines.join('\n') };
}

function ReportSectionBlock({ heading, body }) {
    const meta = REPORT_SECTION_META[heading.toLowerCase().trim()] || { color: '#94a3b8', icon: '•' };
    // Detect bullet lines ("- ..."), render as a list; otherwise render as paragraphs.
    const lines = body.filter(l => l.trim().length > 0);
    const bullets = lines.filter(l => /^[-*]\s+/.test(l.trim()));
    const useBullets = bullets.length >= 2 && bullets.length === lines.length;
    return (
        <Box sx={{
            mt: 1.5, p: 2, borderRadius: 2,
            background: 'rgba(255,255,255,0.04)',
            border: '1px solid rgba(255,255,255,0.07)',
            borderLeft: `3px solid ${meta.color}`,
            transition: 'all .25s ease',
        }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.8 }}>
                <Typography sx={{ fontSize: 15 }}>{meta.icon}</Typography>
                <Typography variant="subtitle2" sx={{
                    fontWeight: 700, color: meta.color, letterSpacing: 0.3,
                    textTransform: 'uppercase', fontSize: 12,
                }}>
                    {heading}
                </Typography>
            </Box>
            {useBullets ? (
                <Box component="ul" sx={{
                    pl: 2.5, m: 0, listStyleType: 'disc',
                    '& li': { mb: 0.6, lineHeight: 1.55, fontSize: 14, color: 'rgba(255,255,255,0.92)' },
                }}>
                    {lines.map((l, i) => (
                        <li key={i}>{l.replace(/^[-*]\s+/, '')}</li>
                    ))}
                </Box>
            ) : (
                lines.map((l, i) => (
                    <Typography key={i} variant="body2" sx={{
                        mb: 0.6, lineHeight: 1.65, color: 'rgba(255,255,255,0.92)', fontSize: 14,
                    }}>
                        {l}
                    </Typography>
                ))
            )}
        </Box>
    );
}

/* ── Anatomy views grid — multi-colormap MRI views for Advanced mode ──
   Renders 2x3 grid of decent-sized thumbnails. Click any tile to open
   a full-screen-ish dialog with the image at large scale + label. */
function AnatomyViewsGrid({ views }) {
    const { t } = useT();
    const [zoomed, setZoomed] = useState(null);   // { key, label, accent, src }
    if (!views) return null;

    const items = [
        { key: 'original', label: t('anatomy.original'), accent: '#94a3b8' },
        { key: 'inverted', label: t('anatomy.inverted'), accent: '#cbd5e1' },
        { key: 'hot',      label: t('anatomy.hot'),      accent: '#f5576c' },
        { key: 'jet',      label: t('anatomy.jet'),      accent: '#22d3ee' },
        { key: 'bone',     label: t('anatomy.bone'),     accent: '#a3b3ff' },
        { key: 'viridis',  label: t('anatomy.viridis'),  accent: '#a78bfa' },
    ].filter(it => views[it.key]);
    if (items.length === 0) return null;

    return (
        <>
            <Paper className="glass-card" elevation={0}
                sx={{
                    p: 3, mt: 2.5, borderRadius: 3,
                    background: 'linear-gradient(135deg, rgba(34,211,238,0.05) 0%, rgba(167,139,250,0.05) 100%)',
                    border: '1px solid rgba(167,139,250,0.25)',
                }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.2, mb: 0.5 }}>
                    <Box sx={{
                        width: 4, height: 18, borderRadius: 1,
                        background: 'linear-gradient(180deg, #22d3ee, #a78bfa)',
                        boxShadow: '0 0 8px rgba(34,211,238,0.5)',
                    }} />
                    <Typography sx={{
                        fontWeight: 800, fontSize: 14, letterSpacing: 0.8,
                        background: 'linear-gradient(90deg, #67e8f9, #c4b5fd)',
                        WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
                        textTransform: 'uppercase',
                    }}>
                        {t('anatomy.title')}
                    </Typography>
                </Box>
                <Typography variant="caption" sx={{ display: 'block', opacity: 0.7, mb: 1.5, fontSize: 11 }}>
                    {t('anatomy.subtitle')} · {t('anatomy.clickHint')}
                </Typography>
                <Box sx={{
                    display: 'grid',
                    gridTemplateColumns: { xs: 'repeat(2, 1fr)', sm: 'repeat(3, 1fr)' },
                    gap: 1.2,
                }}>
                    {items.map(it => {
                        const src = `data:image/png;base64,${views[it.key]}`;
                        return (
                            <Box key={it.key}
                                onClick={() => setZoomed({ ...it, src })}
                                sx={{
                                    borderRadius: 1.5, overflow: 'hidden',
                                    background: 'rgba(0,0,0,0.25)',
                                    border: `1px solid ${it.accent}55`,
                                    cursor: 'zoom-in',
                                    transition: 'transform .15s ease, box-shadow .15s ease, border-color .15s ease',
                                    '&:hover': {
                                        transform: 'translateY(-2px)',
                                        borderColor: it.accent,
                                        boxShadow: `0 6px 18px ${it.accent}44`,
                                    },
                                }}>
                                <Box component="img"
                                    src={src}
                                    alt={it.label}
                                    sx={{ width: '100%', display: 'block', aspectRatio: '1', objectFit: 'cover' }} />
                                <Typography sx={{
                                    fontSize: 10.5, fontWeight: 700, letterSpacing: 0.6,
                                    color: it.accent, textTransform: 'uppercase',
                                    textAlign: 'center', py: 0.6,
                                    borderTop: `1px solid ${it.accent}33`,
                                }}>{it.label}</Typography>
                            </Box>
                        );
                    })}
                </Box>
            </Paper>

            {/* Zoom dialog */}
            <Dialog
                open={!!zoomed}
                onClose={() => setZoomed(null)}
                maxWidth="lg" fullWidth
                PaperProps={{ sx: { background: 'rgba(15,15,25,0.95)', backdropFilter: 'blur(12px)' } }}
            >
                <DialogContent sx={{ p: 2, position: 'relative' }}>
                    {zoomed && (
                        <>
                            <Box sx={{
                                position: 'absolute', top: 12, left: 16, zIndex: 2,
                                px: 1.4, py: 0.5, borderRadius: 1.5,
                                background: `${zoomed.accent}22`, color: zoomed.accent,
                                border: `1px solid ${zoomed.accent}66`,
                                fontSize: 12, fontWeight: 800, letterSpacing: 0.8,
                                textTransform: 'uppercase',
                            }}>
                                {zoomed.label}
                            </Box>
                            <Box component="img"
                                src={zoomed.src}
                                alt={zoomed.label}
                                sx={{
                                    width: '100%', maxHeight: '85vh',
                                    objectFit: 'contain', display: 'block',
                                    borderRadius: 2,
                                }} />
                        </>
                    )}
                </DialogContent>
            </Dialog>
        </>
    );
}


function ReportCard({ r, mode: modeProp, setMode: setModeProp, onGoToAtlas }) {
    const { t, lang } = useT();
    const [report, setReport] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    // Allow parent to control mode (so other panels can react to Basic/Advanced).
    const [internalMode, setInternalMode] = useState('basic');
    const mode    = modeProp    !== undefined ? modeProp    : internalMode;
    const setMode = setModeProp !== undefined ? setModeProp : setInternalMode;

    const fetchReport = useCallback(async () => {
        if (!r) return;
        setLoading(true);
        setError(null);
        setReport(null);
        try {
            const resp = await axios.post(`${API_URL}/api/explain`,
                { prediction: r, language: lang, mode },
                { timeout: 420000 });
            if (resp.data?.success) setReport(resp.data);
            else setError(resp.data?.error || 'Unknown error');
        } catch (e) {
            setError(e.response?.data?.error || e.message || 'Network error');
        } finally {
            setLoading(false);
        }
    }, [r, lang, mode]);

    // Auto-fetch when a fresh prediction arrives, language changes, or mode flips
    useEffect(() => { fetchReport(); }, [fetchReport]);

    if (!r) return null;

    const parsed = report?.text ? parseReportMarkdown(report.text) : null;

    return (
        <Paper className="glass-card" elevation={0} sx={{ p: 4, mt: 4, borderLeft: '4px solid #667eea' }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 1, mb: 1 }}>
                <Typography variant="h6" sx={{ fontWeight: 700, color: '#667eea' }}>
                    {t('report.title')}
                </Typography>
                <Button size="small" onClick={fetchReport} disabled={loading}
                    sx={{ color: '#a3b3ff', textTransform: 'none', fontWeight: 600 }}>
                    {loading ? <CircularProgress size={14} sx={{ color: '#a3b3ff', mr: 1 }} /> : null}
                    {t('report.regenerate')}
                </Button>
            </Box>
            <Typography variant="caption" sx={{ display: 'block', opacity: 0.75, mb: 1.5 }}>
                {t('report.subtitle')}
            </Typography>

            {/* Basic / Advanced mode toggle */}
            <Box sx={{
                display: 'inline-flex', gap: 0.5, p: 0.5, mb: 2,
                borderRadius: 2,
                background: 'rgba(255,255,255,0.05)',
                border: '1px solid rgba(255,255,255,0.07)',
            }}>
                {['basic', 'advanced'].map(m => {
                    const active = mode === m;
                    return (
                        <Button
                            key={m}
                            size="small"
                            disableRipple
                            onClick={() => setMode(m)}
                            sx={{
                                px: 2, py: 0.4, minWidth: 0, textTransform: 'none',
                                fontWeight: 700, letterSpacing: 0.3, fontSize: 12,
                                borderRadius: 1.5,
                                color: active ? '#fff' : 'rgba(255,255,255,0.65)',
                                background: active
                                    ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
                                    : 'transparent',
                                boxShadow: active ? '0 2px 8px rgba(102,126,234,0.35)' : 'none',
                                '&:hover': {
                                    background: active
                                        ? 'linear-gradient(135deg, #5568d3 0%, #693f93 100%)'
                                        : 'rgba(255,255,255,0.07)',
                                },
                            }}
                        >
                            {t(`report.mode.${m}`)}
                        </Button>
                    );
                })}
            </Box>
            <Typography variant="caption" sx={{ display: 'block', opacity: 0.6, mb: 1.5, fontStyle: 'italic' }}>
                {t(`report.mode.${mode}.hint`)}
            </Typography>

            {loading && !report && (
                <Box sx={{
                    p: 3, borderRadius: 2, background: 'rgba(255,255,255,0.04)',
                    border: '1px solid rgba(255,255,255,0.06)',
                    display: 'flex', alignItems: 'center', gap: 2
                }}>
                    <CircularProgress size={18} sx={{ color: '#667eea' }} />
                    <Typography variant="body2" sx={{ opacity: 0.85 }}>
                        {t('report.loading')}
                    </Typography>
                </Box>
            )}

            {error && (
                <Box sx={{
                    p: 2, borderRadius: 2,
                    background: 'rgba(245,87,108,0.10)',
                    border: '1px solid rgba(245,87,108,0.45)',
                }}>
                    <Typography variant="body2" sx={{ fontWeight: 700, color: '#f5576c', mb: 0.5 }}>
                        {t('report.error.title')}
                    </Typography>
                    <Typography variant="caption" sx={{ opacity: 0.85 }}>{error}</Typography>
                </Box>
            )}

            {/* Unified banner — type · urgency · immediate action.
                Visible in both modes (Paciente / Médico). Color is driven by
                the deterministic urgency level (urgent / soon / routine / reassure). */}
            {r?.malignancy?.urgency && (
                <Box sx={{
                    mb: 2, p: 1.6, borderRadius: 2,
                    background: `linear-gradient(90deg, ${r.malignancy.urgency.color}22 0%, ${r.malignancy.urgency.color}08 100%)`,
                    border: `1px solid ${r.malignancy.urgency.color}55`,
                    display: 'flex', alignItems: 'center', gap: 1.5, flexWrap: 'wrap',
                }}>
                    <Box sx={{
                        px: 1.2, py: 0.5, borderRadius: 1.2,
                        background: r.malignancy.urgency.color, color: '#fff',
                        fontSize: 12, fontWeight: 800, letterSpacing: 0.5,
                        textTransform: 'uppercase',
                    }}>{r.malignancy.urgency.label}</Box>
                    <Typography sx={{ fontSize: 13, fontWeight: 700, color: '#fff' }}>
                        {r.prediction?.class}
                    </Typography>
                    <Typography sx={{ fontSize: 12.5, opacity: 0.85, flex: 1, minWidth: 200 }}>
                        {r.malignancy.urgency.action}
                    </Typography>
                </Box>
            )}

            {/* "View in Brain Atlas" chip — only renders when we have a
                resolvable region and the parent wired up the navigation. */}
            {(() => {
                const regionLabel = r?.malignancy?.region?.label;
                const regionSide  = r?.malignancy?.region?.side;
                const atlasId = regionLabelToAtlasId(regionLabel, regionSide, r?.prediction?.class);
                if (!atlasId || !onGoToAtlas) return null;
                return (
                    <Box
                        onClick={() => onGoToAtlas(atlasId)}
                        sx={{
                            mb: 2, p: 1.4, borderRadius: 2, cursor: 'pointer',
                            display: 'flex', alignItems: 'center', gap: 1.2,
                            background: 'linear-gradient(90deg, rgba(167,139,250,0.10) 0%, rgba(34,211,238,0.08) 100%)',
                            border: '1px solid rgba(167,139,250,0.35)',
                            transition: 'all .18s',
                            '&:hover': {
                                background: 'linear-gradient(90deg, rgba(167,139,250,0.18) 0%, rgba(34,211,238,0.14) 100%)',
                                borderColor: 'rgba(167,139,250,0.7)',
                                transform: 'translateY(-1px)',
                                boxShadow: '0 6px 18px rgba(167,139,250,0.25)',
                            },
                        }}
                    >
                        <Box sx={{
                            width: 32, height: 32, borderRadius: '50%',
                            background: 'rgba(167,139,250,0.18)',
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            fontSize: 17,
                        }}>🧠</Box>
                        <Box sx={{ flex: 1, minWidth: 0 }}>
                            <Typography sx={{
                                fontSize: 11, fontWeight: 700, letterSpacing: 0.5,
                                color: '#c4b5fd', textTransform: 'uppercase',
                            }}>
                                {t('report.view_in_atlas')}
                            </Typography>
                            <Typography sx={{ fontSize: 13, fontWeight: 600, color: '#fff', lineHeight: 1.3 }} noWrap>
                                {regionLabel}
                            </Typography>
                        </Box>
                        <Typography sx={{ fontSize: 18, color: '#a78bfa', mr: 0.5 }}>›</Typography>
                    </Box>
                );
            })()}

            {parsed && parsed.sections.length > 0 && (
                <Box>
                    {parsed.sections.map((s, i) => (
                        <ReportSectionBlock key={i} heading={s.heading} body={s.body} />
                    ))}

                    {/* ICD-10 / SNOMED codes panel intentionally removed — too
                        technical for the on-screen report. The data still ships in
                        r.malignancy.medical_codes for any future PDF / API consumer. */}
                    {parsed.trailer && (
                        <Box sx={{
                            mt: 1.5, p: 1.5, borderRadius: 2,
                            background: 'rgba(245,166,35,0.06)',
                            border: '1px dashed rgba(245,166,35,0.4)',
                        }}>
                            <Typography variant="caption" sx={{
                                display: 'block', color: '#f5a623',
                                fontStyle: 'italic', lineHeight: 1.5,
                            }}>
                                {parsed.trailer}
                            </Typography>
                        </Box>
                    )}
                </Box>
            )}

            {/* Fallback: legacy single-paragraph format (mode not understood) */}
            {report && parsed && parsed.sections.length === 0 && report.text && (
                <Box sx={{
                    p: 3, borderRadius: 2,
                    background: 'rgba(255,255,255,0.04)',
                    border: '1px solid rgba(255,255,255,0.06)',
                    fontFamily: 'Georgia, "Times New Roman", serif',
                    lineHeight: 1.7, fontSize: 15,
                }}>
                    {report.text}
                </Box>
            )}

            <Typography variant="caption" sx={{
                display: 'block', mt: 2, opacity: 0.55, fontStyle: 'italic', fontSize: 11,
            }}>
                {t('report.disclaimer')}
                {report?.model && <> &middot; <code style={{ opacity: 0.7 }}>{report.model}</code></>}
                {typeof report?.total_duration_ms === 'number' &&
                    <> &middot; {(report.total_duration_ms / 1000).toFixed(1)}s</>}
            </Typography>
        </Paper>
    );
}

const API_URL = process.env.REACT_APP_API_URL || '';


/* ── Sequence-type warning banner ───────────────────────────────── */
function SequenceWarningBanner({ r }) {
    const sw = r?.preprocessing?.sequence_warning;
    if (!sw || !sw.is_non_t1) return null;
    return (
        <Paper elevation={0} sx={{
            p: 2.5, mb: 3,
            background: 'rgba(138,43,226,0.12)',
            border: '1px solid rgba(138,43,226,0.50)',
            borderRadius: 2,
        }}>
            <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'flex-start', flexWrap: 'wrap' }}>
                <Chip label="NON-T1 SEQUENCE DETECTED" size="small" sx={{
                    bgcolor: 'rgba(138,43,226,0.25)', color: '#c084fc',
                    fontWeight: 700, letterSpacing: 0.8, flexShrink: 0,
                }} />
                <Typography variant="body2" sx={{ flex: 1, minWidth: 280, opacity: 0.92 }}>
                    This scan appears to be <strong>T2-weighted, FLAIR, or DWI</strong> (tissue-contrast CV = {sw.cv}, threshold = 0.28).
                    This model was trained exclusively on <strong>T1 sequences</strong> — predictions on T2/FLAIR/DWI scans
                    are <strong>unreliable regardless of confidence score</strong>. Please use a T1 sequence for accurate classification.
                </Typography>
            </Box>
        </Paper>
    );
}


/* ── Tumor-side OOD banner (shown when tumor model is operating outside its training distribution) ─ */
function TumorOODBanner({ r }) {
    if (!r || !r.ood || typeof r.ood.energy_score === 'undefined') return null;
    const ood = r.ood;
    const u = r.uncertainty || {};
    const isOOD = !!ood.is_ood;
    const needsReview = !!u.needs_review;
    if (!isOOD && !needsReview) return null;

    const color = isOOD ? '#f5576c' : '#f5a623';
    const label = isOOD ? 'OUT OF DISTRIBUTION' : 'EXPERT REVIEW NEEDED';
    const headline = isOOD
        ? `Scan is outside the tumor model's training distribution. The "${r.prediction.class}" prediction below — including any "No Tumor" verdict — may be unreliable.`
        : 'One or more confidence/uncertainty thresholds were exceeded. Treat the prediction as tentative.';

    return (
        <Paper elevation={0} sx={{
            p: 2.5, mb: 3, mt: 2,
            background: isOOD ? 'rgba(245,87,108,0.10)' : 'rgba(245,166,35,0.10)',
            border: `1px solid ${isOOD ? 'rgba(245,87,108,0.45)' : 'rgba(245,166,35,0.45)'}`,
            borderRadius: 2,
        }}>
            <Box sx={{ display: 'flex', gap: 1.5, alignItems: 'flex-start', flexWrap: 'wrap' }}>
                <Chip label={label} size="small" sx={{
                    bgcolor: `${color}25`, color, fontWeight: 700, letterSpacing: 0.8
                }} />
                <Typography variant="body2" sx={{ flex: 1, minWidth: 280, opacity: 0.92 }}>
                    {headline}
                    <Box component="span" sx={{ display: 'block', mt: 0.5, fontSize: 12, opacity: 0.7 }}>
                        Energy E(x) = {ood.energy_score} &middot; val p95 threshold = {ood.threshold_p95} ({ood.best_model}) &middot;
                        confidence = {(u.mean_confidence * 100).toFixed(1)}% &middot;
                        epistemic = {u.epistemic} &middot;
                        top-2 gap = {u.top2_gap}
                    </Box>
                </Typography>
            </Box>
        </Paper>
    );
}


/* ── Focus-Crop Self-Check ──────────────────────────────── */
function FocusCropCard({ fc }) {
    if (!fc || !fc.enabled) return null;
    const a = fc.agreement || {};
    const verdict = a.verdict || 'unknown';
    const meta = {
        consistent:         { color: '#38ef7d', label: 'CONSISTENT',         msg: 'Cropped re-classification agrees with full-image prediction.' },
        confidence_dropped: { color: '#f5a623', label: 'CONFIDENCE DROPPED', msg: 'Same class predicted, but confidence dropped sharply on the crop. Model may be using non-tumor features.' },
        class_changed:      { color: '#f5576c', label: 'CLASS CHANGED',      msg: 'Cropped image classifies as a different class. Model was likely relying on non-tumor (background / scanner) features.' },
        unknown:            { color: '#94a3b8', label: 'UNKNOWN',            msg: '' },
    }[verdict] || { color: '#94a3b8', label: 'UNKNOWN', msg: '' };

    return (
        <Card className="xai-card" sx={{ mt: 3 }}>
            <CardContent sx={{ p: 3 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1, flexWrap: 'wrap', gap: 1 }}>
                    <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
                        Focus-Crop Self-Check
                    </Typography>
                    <Chip label={meta.label} size="small" sx={{
                        bgcolor: `${meta.color}22`, color: meta.color,
                        border: `1px solid ${meta.color}55`,
                        fontWeight: 700, letterSpacing: 0.5
                    }} />
                </Box>
                <Typography variant="caption" sx={{ display: 'block', opacity: 0.75, mb: 2 }}>
                    The classifier was re-run on its own Grad-CAM++ attention region
                    (a {fc.crop_size?.[0]}×{fc.crop_size?.[1]} crop). If both passes agree, the model
                    is genuinely focused on the tumor; disagreement signals reliance on non-tumor features.
                </Typography>

                <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 2, alignItems: 'stretch' }}>
                    <Box sx={{ background: 'rgba(255,255,255,0.04)', borderRadius: 2, p: 1.5, border: '1px solid rgba(255,255,255,0.06)' }}>
                        <Typography variant="caption" sx={{ letterSpacing: 0.8, opacity: 0.65, fontSize: 10, textTransform: 'uppercase' }}>
                            Pass 1 — Full image
                        </Typography>
                        <Typography sx={{ fontWeight: 700, fontSize: 15, mt: 0.5, color: '#fff' }}>
                            {fc.full_prediction.class}
                        </Typography>
                        <Typography variant="body2" sx={{ opacity: 0.85 }}>
                            {(fc.full_prediction.confidence * 100).toFixed(1)}%
                        </Typography>
                    </Box>
                    <Box sx={{ background: 'rgba(255,255,255,0.04)', borderRadius: 2, p: 1.5, border: `1px solid ${meta.color}33` }}>
                        <Typography variant="caption" sx={{ letterSpacing: 0.8, opacity: 0.65, fontSize: 10, textTransform: 'uppercase' }}>
                            Pass 2 — Tumor crop
                        </Typography>
                        <Typography sx={{ fontWeight: 700, fontSize: 15, mt: 0.5, color: '#fff' }}>
                            {fc.crop_prediction.class}
                        </Typography>
                        <Typography variant="body2" sx={{ opacity: 0.85 }}>
                            {(fc.crop_prediction.confidence * 100).toFixed(1)}%
                            <Box component="span" sx={{
                                ml: 1, fontSize: 11,
                                color: a.confidence_delta >= 0 ? '#38ef7d' : (a.confidence_delta < -0.15 ? '#f5576c' : '#f5a623')
                            }}>
                                ({a.confidence_delta >= 0 ? '+' : ''}{(a.confidence_delta * 100).toFixed(1)}pp vs full)
                            </Box>
                        </Typography>
                    </Box>
                </Box>

                {fc.crop_image && (
                    <Box sx={{ mt: 2, textAlign: 'center' }}>
                        <Typography variant="caption" sx={{ display: 'block', opacity: 0.65, mb: 0.5 }}>
                            Crop fed to Pass 2 (bbox: {fc.bbox_orig.w}×{fc.bbox_orig.h} px @ ({fc.bbox_orig.x}, {fc.bbox_orig.y}))
                        </Typography>
                        <Box sx={{
                            display: 'inline-block', borderRadius: 2, overflow: 'hidden',
                            border: `1px solid ${meta.color}55`,
                        }}>
                            <CardMedia component="img"
                                image={`data:image/png;base64,${fc.crop_image}`}
                                alt="Tumor focus crop"
                                sx={{ maxHeight: 180, width: 'auto', objectFit: 'contain', display: 'block' }} />
                        </Box>
                    </Box>
                )}

                {meta.msg && (
                    <Box sx={{
                        mt: 2, p: 1.5, borderRadius: 2,
                        background: 'rgba(255,255,255,0.04)',
                        borderLeft: `3px solid ${meta.color}`,
                    }}>
                        <Typography variant="body2" sx={{ opacity: 0.9 }}>
                            <strong style={{ color: meta.color }}>Verdict:</strong> {meta.msg}
                        </Typography>
                    </Box>
                )}
            </CardContent>
        </Card>
    );
}


/* ── Malignancy Card ─────────────────────────────────── */
function MalignancyStatBox({ label, value, sub }) {
    return (
        <Box sx={{
            background: 'rgba(255,255,255,0.04)',
            border: '1px solid rgba(255,255,255,0.06)',
            borderRadius: 1.5,
            px: 1.5, py: 1
        }}>
            <Typography variant="caption" sx={{ letterSpacing: 0.5, textTransform: 'uppercase', opacity: 0.7, fontSize: 10 }}>
                {label}
            </Typography>
            <Typography sx={{ fontSize: 16, fontWeight: 600, color: '#fff', mt: 0.25 }}>
                {value}
            </Typography>
            {sub && (
                <Typography variant="caption" sx={{ opacity: 0.55, fontSize: 11 }}>
                    {sub}
                </Typography>
            )}
        </Box>
    );
}

// Dialog for drawing a manual tumor bounding box on the input image.
// Opens fullscreen-ish, shows the original image, lets the user click+drag a
// rectangle, then POSTs the crop to /api/predict_manual_bbox.
function ManualBboxDialog({ open, onClose, imageFile, previewUrl, onResult }) {
    const { t, lang } = useT();
    const imgRef = useRef(null);
    const containerRef = useRef(null);
    const [box, setBox] = useState(null);          // {x0, y0, x1, y1} in DISPLAYED px
    const [drag, setDrag] = useState(null);        // {x0, y0} during drag
    const [natural, setNatural] = useState(null);  // {w, h} of the original image
    const [submitting, setSubmitting] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (!open) { setBox(null); setDrag(null); setError(null); }
    }, [open]);

    const onImgLoad = (e) => {
        setNatural({ w: e.target.naturalWidth, h: e.target.naturalHeight });
    };

    const getXY = (evt) => {
        const rect = imgRef.current.getBoundingClientRect();
        const x = Math.max(0, Math.min(rect.width, evt.clientX - rect.left));
        const y = Math.max(0, Math.min(rect.height, evt.clientY - rect.top));
        return { x, y };
    };

    const onDown = (e) => {
        e.preventDefault();
        const { x, y } = getXY(e);
        setDrag({ x0: x, y0: y });
        setBox({ x0: x, y0: y, x1: x, y1: y });
    };
    const onMove = (e) => {
        if (!drag) return;
        const { x, y } = getXY(e);
        setBox({ x0: drag.x0, y0: drag.y0, x1: x, y1: y });
    };
    const onUp = () => setDrag(null);

    const norm = box ? {
        x: Math.min(box.x0, box.x1),
        y: Math.min(box.y0, box.y1),
        w: Math.abs(box.x1 - box.x0),
        h: Math.abs(box.y1 - box.y0),
    } : null;

    const canSubmit = norm && norm.w > 10 && norm.h > 10 && natural && imageFile;

    const submit = async () => {
        if (!canSubmit) return;
        setSubmitting(true); setError(null);
        try {
            const rect = imgRef.current.getBoundingClientRect();
            const sx = natural.w / rect.width;
            const sy = natural.h / rect.height;
            const bbox = {
                x: Math.round(norm.x * sx),
                y: Math.round(norm.y * sy),
                w: Math.round(norm.w * sx),
                h: Math.round(norm.h * sy),
            };
            const fd = new FormData();
            fd.append('image', imageFile);
            fd.append('bbox', JSON.stringify(bbox));
            fd.append('language', lang);
            const resp = await axios.post(`${API_URL}/api/predict_manual_bbox`, fd, { timeout: 180000 });
            onResult(resp.data);
            onClose();
        } catch (e) {
            setError(e.response?.data?.error || 'Manual analysis failed. Please try again.');
        } finally {
            setSubmitting(false);
        }
    };

    return (
        <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
            <DialogTitle sx={{ fontWeight: 700 }}>
                {t('manual.title')}
                <Typography variant="caption" sx={{ display: 'block', opacity: 0.7, fontWeight: 400 }}>
                    {t('manual.subtitle')}
                </Typography>
            </DialogTitle>
            <DialogContent>
                {previewUrl ? (
                    <Box ref={containerRef} sx={{
                        position: 'relative', userSelect: 'none', display: 'inline-block', maxWidth: '100%',
                    }}>
                        <img
                            ref={imgRef}
                            src={previewUrl}
                            alt="MRI to annotate"
                            onLoad={onImgLoad}
                            onMouseDown={onDown}
                            onMouseMove={onMove}
                            onMouseUp={onUp}
                            onMouseLeave={onUp}
                            draggable={false}
                            style={{ display: 'block', maxWidth: '100%', maxHeight: '70vh', cursor: 'crosshair' }}
                        />
                        {norm && (
                            <Box sx={{
                                position: 'absolute',
                                left: norm.x, top: norm.y, width: norm.w, height: norm.h,
                                border: '2px solid #38ef7d',
                                background: 'rgba(56, 239, 125, 0.10)',
                                pointerEvents: 'none',
                            }} />
                        )}
                    </Box>
                ) : (
                    <Typography sx={{ opacity: 0.7 }}>{t('manual.no_image')}</Typography>
                )}
                {error && <Alert severity="error" sx={{ mt: 2 }}>{error}</Alert>}
            </DialogContent>
            <DialogActions sx={{ px: 3, pb: 2 }}>
                <Button onClick={() => setBox(null)} disabled={!box || submitting}>{t('manual.clear')}</Button>
                <Button onClick={onClose} disabled={submitting}>{t('manual.cancel')}</Button>
                <Button onClick={submit} disabled={!canSubmit || submitting} variant="contained"
                    sx={{ background: 'linear-gradient(135deg,#667eea,#764ba2)' }}>
                    {submitting ? t('manual.submitting') : t('manual.submit')}
                </Button>
            </DialogActions>
        </Dialog>
    );
}


function MalignancyCard({ data, predictedClass, cropImage, cropBbox, imageFile, previewUrl, onManualResult }) {
    const { t } = useT();
    const [manualOpen, setManualOpen] = useState(false);
    const [symptomAdjust, setSymptomAdjust] = useState(null);
    if (!data) return null;
    const isNA = data.base_risk === 'N/A';
    const baseScore = Number(data.score) || 0;
    const score = symptomAdjust?.adjusted ?? baseScore;
    const riskColor = { HIGH: '#f5576c', 'LOW-MEDIUM': '#f5a623', LOW: '#38ef7d' }[data.base_risk] || '#94a3b8';
    const sizeLabel = (data.size_category || '').replace(/^./, c => c.toUpperCase());
    const mg = data.medgemma_assessment || {};
    const locationText = data.region?.label || mg.tumor_location || '—';
    const sizeFromMedGemma = data.size_source === 'medgemma';
    // 5%-wide bucket helper, mirrors backend size_pct_to_range
    const pctToBucket = (s) => {
        if (s == null) return null;
        const n = Number(s);
        if (!Number.isFinite(n) || n <= 0) return null;
        if (n < 1) return '<1%';
        if (n < 5) return '1-5%';
        if (n >= 30) return '30%+';
        const lo = Math.floor(n / 5) * 5;
        return `${lo}-${lo + 5}%`;
    };
    const mgSizeRange = pctToBucket(mg.estimated_size_pct);

    return (
        <Card className="xai-card" sx={{
            mt: 3,
            position: 'relative',
            overflow: 'hidden',
            border: '1px solid rgba(167,139,250,0.25)',
            background: 'linear-gradient(135deg, rgba(34,211,238,0.04) 0%, rgba(167,139,250,0.06) 50%, rgba(244,114,182,0.04) 100%)',
            boxShadow: '0 8px 32px rgba(102,126,234,0.18)',
            '&::before': {
                content: '""', position: 'absolute', top: 0, left: 0, right: 0, height: 2,
                background: 'linear-gradient(90deg, #22d3ee, #a78bfa, #f472b6)',
                opacity: 0.7,
            },
        }}>
            <CardContent sx={{ p: 3 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2, flexWrap: 'wrap', gap: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.2 }}>
                        <Box sx={{
                            width: 6, height: 22, borderRadius: 1,
                            background: 'linear-gradient(180deg, #22d3ee, #a78bfa)',
                            boxShadow: '0 0 10px rgba(34,211,238,0.6)',
                        }} />
                        <Typography sx={{
                            fontWeight: 800, fontSize: 16, letterSpacing: 0.6,
                            background: 'linear-gradient(90deg, #67e8f9, #c4b5fd)',
                            WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
                            textTransform: 'uppercase',
                        }}>
                            {t('malig.title')}
                        </Typography>
                    </Box>
                    {predictedClass && (
                        <Box sx={{
                            px: 1.4, py: 0.5, borderRadius: 1.5,
                            background: 'rgba(167,139,250,0.15)',
                            border: '1px solid rgba(167,139,250,0.4)',
                            fontSize: 11, letterSpacing: 0.5,
                        }}>
                            <span style={{ opacity: 0.75 }}>{t('malig.predicted')}: </span>
                            <strong style={{ color: '#e9d5ff' }}>{predictedClass}</strong>
                        </Box>
                    )}
                </Box>

                {isNA ? (
                    <Box sx={{ p: 2.5, borderRadius: 2, background: 'rgba(255,255,255,0.06)' }}>
                        <Typography variant="body1" sx={{ fontWeight: 600, mb: 0.5 }}>{t('malig.no_tumor.title')}</Typography>
                        <Typography variant="body2" sx={{ opacity: 0.8 }}>
                            {t('malig.no_tumor.body')}
                        </Typography>
                    </Box>
                ) : (
                    <>
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2.5, alignItems: 'center', mb: 2 }}>
                            <Box sx={{
                                textAlign: 'center', px: 3, py: 2, minWidth: 130,
                                borderRadius: 2, background: 'linear-gradient(135deg,#667eea 0%,#764ba2 100%)',
                                boxShadow: '0 4px 16px rgba(102,126,234,0.3)'
                            }}>
                                <Typography variant="caption" sx={{ letterSpacing: 1.5, opacity: 0.85, color: '#fff' }}>{t('malig.score')}</Typography>
                                <Typography sx={{ fontSize: 44, fontWeight: 700, lineHeight: 1, color: '#fff', textShadow: '0 2px 8px rgba(0,0,0,0.3)' }}>
                                    {score.toFixed(1)}
                                </Typography>
                                <Typography variant="caption" sx={{ opacity: 0.85, color: '#fff' }}>{t('malig.outof')}</Typography>
                                {symptomAdjust && symptomAdjust.bonus > 0 && (
                                    <Typography sx={{
                                        mt: 0.5, fontSize: 10, fontWeight: 700,
                                        color: '#fde68a', letterSpacing: 0.4,
                                    }}>
                                        +{symptomAdjust.bonus} ({symptomAdjust.count} {t('symptoms.short')})
                                    </Typography>
                                )}
                            </Box>
                            <Box sx={{ flex: 1, minWidth: 240, display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(150px,1fr))', gap: 1 }}>
                                <MalignancyStatBox
                                    label={t('malig.risk')}
                                    value={<span style={{ color: riskColor }}>{data.base_risk}</span>}
                                    sub={`${data.base_score} / 10 ${t('malig.baseline')}`}
                                />
                                <MalignancyStatBox
                                    label={t('malig.size') + ' (pipeline)'}
                                    value={data.size_range
                                        || (typeof data.size_pct === 'number' ? `~${data.size_pct}%` : '—')}
                                    sub={'YOLO + SAM' + (sizeLabel ? ` · ${sizeLabel}` : '')}
                                />
                                <MalignancyStatBox
                                    label={t('malig.size') + ' (MedGemma)'}
                                    value={mgSizeRange
                                        || (mg.estimated_size_pct != null ? `~${mg.estimated_size_pct}%` : '—')}
                                    sub={mg.estimated_size_pct != null
                                        ? `LLM on original · ${mg.size_category || ''}`.trim()
                                        : 'no estimate'}
                                />
                                <MalignancyStatBox
                                    label={t('malig.location')}
                                    value={locationText}
                                    sub={mg.boundary_desc
                                        ? `${mg.boundary_desc.length > 60 ? mg.boundary_desc.slice(0, 57) + '…' : mg.boundary_desc}`
                                        : undefined}
                                />
                            </Box>
                        </Box>

                        {/* Tumor region crop with green bbox drawn server-side — centered + prominent */}
                        {cropImage && (
                            <Box sx={{ mt: 2.5 }}>
                                <Typography variant="caption" sx={{ display: 'block', opacity: 0.7, mb: 1, letterSpacing: 0.5, textTransform: 'uppercase', fontSize: 10, textAlign: 'center' }}>
                                    {t('malig.crop_label')}
                                    {cropBbox && ` · ${cropBbox.w}×${cropBbox.h} px`}
                                    {data.bbox_source === 'manual' && ` · ${t('malig.manual_region')}`}
                                </Typography>
                                <Box sx={{
                                    display: 'flex', justifyContent: 'center',
                                }}>
                                    <Box sx={{
                                        borderRadius: 2, overflow: 'hidden',
                                        border: `1px solid ${data.bbox_source === 'manual' ? 'rgba(56,239,125,0.55)' : 'rgba(102,126,234,0.4)'}`,
                                        boxShadow: '0 4px 16px rgba(0,0,0,0.35)',
                                        maxWidth: '100%',
                                    }}>
                                        <CardMedia
                                            component="img"
                                            image={`data:image/png;base64,${cropImage}`}
                                            alt="Tumor region with bounding box"
                                            sx={{ maxHeight: 360, width: 'auto', display: 'block', objectFit: 'contain' }}
                                        />
                                    </Box>
                                </Box>
                            </Box>
                        )}

                        {/* Manual-bbox controls — only show if we have the source image to draw on */}
                        {(imageFile && previewUrl) && (
                            <Box sx={{ mt: 1.5, display: 'flex', justifyContent: 'center', gap: 1, flexWrap: 'wrap' }}>
                                <Button
                                    size="small"
                                    variant="outlined"
                                    onClick={() => setManualOpen(true)}
                                    sx={{
                                        textTransform: 'none', borderRadius: 2,
                                        borderColor: 'rgba(56,239,125,0.55)', color: '#38ef7d',
                                        '&:hover': { borderColor: '#38ef7d', background: 'rgba(56,239,125,0.10)' },
                                    }}
                                >
                                    {t('malig.draw_btn')}
                                </Button>
                                {data.bbox_source === 'manual' && (
                                    <Button
                                        size="small"
                                        onClick={() => onManualResult && onManualResult(null)}
                                        sx={{ textTransform: 'none', color: 'rgba(255,255,255,0.7)' }}
                                    >
                                        {t('malig.revert_btn')}
                                    </Button>
                                )}
                            </Box>
                        )}
                        <ManualBboxDialog
                            open={manualOpen}
                            onClose={() => setManualOpen(false)}
                            imageFile={imageFile}
                            previewUrl={previewUrl}
                            onResult={(res) => { if (onManualResult) onManualResult(res); }}
                        />

                        {/* Symptom selector — patient checks symptoms to live-adjust the score */}
                        <SymptomSelector
                            predictedClass={predictedClass}
                            baseScore={baseScore}
                            onAdjustedScore={setSymptomAdjust}
                        />

                        {/* Clinical Context (hybrid static + MedGemma) */}
                        <ClinicalContextPanel data={data} predictedClass={predictedClass} />


                        {false && data.medgemma_assessment && (
                            <Box sx={{ display: 'none' }}>
                                {/* Removed: dual-pane Grad-CAM++ vs MedGemma comparison.
                                    MedGemma's location/size are now the primary values shown
                                    in the stat boxes above. Keep this block dead for now to
                                    minimise diff; we'll drop the dead code on the next pass. */}
                                {(() => {
                                    const mg = data.medgemma_assessment || {};
                                    const cv = mg.cross_validation || {};
                                    const overallColor = {
                                        consistent: '#38ef7d', partial: '#f5a623', inconsistent: '#f5576c'
                                    }[cv.overall] || '#94a3b8';
                                    const overallIcon = {
                                        consistent: '\u2713', partial: '\u25B3', inconsistent: '\u2717'
                                    }[cv.overall] || '?';
                                    return (
                                        <>
                                            {cv.overall && cv.overall !== 'unknown' && (
                                                <Box sx={{
                                                    p: 1.5, mb: 1.5, borderRadius: 1.5,
                                                    background: `${overallColor}18`,
                                                    border: `1px solid ${overallColor}40`,
                                                    display: 'flex', alignItems: 'center', gap: 1.5,
                                                }}>
                                                    <Typography sx={{ fontSize: 22, lineHeight: 1, color: overallColor }}>{overallIcon}</Typography>
                                                    <Box>
                                                        <Typography variant="body2" sx={{ fontWeight: 700, color: overallColor, textTransform: 'uppercase', fontSize: 11, letterSpacing: 1 }}>
                                                            {cv.overall === 'consistent' ? 'Agreement' : cv.overall === 'partial' ? 'Partial Agreement' : 'Disagreement'}
                                                        </Typography>
                                                        <Typography variant="caption" sx={{ opacity: 0.8, fontSize: 10 }}>
                                                            {cv.verdict || 'Cross-validation between Grad-CAM++ and MedGemma'}
                                                        </Typography>
                                                    </Box>
                                                </Box>
                                            )}
                                            <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1 }}>
                                                <Box sx={{ p: 1.5, borderRadius: 1, background: 'rgba(102,126,234,0.1)', border: '1px solid rgba(102,126,234,0.2)' }}>
                                                    <Typography variant="caption" sx={{ fontWeight: 700, color: '#667eea', display: 'block', mb: 1, fontSize: 10, letterSpacing: 1, textTransform: 'uppercase' }}>
                                                        Grad-CAM++ (pixel analysis)
                                                    </Typography>
                                                    {data.region?.label && (
                                                        <Typography variant="body2" sx={{ fontSize: 12, mb: 0.5 }}>
                                                            <strong>Location:</strong> {data.region.label}
                                                        </Typography>
                                                    )}
                                                    {(data.size_range || typeof data.size_pct === 'number') && (
                                                        <Typography variant="body2" sx={{ fontSize: 12, mb: 0.5 }}>
                                                            <strong>Size:</strong> {data.size_range || `~${data.size_pct}%`}
                                                        </Typography>
                                                    )}
                                                    {cv.cam_quadrant && (
                                                        <Typography variant="caption" sx={{ opacity: 0.6, fontSize: 10 }}>
                                                            Quadrant: {cv.cam_quadrant}
                                                        </Typography>
                                                    )}
                                                </Box>
                                                <Box sx={{ p: 1.5, borderRadius: 1, background: 'rgba(245,166,35,0.08)', border: '1px solid rgba(245,166,35,0.2)' }}>
                                                    <Typography variant="caption" sx={{ fontWeight: 700, color: '#f5a623', display: 'block', mb: 1, fontSize: 10, letterSpacing: 1, textTransform: 'uppercase' }}>
                                                        MedGemma (vision AI)
                                                    </Typography>
                                                    {mg.tumor_location && (
                                                        <Typography variant="body2" sx={{ fontSize: 12, mb: 0.5 }}>
                                                            <strong>Location:</strong> {mg.tumor_location}
                                                        </Typography>
                                                    )}
                                                    {mg.estimated_size_pct != null && (
                                                        <Typography variant="body2" sx={{ fontSize: 12, mb: 0.5 }}>
                                                            <strong>Size:</strong> ~{mg.estimated_size_pct}%
                                                        </Typography>
                                                    )}
                                                    {mg.bbox_quadrant && (
                                                        <Typography variant="caption" sx={{ opacity: 0.6, fontSize: 10 }}>
                                                            Quadrant: {mg.bbox_quadrant}
                                                        </Typography>
                                                    )}
                                                </Box>
                                            </Box>
                                            {mg.boundary_desc && (
                                                <Box sx={{ mt: 1, p: 1, borderRadius: 1, background: 'rgba(255,255,255,0.04)' }}>
                                                    <Typography variant="caption" sx={{ fontWeight: 600, opacity: 0.7 }}>Margins (MedGemma): </Typography>
                                                    <Typography variant="caption" sx={{ opacity: 0.85 }}>{mg.boundary_desc}</Typography>
                                                </Box>
                                            )}
                                            {mg.confidence_note && (
                                                <Typography variant="caption" sx={{ display: 'block', mt: 1, opacity: 0.65, fontStyle: 'italic', fontSize: 10 }}>
                                                    {mg.confidence_note}
                                                </Typography>
                                            )}
                                        </>
                                    );
                                })()}
                                {typeof data.medgemma_duration_ms === 'number' && (
                                    <Typography variant="caption" sx={{ display: 'block', mt: 1, opacity: 0.45, fontSize: 10 }}>
                                        MedGemma inference: {(data.medgemma_duration_ms / 1000).toFixed(1)}s
                                    </Typography>
                                )}
                            </Box>
                        )}

                        {/* Tight footer: who measured size + how long MedGemma took. */}
                        <Typography variant="caption" sx={{
                            display: 'block', mt: 1.5, opacity: 0.55, fontSize: 10, fontStyle: 'italic',
                        }}>
                            {sizeFromMedGemma ? t('malig.footer.medgemma') : t('malig.footer.sam')}
                            {typeof data.medgemma_duration_ms === 'number' &&
                                <> · MedGemma {(data.medgemma_duration_ms / 1000).toFixed(1)}s</>}
                        </Typography>
                    </>
                )}
            </CardContent>
        </Card>
    );
}

/* ── Anatomical region + clinical-prior consistency check ─── */
function RegionConsistency({ region, predictedClass }) {
    const consistency = region.consistency || 'unknown';
    const meta = {
        typical:  { color: '#38ef7d', label: 'TYPICAL',  icon: '✓', verdict: `Attention region matches a typical ${predictedClass} location.` },
        atypical: { color: '#f5a623', label: 'ATYPICAL', icon: '!', verdict: `Attention region is unusual for ${predictedClass} — consider review.` },
        unknown:  { color: '#94a3b8', label: 'UNCERTAIN', icon: '?', verdict: 'Insufficient prior info for this region.' },
    }[consistency] || { color: '#94a3b8', label: 'UNCERTAIN', icon: '?', verdict: '' };

    return (
        <Box sx={{
            p: 2, borderRadius: 1.5,
            background: 'rgba(255,255,255,0.04)',
            border: `1px solid ${meta.color}33`
        }}>
            <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1, flexWrap: 'wrap', gap: 1 }}>
                <Typography variant="body2" sx={{ fontWeight: 700, color: '#fff' }}>
                    Anatomical region · prior-consistency check
                </Typography>
                <Chip
                    label={`${meta.icon} ${meta.label}`}
                    size="small"
                    sx={{
                        bgcolor: `${meta.color}22`,
                        color: meta.color,
                        border: `1px solid ${meta.color}55`,
                        fontWeight: 700, letterSpacing: 0.5
                    }}
                />
            </Box>
            <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(140px,1fr))', gap: 1, mb: 1.5 }}>
                <MalignancyStatBox label="Region" value={region.label} sub={`${region.side} · ${region.axial_zone}`} />
                {region.centroid && (
                    <MalignancyStatBox label="Centroid" value={`(${region.centroid.x}, ${region.centroid.y})`} sub="bbox center, 224×224 frame" />
                )}
            </Box>
            <Typography variant="body2" sx={{ opacity: 0.85, mb: 0.5 }}>
                <strong style={{ color: meta.color }}>{meta.verdict}</strong>
            </Typography>
            {region.explanation && (
                <Typography variant="caption" sx={{ display: 'block', opacity: 0.7, lineHeight: 1.5 }}>
                    {region.explanation}
                </Typography>
            )}
            {region.disclaimer && (
                <Typography variant="caption" sx={{ display: 'block', opacity: 0.5, mt: 1, fontStyle: 'italic', fontSize: 10 }}>
                    {region.disclaimer}
                </Typography>
            )}
        </Box>
    );
}

/* ── Stat Pill ───────────────────────────────────────── */
function Stat({ label, value, color }) {
    return (
        <Box sx={{ textAlign: 'center', px: 1.5 }}>
            <Typography variant="h6" sx={{ fontWeight: 700, color: color || '#667eea' }}>{value}</Typography>
            <Typography variant="caption" sx={{ opacity: 0.8 }}>{label}</Typography>
        </Box>
    );
}

/* ── Dataset Badge Colors ────────────────────────────── */
const DS_COLORS = { kaggle_mri: '#667eea', brisc2025: '#38ef7d', hospital: '#f5a623' };
const CLASS_COLORS = { glioma: '#f5576c', meningioma: '#667eea', pituitary: '#f5a623', no_tumor: '#38ef7d' };

/* ── Sample Gallery Component ────────────────────────── */
function SampleGallery({ onSelectSample }) {
    const [datasets, setDatasets] = useState(null);
    const [activeDs, setActiveDs] = useState(0);
    const [activeClass, setActiveClass] = useState('glioma');
    const [previewImg, setPreviewImg] = useState(null);
    const [loadingSample, setLoadingSample] = useState(null);

    useEffect(() => {
        axios.get(`${API_URL}/api/samples`).then(res => {
            setDatasets(res.data);
        }).catch(() => { });
    }, []);

    const dsKeys = datasets ? Object.keys(datasets) : [];
    const currentDs = dsKeys[activeDs];
    const dsData = datasets?.[currentDs];
    const classData = dsData?.classes?.[activeClass];

    const handleUseSample = useCallback(async (imgUrl) => {
        setLoadingSample(imgUrl);
        try {
            const resp = await axios.get(`${API_URL}${imgUrl}`, { responseType: 'blob' });
            const filename = imgUrl.split('/').pop();
            const file = new File([resp.data], filename, { type: resp.data.type || 'image/jpeg' });
            onSelectSample(file);
        } catch (e) { /* ignore */ }
        setLoadingSample(null);
    }, [onSelectSample]);

    if (!datasets) return (
        <Box sx={{ textAlign: 'center', py: 4 }}>
            <CircularProgress size={30} sx={{ color: '#667eea' }} />
            <Typography variant="body2" sx={{ mt: 1, opacity: 0.7 }}>Loading sample data...</Typography>
        </Box>
    );

    return (
        <Box>
            {/* Dataset Tabs */}
            <Tabs value={activeDs} onChange={(_, v) => { setActiveDs(v); setActiveClass('glioma'); }}
                variant="fullWidth"
                sx={{
                    mb: 2, '& .MuiTab-root': { color: 'rgba(255,255,255,0.6)', fontWeight: 600, fontSize: '0.75rem', minHeight: 48 },
                    '& .Mui-selected': { color: '#fff !important' },
                    '& .MuiTabs-indicator': { background: DS_COLORS[currentDs] || '#667eea', height: 3, borderRadius: 2 }
                }}>
                {dsKeys.map((k, i) => (
                    <Tab key={k} label={
                        <Box sx={{ textAlign: 'center' }}>
                            <Typography variant="caption" sx={{ fontWeight: 700, display: 'block', lineHeight: 1.2 }}>
                                {datasets[k].name.length > 25 ? datasets[k].name.substring(0, 25) + '...' : datasets[k].name}
                            </Typography>
                            <Chip label={`${(datasets[k].total_images || 0).toLocaleString()} images`} size="small"
                                sx={{ mt: 0.3, height: 16, fontSize: '0.6rem', bgcolor: `${DS_COLORS[k]}33`, color: DS_COLORS[k], fontWeight: 600 }} />
                        </Box>
                    } />
                ))}
            </Tabs>

            {/* Dataset Info Bar */}
            {dsData && (
                <Box sx={{ p: 2, mb: 2, borderRadius: 2, background: 'rgba(255,255,255,0.06)', border: `1px solid ${DS_COLORS[currentDs]}33` }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 1 }}>
                        <Box>
                            <Typography variant="subtitle2" sx={{ fontWeight: 700, color: DS_COLORS[currentDs] }}>
                                {dsData.name}</Typography>
                            <Typography variant="caption" sx={{ opacity: 0.7, display: 'block', maxWidth: 500 }}>
                                {dsData.description}</Typography>
                        </Box>
                        <Button size="small" variant="outlined"
                            href={dsData.source} target="_blank" rel="noopener noreferrer"
                            startIcon={<OpenInNewIcon />}
                            sx={{
                                borderColor: DS_COLORS[currentDs], color: DS_COLORS[currentDs], fontWeight: 600, fontSize: '0.7rem',
                                '&:hover': { borderColor: DS_COLORS[currentDs], bgcolor: `${DS_COLORS[currentDs]}22` }
                            }}>
                            Source
                        </Button>
                    </Box>
                </Box>
            )}

            {/* Class Selector */}
            <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
                {['glioma', 'meningioma', 'pituitary', 'no_tumor'].map(cls => (
                    <Chip key={cls} label={cls === 'no_tumor' ? 'No Tumor' : cls.charAt(0).toUpperCase() + cls.slice(1)}
                        onClick={() => setActiveClass(cls)}
                        sx={{
                            fontWeight: 600, cursor: 'pointer', transition: 'all 0.2s',
                            background: activeClass === cls ? CLASS_COLORS[cls] : 'rgba(255,255,255,0.1)',
                            color: activeClass === cls ? (cls === 'no_tumor' || cls === 'pituitary' ? '#000' : '#fff') : 'rgba(255,255,255,0.7)',
                            border: `1px solid ${activeClass === cls ? CLASS_COLORS[cls] : 'rgba(255,255,255,0.2)'}`,
                            '&:hover': { background: `${CLASS_COLORS[cls]}88` }
                        }} />
                ))}
            </Box>

            {/* Class Info */}
            {classData?.info && (
                <Box sx={{ px: 2, py: 1, mb: 2, borderRadius: 1.5, background: `${CLASS_COLORS[activeClass]}15`, borderLeft: `3px solid ${CLASS_COLORS[activeClass]}` }}>
                    <Typography variant="body2" sx={{ fontWeight: 600, color: CLASS_COLORS[activeClass] }}>
                        {classData.info.label}</Typography>
                    <Typography variant="caption" sx={{ opacity: 0.8 }}>
                        {classData.info.description}</Typography>
                </Box>
            )}

            {/* Image Grid */}
            {classData?.images?.length > 0 ? (
                <Grid container spacing={1.5}>
                    {classData.images.map((imgUrl, idx) => (
                        <Grid item xs={6} sm={3} key={idx}>
                            <Card sx={{
                                background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.12)',
                                borderRadius: 2, overflow: 'hidden', transition: 'all 0.3s',
                                '&:hover': { transform: 'translateY(-4px)', boxShadow: `0 8px 30px ${CLASS_COLORS[activeClass]}33`, borderColor: CLASS_COLORS[activeClass] }
                            }}>
                                <CardMedia component="img" image={`${API_URL}${imgUrl}`} alt={`${activeClass} sample ${idx + 1}`}
                                    sx={{ height: 120, objectFit: 'cover', cursor: 'pointer' }}
                                    onClick={() => setPreviewImg(`${API_URL}${imgUrl}`)} />
                                <Box sx={{ p: 1, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <Typography variant="caption" sx={{ opacity: 0.7, fontSize: '0.6rem' }}>
                                        Sample {idx + 1}</Typography>
                                    <Box sx={{ display: 'flex', gap: 0.5 }}>
                                        <Tooltip title="Use this image for analysis">
                                            <IconButton size="small" onClick={() => handleUseSample(imgUrl)}
                                                disabled={loadingSample === imgUrl}
                                                sx={{ color: '#667eea', p: 0.5, '&:hover': { bgcolor: 'rgba(102,126,234,0.2)' } }}>
                                                {loadingSample === imgUrl ? <CircularProgress size={14} /> : <ScienceIcon sx={{ fontSize: 16 }} />}
                                            </IconButton>
                                        </Tooltip>
                                        <Tooltip title="Download image">
                                            <IconButton size="small" component="a" href={`${API_URL}${imgUrl}`} download
                                                sx={{ color: '#38ef7d', p: 0.5, '&:hover': { bgcolor: 'rgba(56,239,125,0.2)' } }}>
                                                <DownloadIcon sx={{ fontSize: 16 }} />
                                            </IconButton>
                                        </Tooltip>
                                    </Box>
                                </Box>
                            </Card>
                        </Grid>
                    ))}
                </Grid>
            ) : (
                <Typography variant="body2" sx={{ textAlign: 'center', py: 3, opacity: 0.5 }}>
                    No samples available for this class</Typography>
            )}

            {/* Image Preview Dialog */}
            <Dialog open={!!previewImg} onClose={() => setPreviewImg(null)} maxWidth="sm" fullWidth
                PaperProps={{ sx: { bgcolor: 'rgba(20,20,30,0.95)', backdropFilter: 'blur(20px)' } }}>
                <DialogContent sx={{ p: 1, textAlign: 'center' }}>
                    {previewImg && <img src={previewImg} alt="Preview" style={{ maxWidth: '100%', borderRadius: 8 }} />}
                </DialogContent>
            </Dialog>
        </Box>
    );
}

/* ── Floating MedGemma chat panel ─────────────────────────────────── */
function ChatbotPanel({ open, onClose, scanResult }) {
    const { t, lang } = useT();
    const [audience, setAudience] = useState('patient'); // 'patient' | 'doctor'
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [sending, setSending] = useState(false);
    const [error, setError] = useState(null);
    const listRef = useRef(null);

    // Auto-scroll to newest message
    useEffect(() => {
        const el = listRef.current;
        if (el) el.scrollTop = el.scrollHeight;
    }, [messages, sending]);

    // Derive scan context from the current prediction (if any)
    const scanContext = useMemo(() => {
        if (!scanResult) return null;
        const pred = scanResult.prediction || {};
        const mal  = scanResult.malignancy || {};
        const reg  = mal.region || {};
        const symptoms = (scanResult.user_symptoms || []).map(s => s.label || s);
        if ((pred.class || '') === 'No Tumor') {
            return { predicted_class: 'No Tumor', confidence: pred.confidence, symptoms };
        }
        return {
            predicted_class: pred.class,
            confidence:      pred.confidence,
            size_range:      mal.size_range,
            size_pct:        mal.size_pct,
            base_risk:       mal.base_risk,
            score:           mal.score,
            location:        reg.label,
            side:            reg.side,
            symptoms,
        };
    }, [scanResult]);

    const send = useCallback(async () => {
        const text = input.trim();
        if (!text || sending) return;
        const next = [...messages, { role: 'user', content: text }];
        setMessages(next);
        setInput('');
        setSending(true);
        setError(null);
        try {
            const resp = await axios.post(`${API_URL}/api/chat`, {
                messages:     next.slice(-10),
                audience,
                language:     lang,
                scan_context: scanContext,
            }, { timeout: 260000 });
            const reply = resp.data?.reply;
            if (reply) {
                setMessages(prev => [...prev, { role: 'assistant', content: reply,
                                                 refused: !!resp.data.refused }]);
            } else {
                setError(resp.data?.error || t('chat.error_generic'));
            }
        } catch (e) {
            setError(e.response?.data?.error || e.message || t('chat.error_generic'));
        } finally {
            setSending(false);
        }
    }, [input, sending, messages, audience, lang, scanContext, t]);


    const onKey = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            send();
        }
    };

    const PURPLE = '#a78bfa';
    const CYAN   = '#22d3ee';

    return (
        <Drawer
            anchor="right"
            open={open}
            onClose={onClose}
            PaperProps={{
                sx: {
                    width: { xs: '100%', sm: 420 },
                    background: 'linear-gradient(180deg, rgba(15,15,25,0.98) 0%, rgba(20,15,30,0.98) 100%)',
                    backdropFilter: 'blur(20px)',
                    borderLeft: `1px solid ${PURPLE}33`,
                },
            }}>
            <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                {/* Header */}
                <Box sx={{
                    p: 2, display: 'flex', alignItems: 'center', gap: 1.2,
                    borderBottom: '1px solid rgba(255,255,255,0.08)',
                }}>
                    <Box sx={{
                        width: 36, height: 36, borderRadius: '50%',
                        background: `linear-gradient(135deg, ${PURPLE}, ${CYAN})`,
                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                        boxShadow: `0 4px 14px ${PURPLE}55`,
                    }}>
                        <ChatIcon sx={{ color: '#fff', fontSize: 18 }} />
                    </Box>
                    <Box sx={{ flex: 1, minWidth: 0 }}>
                        <Typography sx={{ fontSize: 14, fontWeight: 700, color: '#fff', lineHeight: 1.1 }}>
                            {t('chat.title')}
                        </Typography>
                        <Typography sx={{ fontSize: 10.5, opacity: 0.65, lineHeight: 1.2 }}>
                            {scanContext
                                ? `${t('chat.context_loaded')} · ${scanContext.predicted_class || ''}`
                                : t('chat.no_context')}
                        </Typography>
                    </Box>
                    <IconButton onClick={onClose} size="small" sx={{ color: 'rgba(255,255,255,0.7)' }}>
                        <CloseIcon fontSize="small" />
                    </IconButton>
                </Box>

                {/* Audience toggle */}
                <Box sx={{ px: 2, py: 1.2, display: 'flex', gap: 0.5,
                           borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
                    {['patient', 'doctor'].map(a => {
                        const active = audience === a;
                        return (
                            <Button
                                key={a}
                                size="small"
                                disableRipple
                                onClick={() => setAudience(a)}
                                sx={{
                                    flex: 1, py: 0.5, textTransform: 'none',
                                    fontWeight: 700, fontSize: 11, letterSpacing: 0.3,
                                    color: active ? '#fff' : 'rgba(255,255,255,0.55)',
                                    background: active
                                        ? `linear-gradient(135deg, ${PURPLE}, ${CYAN})`
                                        : 'rgba(255,255,255,0.04)',
                                    borderRadius: 1.5,
                                    '&:hover': { background: active
                                        ? `linear-gradient(135deg, ${PURPLE}, ${CYAN})`
                                        : 'rgba(255,255,255,0.08)' },
                                }}>
                                {t(`chat.audience.${a}`)}
                            </Button>
                        );
                    })}
                </Box>

                {/* Messages list */}
                <Box ref={listRef} sx={{
                    flex: 1, overflowY: 'auto', p: 2,
                    display: 'flex', flexDirection: 'column', gap: 1.2,
                }}>
                    {messages.length === 0 && (
                        <Box sx={{
                            mt: 2, p: 2, borderRadius: 2,
                            background: 'rgba(167,139,250,0.06)',
                            border: '1px dashed rgba(167,139,250,0.3)',
                        }}>
                            <Typography sx={{ fontSize: 12.5, opacity: 0.85, lineHeight: 1.55 }}>
                                {t('chat.welcome')}
                            </Typography>
                        </Box>
                    )}
                    {messages.map((m, i) => (
                        <Box key={i} sx={{
                            alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
                            maxWidth: '85%',
                            p: 1.3, borderRadius: 2,
                            background: m.role === 'user'
                                ? `linear-gradient(135deg, ${PURPLE}33, ${CYAN}22)`
                                : (m.refused
                                    ? 'rgba(245,166,35,0.10)'
                                    : 'rgba(255,255,255,0.05)'),
                            border: `1px solid ${m.role === 'user'
                                ? `${PURPLE}55`
                                : (m.refused ? 'rgba(245,166,35,0.4)' : 'rgba(255,255,255,0.08)')}`,
                            fontSize: 13, lineHeight: 1.5,
                            color: '#fff',
                            whiteSpace: 'pre-wrap',
                        }}>
                            {m.content}
                        </Box>
                    ))}
                    {sending && (
                        <Box sx={{ alignSelf: 'flex-start', display: 'flex', alignItems: 'center', gap: 1,
                                   px: 1.5, py: 1, borderRadius: 2,
                                   background: 'rgba(255,255,255,0.04)' }}>
                            <CircularProgress size={12} sx={{ color: PURPLE }} />
                            <Typography sx={{ fontSize: 11.5, opacity: 0.7 }}>
                                {t('chat.thinking')}
                            </Typography>
                        </Box>
                    )}
                    {error && (
                        <Box sx={{ alignSelf: 'flex-start', p: 1.3, borderRadius: 2,
                                   background: 'rgba(245,87,108,0.10)',
                                   border: '1px solid rgba(245,87,108,0.4)',
                                   fontSize: 12, color: '#f5576c' }}>
                            {error}
                        </Box>
                    )}
                </Box>

                {/* Input */}
                <Box sx={{ p: 1.5, borderTop: '1px solid rgba(255,255,255,0.08)',
                           display: 'flex', gap: 1, alignItems: 'flex-end' }}>
                    <TextField
                        multiline
                        maxRows={4}
                        size="small"
                        fullWidth
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyDown={onKey}
                        placeholder={t('chat.placeholder')}
                        disabled={sending}
                        InputProps={{
                            sx: {
                                fontSize: 13,
                                background: 'rgba(255,255,255,0.04)',
                                borderRadius: 1.5,
                                color: '#fff',
                                '& fieldset': { borderColor: 'rgba(255,255,255,0.12)' },
                                '&:hover fieldset': { borderColor: 'rgba(167,139,250,0.4)' },
                            },
                        }}
                    />
                    <IconButton
                        onClick={send}
                        disabled={sending || !input.trim()}
                        sx={{
                            background: `linear-gradient(135deg, ${PURPLE}, ${CYAN})`,
                            color: '#fff',
                            '&:hover': { background: `linear-gradient(135deg, ${PURPLE}, ${CYAN})`, opacity: 0.9 },
                            '&.Mui-disabled': { background: 'rgba(255,255,255,0.08)', color: 'rgba(255,255,255,0.3)' },
                        }}>
                        <SendIcon fontSize="small" />
                    </IconButton>
                </Box>

                {/* Disclaimer */}
                <Box sx={{ px: 1.5, pb: 1.5 }}>
                    <Typography sx={{ fontSize: 9.5, opacity: 0.45, lineHeight: 1.4, fontStyle: 'italic' }}>
                        {t('chat.disclaimer')}
                    </Typography>
                </Box>
            </Box>
        </Drawer>
    );
}

function App() {
    const { t, lang } = useT();
    const [image, setImage] = useState(null);
    const [preview, setPreview] = useState(null);
    const [result, setResult] = useState(null);
    const [manualResult, setManualResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [view, setView] = useState('analysis');  // 'analysis' | 'metrics' | 'pipeline'
    const [prepMode, setPrepMode] = useState('standard');  // 'standard' | 'cross_scanner'
    const [reportMode, setReportMode] = useState('basic'); // 'basic' | 'advanced' (lifted from ReportCard)
    // When the user clicks "View in Brain Atlas" on a report, we stash the
    // target atlas region id here and switch view to 'atlas'. BrainAtlasPage
    // reads it on mount and pins that region, then clears the request.
    const [atlasPinRequest, setAtlasPinRequest] = useState(null);
    const goToAtlasRegion = useCallback((regionId) => {
        if (!regionId) return;
        setAtlasPinRequest(regionId);
        setView('atlas');
    }, []);
    // Floating MedGemma chat
    const [chatOpen, setChatOpen] = useState(false);

    const onDrop = (files) => {
        const file = files[0];
        setImage(file);
        setPreview(URL.createObjectURL(file));
        setResult(null);
        setManualResult(null);
        setError(null);
    };

    const handleSelectSample = useCallback((file) => {
        setImage(file);
        setPreview(URL.createObjectURL(file));
        setResult(null);
        setManualResult(null);
        setError(null);
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'image/*': ['.png', '.jpg', '.jpeg'] },
        multiple: false
    });

    const handleAnalyze = async () => {
        if (!image) return;
        setLoading(true);
        setError(null);
        try {
            const formData = new FormData();
            formData.append('image', image);
            formData.append('preprocessing_mode', prepMode);
            formData.append('language', lang);
            const response = await axios.post(`${API_URL}/api/predict`, formData, { timeout: 600000 });
            setResult(response.data);
        } catch (err) {
            setError(err.response?.data?.error || 'Analysis failed. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    const confClass = (c) => c > 0.85 ? 'confidence-high' : c > 0.65 ? 'confidence-medium' : 'confidence-low';
    const pct = (v) => `${(v * 100).toFixed(1)}%`;

    // If the user re-analyzed a manually-drawn region, merge the manual response
    // onto the base result so every prediction-related panel updates:
    //   • prediction header (class chip + confidence)
    //   • Model Comparison (per-model probs + agreement)
    //   • Uncertainty (MC Dropout σ / CI)
    //   • Malignancy card
    // XAI heatmaps + OOD + robustness keep their original values — they describe
    // the original full-image analysis, not the manual region.
    const r = result ? (manualResult ? {
        ...result,
        prediction:  manualResult.prediction,
        uncertainty: manualResult.uncertainty,
        models:      manualResult.models,
        agreement:   manualResult.agreement,
        best_model:  manualResult.best_model,
        malignancy:  manualResult.malignancy,
        manual_region: manualResult.manual_bbox_original,
    } : result) : null;
    const u = r?.uncertainty;

    const malignancy = r?.malignancy;
    const malignancyPredictedClass = r?.prediction?.class;
    const rob = r?.robustness;
    const xai = r?.xai?.levels;
    const crossDs = r?.cross_dataset;
    const mods = r?.models;
    const bestModel = r?.best_model;

    return (
        <div className="App">
            {/* Dither background */}
            <div style={{ position: 'fixed', top: 0, left: 0, width: '100%', height: '100vh', zIndex: 0 }}>
                <Dither waveColor={[0.61, 0.61, 0.61]} disableAnimation={false}
                    enableMouseInteraction={true} mouseRadius={0.3} colorNum={4}
                    waveAmplitude={0.3} waveFrequency={3} waveSpeed={0.05} />
            </div>

            <div style={{ position: 'relative', zIndex: 1 }}>
                {/* ── Header ── */}
                <AppBar position="static" className="modern-header" elevation={0}
                    sx={{
                        background: 'rgba(0,0,0,0.5)!important', backdropFilter: 'blur(30px) saturate(180%)',
                        WebkitBackdropFilter: 'blur(30px) saturate(180%)',
                        borderBottom: '1px solid rgba(255,255,255,0.15)', boxShadow: '0 4px 30px rgba(0,0,0,0.3)'
                    }}>
                    <Toolbar sx={{ minHeight: '48px !important', px: 2, display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: 1.5 }}>
                        <NavigationToggle view={view} setView={setView} />
                        <LanguageToggle />
                    </Toolbar>
                </AppBar>

                <Container maxWidth="xl" sx={{ py: 4 }}>
                    {error && <Alert severity="error" onClose={() => setError(null)} sx={{ mb: 2 }}>{error}</Alert>}

                    {view === 'metrics' && <MetricsPage />}
                    {view === 'atlas' && (
                        <BrainAtlasPage
                            pinRequest={atlasPinRequest}
                            clearPinRequest={() => setAtlasPinRequest(null)}
                        />
                    )}
                    {view === 'pipeline' && <PipelinePage />}

                    {view === 'analysis' && <>
                    {/* ═══ Row 1: Upload + Prediction ═══ */}
                    <Grid container spacing={4}>
                        {/* Upload panel */}
                        <Grid item xs={12} md={5}>
                            <Paper className="glass-card" elevation={0} sx={{ p: 4 }}>
                                <Typography variant="h5" gutterBottom sx={{ fontWeight: 700, color: '#667eea', mb: 3 }}>
                                    Upload MRI Scan</Typography>

                                <Box {...getRootProps()} className={`upload-zone ${isDragActive ? 'active' : ''}`}
                                    sx={{ p: 6, textAlign: 'center', minHeight: 350, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                    <input {...getInputProps()} />
                                    {preview
                                        ? <img src={preview} alt="Preview" className="image-preview"
                                            style={{ maxWidth: '100%', maxHeight: 300, objectFit: 'contain' }} />
                                        : <Box>
                                            <Typography variant="h6" sx={{ mb: 1, color: '#667eea', fontWeight: 600 }}>
                                                {isDragActive ? 'Drop it here!' : 'Upload MRI Image'}</Typography>
                                            <Typography variant="body2" color="text.secondary">
                                                Drag & drop or click to browse</Typography>
                                            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                                                Supports: JPG, PNG</Typography>
                                        </Box>}
                                </Box>
                                {image && <Box sx={{ mt: 3 }}>
                                    <Button fullWidth className="modern-button" variant="contained"
                                        onClick={handleAnalyze} disabled={loading} sx={{ py: 1.5 }}>
                                        {loading
                                            ? <><CircularProgress size={20} sx={{ color: 'white', mr: 1 }} />Analyzing...</>
                                            : 'Analyze MRI'}
                                    </Button>
                                </Box>}
                                {/* Single combined chart: per-model TTA probabilities for all models,
                                    plus MC Dropout spread + epistemic σ injected into the best-model row.
                                    Replaces the previous two cards (ModelComparison + ProbabilityBars). */}
                                <ModelComparisonChart r={r} />
                                {/* Patient-facing questions for the doctor — only shown after a tumor is detected */}
                                {r?.prediction?.class && (
                                    <QuestionsForDoctorCard predictedClass={r.prediction.class} />
                                )}
                                {/* Anatomy color-map grid — visible only when the Diagnostic
                                    Impression card is in Advanced mode. Click any thumbnail to
                                    zoom into a full-screen-ish dialog. */}
                                {/* Anatomy color-map grid — visible only when the Diagnostic
                                    Impression card is in Advanced mode. Click any thumbnail to
                                    zoom into a full-screen-ish dialog. */}
                                {reportMode === 'advanced' && r?.anatomy_views && (
                                    <AnatomyViewsGrid views={r.anatomy_views} />
                                )}
                            </Paper>
                        </Grid>

                        {/* Prediction */}
                        <Grid item xs={12} md={7}>
                            <Paper className="glass-card" elevation={0} sx={{ p: 4 }}>
                                <Typography variant="h5" gutterBottom sx={{ fontWeight: 700, color: '#667eea', mb: 3 }}>
                                    Analysis Results</Typography>

                                {r ? (
                                    <Box className="result-card">
                                        <TrustVerdictStrip r={r} />
                                        <SequenceWarningBanner r={r} />
                                        <TumorOODBanner r={r} />

                                        {/* Malignancy assessment — moved above MedGemma report.
                                            Crop image prefers the always-on tumor_crop_image (from malignancy.bbox),
                                            falling back to the Focus-Crop self-check crop if available. */}
                                        {malignancy && (
                                            <MalignancyCard
                                                data={malignancy}
                                                predictedClass={malignancyPredictedClass}
                                                cropImage={
                                                    malignancy.tumor_crop_image
                                                    || (r.focus_crop?.enabled && r.focus_crop?.crop_image ? r.focus_crop.crop_image : null)
                                                }
                                                cropBbox={
                                                    malignancy.tumor_crop_size
                                                        ? { w: malignancy.tumor_crop_size[0], h: malignancy.tumor_crop_size[1] }
                                                        : (r.focus_crop?.bbox_orig || null)
                                                }
                                                imageFile={image}
                                                previewUrl={preview}
                                                onManualResult={setManualResult}
                                            />
                                        )}

                                        <ReportCard r={r} mode={reportMode} setMode={setReportMode} onGoToAtlas={goToAtlasRegion} />

                                    </Box>
                                ) : (
                                    <Box sx={{ textAlign: 'center', py: 6, opacity: 0.7 }}>
                                        <Typography variant="h6" sx={{ mb: 1 }}>No Results Yet</Typography>
                                        <Typography variant="body2">Upload an MRI image and click Analyze</Typography>
                                    </Box>
                                )}
                            </Paper>
                        </Grid>
                    </Grid>

                    {/* ═══ Row 2: Uncertainty + Robustness ═══ */}
                    {r && (
                        <Grid container spacing={4} sx={{ mt: 0 }}>
                            {/* Uncertainty */}
                            {u && (
                                <Grid item xs={12} md={6}>
                                    <Paper className="glass-card" elevation={0} sx={{ p: 4 }}>
                                        <Typography variant="h6" sx={{ fontWeight: 700, color: '#667eea', mb: 2 }}>
                                            Novel #1: Uncertainty-Aware XAI (MC Dropout)</Typography>
                                        <Box sx={{ display: 'flex', justifyContent: 'space-around', mb: 3 }}>
                                            <Stat label="Epistemic" value={u.epistemic.toFixed(4)} color="#38ef7d" />
                                            <Stat label="Aleatoric" value={u.aleatoric.toFixed(4)} color="#f5a623" />
                                            <Stat label="Total" value={u.total_uncertainty.toFixed(4)}
                                                color={u.total_uncertainty > 0.15 ? '#f5576c' : '#38ef7d'} />
                                        </Box>
                                        <Box sx={{ background: 'rgba(255,255,255,0.1)', borderRadius: 2, p: 2, mb: 2 }}>
                                            <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                                                95% Confidence Interval</Typography>
                                            <Box sx={{
                                                position: 'relative', height: 28, borderRadius: 2, overflow: 'hidden',
                                                background: 'rgba(255,255,255,0.08)'
                                            }}>
                                                <Box sx={{
                                                    position: 'absolute', left: `${u.ci_lower * 100}%`,
                                                    width: `${(u.ci_upper - u.ci_lower) * 100}%`, height: '100%',
                                                    background: 'linear-gradient(90deg,#667eea,#764ba2)', borderRadius: 2, opacity: 0.9
                                                }} />
                                                <Typography variant="caption" sx={{
                                                    position: 'absolute', left: '50%',
                                                    top: '50%', transform: 'translate(-50%,-50%)', fontWeight: 600, zIndex: 1
                                                }}>
                                                    {pct(u.ci_lower)} — {pct(u.ci_upper)}</Typography>
                                            </Box>
                                        </Box>
                                        <Chip label={u.needs_review ? 'EXPERT REVIEW NEEDED' : 'HIGH CONFIDENCE'}
                                            sx={{
                                                fontWeight: 600, background: u.needs_review
                                                    ? 'linear-gradient(135deg,#f5576c,#fa709a)' : 'linear-gradient(135deg,#11998e,#38ef7d)',
                                                color: 'white'
                                            }} />
                                        {r.ood && typeof r.ood.energy_score !== 'undefined' && (
                                            <Box sx={{
                                                mt: 2, p: 1.5, borderRadius: 2,
                                                background: 'rgba(255,255,255,0.04)',
                                                border: `1px solid ${r.ood.is_ood ? 'rgba(245,87,108,0.45)' : 'rgba(56,239,125,0.30)'}`,
                                            }}>
                                                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 1 }}>
                                                    <Box>
                                                        <Typography variant="caption" sx={{ letterSpacing: 0.8, opacity: 0.65, fontSize: 10, textTransform: 'uppercase' }}>
                                                            Energy-based OOD score
                                                        </Typography>
                                                        <Typography variant="body2" sx={{ fontWeight: 700, color: '#fff' }}>
                                                            E(x) = {r.ood.energy_score}
                                                            <Box component="span" sx={{ opacity: 0.6, fontWeight: 400, fontSize: 12, ml: 1 }}>
                                                                (val p95 threshold = {r.ood.threshold_p95}, model: {r.ood.best_model})
                                                            </Box>
                                                        </Typography>
                                                    </Box>
                                                    <Chip
                                                        label={r.ood.is_ood ? 'OUT OF DISTRIBUTION' : 'In distribution'}
                                                        size="small"
                                                        sx={{
                                                            fontWeight: 700,
                                                            bgcolor: r.ood.is_ood ? 'rgba(245,87,108,0.25)' : 'rgba(56,239,125,0.20)',
                                                            color: r.ood.is_ood ? '#f5576c' : '#38ef7d',
                                                        }}
                                                    />
                                                </Box>
                                            </Box>
                                        )}
                                    </Paper>
                                </Grid>
                            )}

                            {/* Robustness */}
                            {rob && (
                                <Grid item xs={12} md={6}>
                                    <Paper className="glass-card" elevation={0} sx={{ p: 4 }}>
                                        <Typography variant="h6" sx={{ fontWeight: 700, color: '#667eea', mb: 2 }}>
                                            Novel #5: Adversarial Robustness Testing</Typography>
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 3, mb: 3 }}>
                                            <Box sx={{ textAlign: 'center' }}>
                                                <Typography variant="h3" sx={{
                                                    fontWeight: 700,
                                                    color: rob.overall_score >= 0.85 ? '#38ef7d' : '#f5a623'
                                                }}>
                                                    {(rob.overall_score * 100).toFixed(0)}%</Typography>
                                                <Typography variant="caption">Overall Score</Typography>
                                            </Box>
                                            <Chip label={rob.fda_ready ? 'FDA-Ready' : 'Needs Improvement'}
                                                sx={{
                                                    fontWeight: 600, background: rob.fda_ready
                                                        ? 'linear-gradient(135deg,#11998e,#38ef7d)' : 'linear-gradient(135deg,#f5576c,#fa709a)',
                                                    color: 'white'
                                                }} />
                                        </Box>
                                        {Object.entries(rob.tests).map(([name, t]) => (
                                            <Box key={name} sx={{
                                                display: 'flex', alignItems: 'center',
                                                justifyContent: 'space-between', mb: 1.5,
                                                p: 1.5, borderRadius: 2, background: 'rgba(255,255,255,0.08)'
                                            }}>
                                                <Typography variant="body2" sx={{ fontWeight: 500, textTransform: 'capitalize' }}>
                                                    {name.replace(/_/g, ' ')}</Typography>
                                                <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                                                    <Chip size="small" label={t.pred_stable ? 'Stable' : 'Changed'}
                                                        sx={{
                                                            background: t.pred_stable ? 'rgba(56,239,125,0.3)' : 'rgba(245,87,108,0.3)',
                                                            color: 'white', fontWeight: 600
                                                        }} />
                                                    <Typography variant="caption" sx={{ fontWeight: 600, minWidth: 50, textAlign: 'right' }}>
                                                        XAI: {(t.xai_stability * 100).toFixed(0)}%</Typography>
                                                </Box>
                                            </Box>
                                        ))}
                                    </Paper>
                                </Grid>
                            )}
                        </Grid>
                    )}

                    {/* ═══ Row 3: Hierarchical 4-Level XAI ═══ */}
                    {xai && (
                        <Paper className="glass-card" elevation={0} sx={{ p: 4, mt: 4 }}>
                            <Typography variant="h6" sx={{ fontWeight: 700, color: '#667eea', mb: 3 }}>
                                Novel #3: Hierarchical 4-Level XAI Framework</Typography>
                            <Grid container spacing={3}>
                                {Object.entries(xai).map(([key, level], idx) => (
                                    <Grid item xs={12} sm={6} md={3} key={key}>
                                        <Card className="xai-card" sx={{ height: '100%' }}>
                                            <CardContent sx={{ p: 2, textAlign: 'center' }}>
                                                <Chip label={`Level ${idx + 1}`} size="small"
                                                    sx={{ mb: 1, background: 'linear-gradient(135deg,#667eea,#764ba2)', color: 'white', fontWeight: 600 }} />
                                                <Typography variant="subtitle2" sx={{ fontWeight: 700, mb: 0.5 }}>
                                                    {level.name}</Typography>
                                                <Typography variant="caption" sx={{ display: 'block', opacity: 0.8, mb: 1.5 }}>
                                                    {level.question}</Typography>
                                                <CardMedia component="img"
                                                    image={`data:image/png;base64,${level.heatmap}`}
                                                    alt={level.name}
                                                    sx={{ borderRadius: 2, width: '100%' }} />
                                            </CardContent>
                                        </Card>
                                    </Grid>
                                ))}
                            </Grid>
                            <Typography variant="caption" sx={{ mt: 2, display: 'block', opacity: 0.7, textAlign: 'center' }}>
                                From broad detection (L1) to deep feature analysis (L4) — providing multi-granularity visual explanations</Typography>
                        </Paper>
                    )}

                    {/* ═══ Row 4: Cross-Dataset ═══ */}
                    {crossDs && (
                        <Paper className="glass-card" elevation={0} sx={{ p: 4, mt: 4 }}>
                            <Typography variant="h6" sx={{ fontWeight: 700, color: '#667eea', mb: 2 }}>
                                Novel #2: Cross-Dataset Generalization</Typography>
                            <Grid container spacing={3}>
                                <Grid item xs={12} sm={4}>
                                    <Box sx={{ textAlign: 'center', p: 2, borderRadius: 2, background: 'rgba(255,255,255,0.08)' }}>
                                        <Typography variant="h4" sx={{ fontWeight: 700, color: '#38ef7d' }}>
                                            {crossDs.brisc2025_test_acc}%</Typography>
                                        <Typography variant="body2">BRISC-2025 Test Accuracy</Typography>
                                    </Box>
                                </Grid>
                                <Grid item xs={12} sm={4}>
                                    <Box sx={{ textAlign: 'center', p: 2, borderRadius: 2, background: 'rgba(255,255,255,0.08)' }}>
                                        <Typography variant="h4" sx={{ fontWeight: 700, color: '#667eea' }}>
                                            {crossDs.mendeley_test_acc}%</Typography>
                                        <Typography variant="body2">Mendeley Cross-Dataset</Typography>
                                    </Box>
                                </Grid>
                                <Grid item xs={12} sm={4}>
                                    <Box sx={{ textAlign: 'center', p: 2, borderRadius: 2, background: 'rgba(255,255,255,0.08)' }}>
                                        <Chip label="VALIDATED" sx={{
                                            fontWeight: 700, mb: 1,
                                            background: 'linear-gradient(135deg,#11998e,#38ef7d)', color: 'white'
                                        }} />
                                        <Typography variant="body2" sx={{ display: 'block' }}>
                                            {crossDs.datasets_tested} Datasets Tested</Typography>
                                    </Box>
                                </Grid>
                            </Grid>
                        </Paper>
                    )}

                    {/* ═══ Row 5: Sample Data Gallery ═══ */}
                    <Paper className="glass-card" elevation={0} sx={{ p: 4, mt: 4 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 1 }}>
                            <ScienceIcon sx={{ color: '#667eea', fontSize: 28 }} />
                            <Typography variant="h5" sx={{ fontWeight: 700, color: '#667eea' }}>
                                Test with Sample Data</Typography>
                        </Box>
                        <Typography variant="body2" sx={{ opacity: 0.7, mb: 3 }}>
                            Don't have an MRI scan? Select a sample image below from any of our 3 validated datasets.
                            Click the <ScienceIcon sx={{ fontSize: 12, verticalAlign: 'middle', mx: 0.3 }} /> icon to load it for analysis,
                            or <DownloadIcon sx={{ fontSize: 12, verticalAlign: 'middle', mx: 0.3 }} /> to download.
                        </Typography>
                        <SampleGallery onSelectSample={handleSelectSample} />
                    </Paper>

                    {/* ═══ About ═══ */}
                    <Paper className="info-card" elevation={0} sx={{ p: 4, mt: 4, mb: 3 }}>
                        <Typography variant="h5" gutterBottom sx={{ fontWeight: 700, color: '#667eea', mb: 1.5 }}>
                            Why Trust This System?</Typography>
                        <Typography variant="body1" sx={{ lineHeight: 1.8, opacity: 0.85, mb: 4, maxWidth: 820 }}>
                            A lot of AI tools for reading medical scans work like a <strong>black box</strong>: they
                            give you a label — "tumor" or "no tumor" — with no way to check their work and no idea
                            of how sure they really are. This system was built to fix that, in four concrete ways.
                        </Typography>

                        {/* Headline stat tiles */}
                        <Grid container spacing={2.5} sx={{ mb: 4 }}>
                            {[
                                { value: '99', suffix: '/ 100', accent: '#38ef7d',
                                  label: 'Correct on scans it had never seen',
                                  sub: 'Tested against 2,114 held-out MRI scans' },
                                { value: '95', suffix: '/ 100', accent: '#22d3ee',
                                  label: 'Still correct on a different hospital’s scans',
                                  sub: 'Tested on equipment and patients outside its training data' },
                            ].map((s, i) => (
                                <Grid item xs={12} sm={6} key={i}>
                                    <Box sx={{
                                        p: 3, borderRadius: 3, height: '100%',
                                        background: `linear-gradient(135deg, ${s.accent}22 0%, ${s.accent}08 100%)`,
                                        border: `1px solid ${s.accent}55`,
                                        display: 'flex', alignItems: 'center', gap: 2.5,
                                    }}>
                                        <Typography sx={{ fontSize: 44, fontWeight: 800, color: s.accent, lineHeight: 1,
                                                          whiteSpace: 'nowrap' }}>
                                            {s.value}<Typography component="span" sx={{ fontSize: 20, fontWeight: 700, opacity: 0.7 }}> {s.suffix}</Typography>
                                        </Typography>
                                        <Box>
                                            <Typography sx={{ fontWeight: 700, fontSize: 15, lineHeight: 1.3 }}>{s.label}</Typography>
                                            <Typography sx={{ fontSize: 12.5, opacity: 0.65, mt: 0.4 }}>{s.sub}</Typography>
                                        </Box>
                                    </Box>
                                </Grid>
                            ))}
                        </Grid>

                        {/* Four trust pillars */}
                        <Grid container spacing={2.5}>
                            {[
                                { icon: GroupsIcon, accent: '#667eea', title: 'Three Models, Not One',
                                  body: 'Every scan is checked independently by three separate AI models. They have to agree before the app shows a confident result — if they disagree, it tells you instead of quietly picking one.' },
                                { icon: VisibilityIcon, accent: '#a78bfa', title: 'Shows Its Work',
                                  body: 'A heat-map highlights exactly which part of the brain the model based its answer on, so you can see it’s actually looking at the tumor — not somewhere random.' },
                                { icon: WarningAmberIcon, accent: '#f5a623', title: 'Knows When It’s Unsure',
                                  body: 'A built-in uncertainty check flags scans that need a second look from a doctor, rather than forcing out a confident-sounding guess when the evidence is thin.' },
                                { icon: ShieldIcon, accent: '#38ef7d', title: 'Stress-Tested',
                                  body: 'Deliberately tested against blurry, noisy, and low-quality images to make sure a bad photo or a different scanner doesn’t quietly change the answer.' },
                            ].map((p, i) => (
                                <Grid item xs={12} sm={6} key={i}>
                                    <Box sx={{
                                        p: 2.75, borderRadius: 3, height: '100%',
                                        background: 'rgba(255,255,255,0.03)',
                                        border: '1px solid rgba(255,255,255,0.09)',
                                        display: 'flex', gap: 2, alignItems: 'flex-start',
                                    }}>
                                        <Box sx={{
                                            width: 42, height: 42, borderRadius: 2, flexShrink: 0,
                                            background: `${p.accent}22`, border: `1px solid ${p.accent}55`,
                                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                                        }}>
                                            <p.icon sx={{ color: p.accent, fontSize: 22 }} />
                                        </Box>
                                        <Box>
                                            <Typography sx={{ fontWeight: 700, fontSize: 15, mb: 0.5 }}>{p.title}</Typography>
                                            <Typography sx={{ fontSize: 13, opacity: 0.75, lineHeight: 1.6 }}>{p.body}</Typography>
                                        </Box>
                                    </Box>
                                </Grid>
                            ))}
                        </Grid>

                        <Typography variant="body2" sx={{ opacity: 0.55, mt: 3.5, fontStyle: 'italic' }}>
                            Multiple models cross-checking each other, visible reasoning, honest uncertainty, and
                            testing on scans and hospitals it had never encountered — this is the kind of scrutiny
                            most AI scan-readers skip.
                        </Typography>
                    </Paper>
                    </>}

                </Container>
            </div>

            {/* Floating MedGemma chat FAB + Drawer */}
            <Fab
                onClick={() => setChatOpen(true)}
                aria-label="chat"
                sx={{
                    position: 'fixed', bottom: 24, right: 24, zIndex: 1200,
                    background: 'linear-gradient(135deg, #a78bfa 0%, #22d3ee 100%)',
                    color: '#fff',
                    boxShadow: '0 8px 32px rgba(167,139,250,0.45)',
                    '&:hover': {
                        background: 'linear-gradient(135deg, #a78bfa 0%, #22d3ee 100%)',
                        opacity: 0.92,
                        transform: 'scale(1.06)',
                    },
                    transition: 'transform .15s, opacity .15s',
                }}>
                <ChatIcon />
            </Fab>
            <ChatbotPanel
                open={chatOpen}
                onClose={() => setChatOpen(false)}
                scanResult={result}
            />
        </div>
    );
}

export default App;
